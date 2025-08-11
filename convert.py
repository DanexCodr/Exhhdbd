#!/usr/bin/env python3
"""
Autopatching ONNX -> TensorFlow -> TFLite converter with extra sanitizers.

This script:
 - applies fixes for Cast/Concat/Slice/Unsqueeze/Reduce nodes (as before)
 - sanitizes initializers (replaces empty/invalid arrays)
 - fills unknown dims in value_info / graph inputs with 1
 - saves patched ONNX, exports TF SavedModel and converts to TFLite
"""

import os
import uuid
import shutil
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import json
import traceback

# constants
FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64
INPUT_ONNX = "model_simplified.onnx"
PATCHED_ONNX = "model_patched.onnx"
TF_MODEL_DIR = "tf_model"
OUTPUT_TFLITE = "model.tflite"

# ---------- helpers ----------
def get_opset_version(model):
    for oi in model.opset_import:
        if oi.domain == "" or oi.domain == "ai.onnx":
            try:
                return int(oi.version)
            except Exception:
                return 0
    try:
        return int(model.opset_import[0].version)
    except Exception:
        return 0

def unique_name(base):
    return f"{base}_{uuid.uuid4().hex[:8]}"

def name_in_initializers(model, name):
    return any(init.name == name for init in model.graph.initializer)

def append_initializer_if_missing(model, tensor_proto):
    if not name_in_initializers(model, tensor_proto.name):
        model.graph.initializer.append(tensor_proto)

# ---------- existing autopatch fixes (kept) ----------
def fix_cast_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Cast":
            if not any(getattr(a, "name", None) == "to" for a in node.attribute):
                print(f"Fixing {node.name or 'Cast'}: setting to=FLOAT")
                node.attribute.append(helper.make_attribute("to", FLOAT))

def fix_concat_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Concat":
            if not any(getattr(a, "name", None) == "axis" for a in node.attribute):
                print(f"Fixing {node.name or 'Concat'}: setting axis=-1")
                node.attribute.append(helper.make_attribute("axis", -1))

def fix_slice_nodes(model):
    opset_version = get_opset_version(model)
    for node in list(model.graph.node):
        if node.op_type != "Slice":
            continue
        if opset_version >= 10:
            attr_map = {getattr(a, "name", None): a for a in node.attribute}
            starts = list(attr_map.get("starts").ints) if "starts" in attr_map and hasattr(attr_map.get("starts"), "ints") else []
            ends   = list(attr_map.get("ends").ints)   if "ends"   in attr_map and hasattr(attr_map.get("ends"), "ints")   else []
            axes   = list(attr_map.get("axes").ints)   if "axes"   in attr_map and hasattr(attr_map.get("axes"), "ints")   else []
            steps  = list(attr_map.get("steps").ints)  if "steps"  in attr_map and hasattr(attr_map.get("steps"), "ints")  else []

            # if everything empty, create conservative minimal values
            if not starts and not ends and not axes and not steps:
                starts = [0]
                ends = [9223372036854775807]
                axes = [0]
                steps = [1]
                print(f"Fixing {node.name or 'Slice'}: injecting minimal starts/ends/axes/steps")

            # keep only data input, then append const inputs in order
            if len(node.input) == 0:
                # defensive: can't do anything
                continue
            data_input = node.input[0]
            node.input[:] = [data_input]

            for name_base, vals in (("starts", starts), ("ends", ends), ("axes", axes), ("steps", steps)):
                const_name = unique_name(f"{node.name or 'Slice'}_{name_base}_const")
                # dims length 0 allowed; use [0] dims for empty -> some checkers accept 0-len
                dims = [len(vals)] if len(vals) > 0 else [0]
                tensor = helper.make_tensor(name=const_name, data_type=INT64, dims=dims, vals=vals)
                append_initializer_if_missing(model, tensor)
                node.input.append(const_name)
                print(f"Fixing {node.name or 'Slice'}: adding {name_base}={vals} as input tensor named {const_name}")

            # remove attribute representation
            del node.attribute[:]
        else:
            for attr_name in ("starts", "ends", "axes", "steps"):
                if not any(getattr(a, "name", None) == attr_name for a in node.attribute):
                    node.attribute.append(helper.make_attribute(attr_name, []))

def fix_unsqueeze_nodes(model):
    opset_version = get_opset_version(model)
    for node in model.graph.node:
        if node.op_type == "Unsqueeze":
            if opset_version < 13:
                if not any(getattr(a, "name", None) == "axes" for a in node.attribute):
                    node.attribute.append(helper.make_attribute("axes", [0]))
            else:
                if len(node.input) == 1:
                    axes_name = unique_name(f"{node.name or 'Unsqueeze'}_axes_const")
                    axes_tensor = helper.make_tensor(name=axes_name, data_type=INT64, dims=[1], vals=[0])
                    append_initializer_if_missing(model, axes_tensor)
                    node.input.append(axes_name)
                    print(f"Fixing {node.name or 'Unsqueeze'}: added axes constant {axes_name}")

def fix_reduce_nodes(model):
    reduce_ops = ("ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin")
    for node in model.graph.node:
        if node.op_type in reduce_ops:
            if not any(getattr(a, "name", None) == "keepdims" for a in node.attribute):
                node.attribute.append(helper.make_attribute("keepdims", 1))

# ---------- NEW: sanitizers to prevent None propagation ----------
def sanitize_initializers(model):
    """
    Ensure no initializer has None values or weird dtypes/dims.
    Replace empty initializers with a tiny valid tensor (0 or single-element) and
    convert object-like arrays to numeric arrays.
    """
    changed = 0
    new_inits = []
    for init in list(model.graph.initializer):
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            arr = None

        if arr is None:
            # replace with single zero float32
            print(f"Sanitizing initializer '{init.name}': unable to read -> replacing with [0.0]")
            new_init = numpy_helper.from_array(np.array([0.0], dtype=np.float32), init.name)
            new_inits.append(new_init)
            changed += 1
            continue

        # if any element is None or nan or infinite, replace them with 0
        if np.isnan(arr).any() or np.isneginf(arr).any() or np.isposinf(arr).any():
            print(f"Sanitizing initializer '{init.name}': contains NaN/inf -> replacing invalid entries with 0")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            new_inits.append(numpy_helper.from_array(arr, init.name))
            changed += 1
            continue

        # if empty (size 0), replace with small default of shape (1,)
        if arr.size == 0:
            dtype = np.float32 if init.data_type == TensorProto.FLOAT else np.int64
            substitute = np.zeros((1,), dtype=dtype)
            print(f"Sanitizing initializer '{init.name}': empty -> replacing with zeros shape (1,)")
            new_inits.append(numpy_helper.from_array(substitute, init.name))
            changed += 1
            continue

        # otherwise keep as-is
        new_inits.append(init)

    if changed:
        # replace initializer list
        model.graph.initializer[:] = new_inits
        print(f"Sanitized {changed} initializers.")
    return changed

def fill_unknown_value_info_dims(model):
    """
    Replace unknown/None dims in graph inputs/value_info/output with 1 to avoid None
    propagating into tf.make_tensor_proto.
    """
    changed = 0
    def fix_vi(vi):
        nonlocal changed
        if not vi.type.HasField("tensor_type"):
            return
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return
        for d in tt.shape.dim:
            if d.HasField("dim_param"):
                # named param -> keep but also set dim_value=1 for TF friendliness
                d.dim_value = 1
                d.ClearField("dim_param")
                changed += 1
            elif not d.HasField("dim_value") or d.dim_value <= 0:
                d.dim_value = 1
                changed += 1

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        try:
            fix_vi(vi)
        except Exception:
            pass

    if changed:
        print(f"Filled {changed} unknown dims in value_info/input/output with 1.")
    return changed

# ---------- diagnostics helpers ----------
def dump_diagnostics(model, filename="diagnostics.json"):
    info = {
        "opset": get_opset_version(model),
        "num_nodes": len(model.graph.node),
        "initializers": [{ "name": i.name, "dims": list(i.dims), "dtype": int(i.data_type)} for i in model.graph.initializer],
        "inputs": [ (vi.name, getattr(vi.type.tensor_type, "shape", None) and [ (d.dim_value if d.HasField("dim_value") else None) for d in vi.type.tensor_type.shape.dim ]) for vi in model.graph.input ],
    }
    open(filename, "w").write(json.dumps(info, indent=2))

# ---------- main pipeline ----------
def autopatch_model(model):
    # original fixes
    fix_reduce_nodes(model)
    fix_cast_nodes(model)
    fix_concat_nodes(model)
    fix_slice_nodes(model)
    fix_unsqueeze_nodes(model)

def main():
    if not os.path.exists(INPUT_ONNX):
        print("Input ONNX not found:", INPUT_ONNX)
        return 2

    print("Loading ONNX model:", INPUT_ONNX)
    model = onnx.load(INPUT_ONNX)

    print("Detected opset:", get_opset_version(model))
    print("Applying autopatches...")
    try:
        autopatch_model(model)
    except Exception as e:
        print("Autopatch failed:", e)
        traceback.print_exc()
        dump_diagnostics(model, "diagnostics_pre_autopatch.json")
        raise

    # sanitizers (new)
    try:
        s1 = sanitize_initializers(model)
        s2 = fill_unknown_value_info_dims(model)
        if s1 or s2:
            print("Sanitizers made changes; saving intermediate patched onnx.")
    except Exception as e:
        print("Sanitizers failed:", e)
        traceback.print_exc()
        dump_diagnostics(model, "diagnostics_sanitizer_failure.json")
        raise

    print("Saving patched ONNX ->", PATCHED_ONNX)
    onnx.save(model, PATCHED_ONNX)

    print("Preparing onnx-tf representation (prepare)...")
    try:
        tf_rep = prepare(model, strict=False)
    except Exception as e:
        print("onnx-tf prepare() failed:", e)
        traceback.print_exc()
        dump_diagnostics(model, "diagnostics_prepare_fail.json")
        raise

    if os.path.exists(TF_MODEL_DIR):
        shutil.rmtree(TF_MODEL_DIR)
    tf_rep.export_graph(TF_MODEL_DIR)

    print("Converting SavedModel -> TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_DIR)
        tflite_model = converter.convert()
        with open(OUTPUT_TFLITE, "wb") as f:
            f.write(tflite_model)
        print("Wrote", OUTPUT_TFLITE)
    except Exception as e:
        print("TFLite conversion failed:", e)
        traceback.print_exc()
        return 1

    print("Conversion pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
