#!/usr/bin/env python3
"""
Autopatching ONNX -> TensorFlow -> TFLite converter.

Fixes common issues for ONNX->TF conversion:
 - missing Cast 'to' attribute
 - missing Concat 'axis'
 - Slice expressed with attributes -> convert to input-based (opset >= 10)
 - Unsqueeze axes handling (attributes for older opsets, input tensor for newer)
 - Reduce* ensure keepdims and axes where appropriate

Saves a patched ONNX (model_patched.onnx), exports a TF SavedModel and converts to TFLite.
"""

import os
import onnx
from onnx import helper
import tensorflow as tf
from onnx_tf.backend import prepare
import numpy as np
import uuid
import shutil

# ====== Default constants ======
FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64
INPUT_ONNX = "model_simplified.onnx"
PATCHED_ONNX = "model_patched.onnx"
TF_MODEL_DIR = "tf_model"
OUTPUT_TFLITE = "model.tflite"


def get_opset_version(model):
    for oi in model.opset_import:
        if oi.domain == "" or oi.domain == "ai.onnx":
            return int(oi.version)
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
    """For opset >= 10 convert Slice attributes to constant inputs (starts/ends/axes/steps)."""
    opset_version = get_opset_version(model)
    for node in list(model.graph.node):
        if node.op_type != "Slice":
            continue

        if opset_version >= 10:
            attr_map = {getattr(a, "name", None): a for a in node.attribute}
            starts = list(attr_map.get("starts").ints) if "starts" in attr_map else []
            ends = list(attr_map.get("ends").ints) if "ends" in attr_map else []
            axes = list(attr_map.get("axes").ints) if "axes" in attr_map else []
            steps = list(attr_map.get("steps").ints) if "steps" in attr_map else []

            if not starts and not ends and not axes and not steps:
                starts = [0]
                ends = [9223372036854775807]
                axes = [0]
                steps = [1]
                print(f"Fixing {node.name or 'Slice'}: injecting minimal starts/ends/axes/steps")

            data_input = node.input[0]
            node.input[:] = [data_input]

            for name_base, vals in (
                ("starts", starts),
                ("ends", ends),
                ("axes", axes),
                ("steps", steps)
            ):
                const_name = unique_name(f"{node.name or 'Slice'}_{name_base}_const")
                tensor = helper.make_tensor(
                    name=const_name,
                    data_type=INT64,
                    dims=[len(vals)] if vals else [0],
                    vals=vals
                )
                append_initializer_if_missing(model, tensor)
                node.input.append(const_name)
                print(f"Fixing {node.name or 'Slice'}: adding {name_base}={vals} as input tensor named {const_name}")

            del node.attribute[:]

        else:
            for attr_name in ("starts", "ends", "axes", "steps"):
                if not any(getattr(a, "name", None) == attr_name for a in node.attribute):
                    print(f"Fixing {node.name or 'Slice'} (opset<{opset_version}): adding {attr_name}=[]")
                    node.attribute.append(helper.make_attribute(attr_name, []))


def fix_unsqueeze_nodes(model):
    opset_version = get_opset_version(model)
    for node in model.graph.node:
        if node.op_type == "Unsqueeze":
            if opset_version < 13:
                if not any(getattr(a, "name", None) == "axes" for a in node.attribute):
                    print(f"Fixing {node.name or 'Unsqueeze'}: setting axes=[0]")
                    node.attribute.append(helper.make_attribute("axes", [0]))
            else:
                if len(node.input) == 1:
                    axes_name = unique_name(f"{node.name or 'Unsqueeze'}_axes_const")
                    axes_tensor = helper.make_tensor(
                        name=axes_name,
                        data_type=INT64,
                        dims=[1],
                        vals=[0]
                    )
                    append_initializer_if_missing(model, axes_tensor)
                    node.input.append(axes_name)
                    print(f"Fixing {node.name or 'Unsqueeze'}: adding axes=[0] as input tensor")


def fix_reduce_nodes(model):
    reduce_ops = ("ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin")
    for node in model.graph.node:
        if node.op_type in reduce_ops:
            if not any(getattr(a, "name", None) == "keepdims" for a in node.attribute):
                print(f"Fixing {node.name or node.op_type}: setting keepdims=1")
                node.attribute.append(helper.make_attribute("keepdims", 1))


def autopatch_model(model):
    print("Cleaning model...")
    fix_reduce_nodes(model)
    print("Fixing Cast ops...")
    fix_cast_nodes(model)
    print("Fixing Concat ops...")
    fix_concat_nodes(model)
    print("Fixing Slice ops...")
    fix_slice_nodes(model)
    print("Fixing Unsqueeze ops...")
    fix_unsqueeze_nodes(model)


def save_model(model, path):
    onnx.save(model, path)
    print(f"Saved patched ONNX to: {path}")


def main():
    if not os.path.exists(INPUT_ONNX):
        print(f"Input ONNX not found: {INPUT_ONNX}")
        return

    print(f"Loading ONNX model from: {INPUT_ONNX}")
    model = onnx.load(INPUT_ONNX)
    opset_v = get_opset_version(model)
    print("Detected opset version:", opset_v)

    autopatch_model(model)
    save_model(model, PATCHED_ONNX)

    print("Converting ONNX → TensorFlow...")
    try:
        tf_rep = prepare(model, strict=False)
    except Exception as e:
        print("onnx-tf prepare() failed:", e)
        raise

    if os.path.exists(TF_MODEL_DIR):
        shutil.rmtree(TF_MODEL_DIR)
    tf_rep.export_graph(TF_MODEL_DIR)

    print("Converting TensorFlow → TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_DIR)
    tflite_model = converter.convert()
    with open(OUTPUT_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Conversion complete! Saved TFLite model as {OUTPUT_TFLITE}")


if __name__ == "__main__":
    main()
