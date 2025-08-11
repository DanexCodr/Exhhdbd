

import os, sys, json, traceback
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

ONNX_FLOAT = TensorProto.FLOAT
ONNX_INT64 = TensorProto.INT64
ONNX_INT32 = TensorProto.INT32

def get_opset_version(model):
    # Prefer the default-domain ('') opset import; fallback to max present
    for oi in model.opset_import:
        if oi.domain == "" or oi.domain == "ai.onnx":
            return oi.version
    # fallback
    return max((oi.version for oi in model.opset_import), default=0)

def safe_name(a): return getattr(a, "name", None)

def fix_repeated_field(attr, field_name):
    if hasattr(attr, field_name):
        val = getattr(attr, field_name)
        if val is None:
            setattr(attr, field_name, [])
    return attr

def ensure_common_defaults(node, opset_v):
    """Fill common missing attributes for ops that often break onnx-tf conversion.
       Respect opset: do NOT add old-style Slice attrs when opset >= 10.
    """
    # Cast -> 'to' (dtype)
    if node.op_type == "Cast":
        if not any(safe_name(a) == "to" for a in node.attribute):
            node.attribute.append(helper.make_attribute("to", TensorProto.FLOAT))

    # Concat -> 'axis'
    if node.op_type == "Concat":
        if not any(safe_name(a) == "axis" for a in node.attribute):
            node.attribute.append(helper.make_attribute("axis", 1))

    # Reduce* -> keepdims + axes default
    if node.op_type.startswith("Reduce"):
        if not any(safe_name(a) == "keepdims" for a in node.attribute):
            node.attribute.append(helper.make_attribute("keepdims", 1))
        if not any(safe_name(a) == "axes" for a in node.attribute):
            node.attribute.append(helper.make_attribute("axes", []))

    # Slice: **only** add attribute defaults for old opsets (<10).
    # For opset >= 10, Slice uses inputs for starts/ends/etc — adding attributes breaks the model.
    if node.op_type == "Slice":
        if opset_v < 10:
            if not any(safe_name(a) == "starts" for a in node.attribute):
                node.attribute.append(helper.make_attribute("starts", []))
            if not any(safe_name(a) == "ends" for a in node.attribute):
                node.attribute.append(helper.make_attribute("ends", []))
            if not any(safe_name(a) == "axes" for a in node.attribute):
                node.attribute.append(helper.make_attribute("axes", []))
        else:
            # opset >= 10: do not inject starts/ends attributes
            pass

def clean_node_attributes(model, opset_v):
    for node in model.graph.node:
        new_attrs = []
        for a in list(node.attribute):
            if a is None:
                continue
            fix_repeated_field(a, "ints")
            fix_repeated_field(a, "floats")
            fix_repeated_field(a, "strings")
            if hasattr(a, "s") and a.s is None:
                a.s = b""
            if hasattr(a, "i") and a.i is None:
                a.i = 0
            new_attrs.append(a)
        del node.attribute[:]
        node.attribute.extend(new_attrs)
        ensure_common_defaults(node, opset_v)

def add_value_info_for_missing_outputs(model):
    present = {vi.name for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
    added = []
    for node in model.graph.node:
        dtype = ONNX_FLOAT
        if node.op_type in ("Shape", "Range"):
            dtype = ONNX_INT64
        for o in node.output:
            if not o:
                continue
            if o not in present:
                vi = helper.make_tensor_value_info(o, dtype, [1])
                model.graph.value_info.append(vi)
                present.add(o)
                added.append((o, node.op_type))
    return added

def gather_names(model):
    names = set()
    for init in model.graph.initializer:
        names.add(init.name)
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        names.add(vi.name)
    for node in model.graph.node:
        for out in node.output:
            if out:
                names.add(out)
    return names

def find_missing_inputs(model):
    present = gather_names(model)
    missing = []
    for node in model.graph.node:
        for inp in node.input:
            if not inp:
                continue
            if inp not in present:
                missing.append(inp)
    return sorted(set(missing))

def inject_dummy_init(model, name, dtype=ONNX_FLOAT):
    if dtype == ONNX_FLOAT:
        arr = np.zeros((1,), dtype=np.float32)
    else:
        arr = np.zeros((1,), dtype=np.int64)
    tensor = numpy_helper.from_array(arr, name)
    model.graph.initializer.append(tensor)
    try:
        vi = helper.make_tensor_value_info(name, dtype, [1])
        model.graph.input.append(vi)
    except Exception:
        pass
    return name

def minimal_for_node(model, idx):
    node = model.graph.node[idx]
    inits = {i.name: i for i in model.graph.initializer}
    new_inits = []
    new_inputs = []
    for inp in node.input:
        if not inp:
            continue
        if inp in inits:
            init = inits[inp]
            new_inits.append(init)
            try:
                new_inputs.append(helper.make_tensor_value_info(inp, init.data_type, list(init.dims)))
            except Exception:
                new_inputs.append(helper.make_tensor_value_info(inp, TensorProto.FLOAT, [1]))
        else:
            new_inputs.append(helper.make_tensor_value_info(inp, TensorProto.FLOAT, [1]))
    new_outputs = []
    for o in node.output:
        if o:
            new_outputs.append(helper.make_tensor_value_info(o, TensorProto.FLOAT, [1]))
    new_graph = helper.make_graph([node], f"node_{idx}_graph", new_inputs, new_outputs, initializer=new_inits)
    return helper.make_model(new_graph, producer_name="isolated_node")

def isolate_failing_node(model, max_nodes=200):
    print("Running per-node isolation (this may take a while)...")
    for idx, node in enumerate(model.graph.node):
        if idx > max_nodes:
            break
        try:
            small = minimal_for_node(model, idx)
            try:
                onnx.checker.check_model(small)
            except Exception:
                pass
            prepare(small, strict=False)
        except Exception as e:
            info = {
                "index": idx,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": [{ "name": safe_name(a), "raw": str(a) } for a in node.attribute],
                "error": traceback.format_exc()
            }
            fn = f"failing_node_{idx}.onnx"
            onnx.save(small, fn)
            open("diagnostics.txt", "w").write(json.dumps(info, indent=2))
            print(f"Found failing node #{idx}: op={node.op_type} name='{node.name}' saved -> {fn}")
            return info
    print("No failing single-node repro found.")
    return None

def main():
    AUTO_FIX = os.environ.get("AUTO_FIX", "0") in ("1", "true", "True")
    infile = os.environ.get("INPUT_ONNX", "model_simplified.onnx")
    print("Input ONNX:", infile)

    model = onnx.load(infile)

    try:
        print("Running shape inference...")
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print("shape inference failed (continuing):", e)

    opset_v = get_opset_version(model)
    print("Detected opset version:", opset_v)

    print("Cleaning node attributes and injecting defaults (opset-aware)...")
    clean_node_attributes(model, opset_v)

    added = add_value_info_for_missing_outputs(model)
    if added:
        print("Added value_info for outputs:", added)

    cleaned = "model_cleaned_for_tf.onnx"
    onnx.save(model, cleaned)
    print("Saved cleaned model as", cleaned)

    try:
        print("Running ONNX checker...")
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print("ONNX checker failure:", e)
        open("diagnostics.txt", "w").write(str(e))
        sys.exit(1)

    missing = find_missing_inputs(model)
    if missing:
        print("Missing inputs referenced by nodes:", missing)
        if AUTO_FIX:
            print("AUTO_FIX enabled — injecting tiny dummies for missing inputs")
            for m in missing:
                inject_dummy_init(model, m)
            onnx.save(model, "model_cleaned_with_dummies.onnx")
        else:
            open("diagnostics.txt", "w").write(json.dumps({"missing_inputs": missing}, indent=2))
            print("Missing inputs found — re-run with AUTO_FIX=1 to auto-inject dummies.")
            sys.exit(1)
    else:
        print("No missing inputs detected.")

    try:
        print("Preparing onnx-tf representation (prepare)...")
        tf_rep = prepare(model, strict=False)
        print("Exporting SavedModel -> 'saved_model' ...")
        tf_rep.export_graph("saved_model")
    except Exception as e:
        print("ONNX->TensorFlow conversion failed (export stage):", e)
        info = isolate_failing_node(model)
        if info:
            print("Isolation produced diagnostics.txt and failing_node_*.onnx. Attach them or paste here for help.")
        else:
            open("error.txt", "w").write(str(e))
            print("No isolation info; see error.txt")
        sys.exit(1)

    try:
        print("Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        tflite_model = converter.convert()
        with open("model.tflite", "wb") as f:
            f.write(tflite_model)
        print("Wrote model.tflite")
    except Exception as e:
        print("SavedModel->TFLite failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
