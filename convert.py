

import os, sys, json, traceback
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

# --- helpers -----------------------------------------------------------------
ONNX_FLOAT = TensorProto.FLOAT
ONNX_INT64 = TensorProto.INT64
ONNX_INT32 = TensorProto.INT32

def safe_name(a): return getattr(a, "name", None)

def fix_repeated_field(attr, field_name):
    # ensure repeated fields are lists rather than None
    if hasattr(attr, field_name):
        val = getattr(attr, field_name)
        if val is None:
            # set to empty sequence (works for lists)
            setattr(attr, field_name, [])
    return attr

def ensure_common_defaults(node):
    """Fill common missing attributes for ops that often break onnx-tf conversion."""
    # Cast -> 'to' (dtype)
    if node.op_type == "Cast":
        if not any(safe_name(a) == "to" for a in node.attribute):
            node.attribute.append(helper.make_attribute("to", TensorProto.FLOAT))
    # Concat -> 'axis'
    if node.op_type == "Concat":
        if not any(safe_name(a) == "axis" for a in node.attribute):
            node.attribute.append(helper.make_attribute("axis", 1))
    # ReduceMean/ReduceSum/Reduce* -> 'keepdims' default 1
    if node.op_type.startswith("Reduce"):
        if not any(safe_name(a) == "keepdims" for a in node.attribute):
            node.attribute.append(helper.make_attribute("keepdims", 1))
        # ensure axes attr exists and is a list of ints (if absent, create empty list)
        if not any(safe_name(a) == "axes" for a in node.attribute):
            node.attribute.append(helper.make_attribute("axes", []))
    # Shape -> nothing usually required, but ensure no None attrs
    # Slice/Reshape -> ensure needed attrs exist
    if node.op_type == "Slice":
        # older ONNX uses attributes starts/ends/axes; ensure they exist as ints lists
        if not any(safe_name(a) == "starts" for a in node.attribute):
            node.attribute.append(helper.make_attribute("starts", []))
        if not any(safe_name(a) == "ends" for a in node.attribute):
            node.attribute.append(helper.make_attribute("ends", []))
        if not any(safe_name(a) == "axes" for a in node.attribute):
            node.attribute.append(helper.make_attribute("axes", []))
    # Reshape -> no-op here
    return node

def clean_node_attributes(model):
    for node in model.graph.node:
        new_attrs = []
        for a in list(node.attribute):
            if a is None:
                continue
            # fix repeated fields that sometimes become None
            fix_repeated_field(a, "ints")
            fix_repeated_field(a, "floats")
            fix_repeated_field(a, "strings")
            # scalar fields
            if hasattr(a, "s") and a.s is None:
                a.s = b""
            if hasattr(a, "i") and a.i is None:
                a.i = 0
            new_attrs.append(a)
        del node.attribute[:]
        node.attribute.extend(new_attrs)
        ensure_common_defaults(node)

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
    # create a 1-element initializer and value_info
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
            # create matching value_info for initializer
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
            # try prepare on that small model
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

# --- main --------------------------------------------------------------------
def main():
    AUTO_FIX = os.environ.get("AUTO_FIX", "0") in ("1", "true", "True")
    infile = os.environ.get("INPUT_ONNX", "model_simplified.onnx")
    print("Input ONNX:", infile)

    model = onnx.load(infile)

    # shape inference
    try:
        print("Running shape inference...")
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print("shape inference failed (continuing):", e)

    # clean attributes & set defaults
    print("Cleaning node attributes and injecting defaults...")
    clean_node_attributes(model)

    # add missing value_info for outputs
    added = add_value_info_for_missing_outputs(model)
    if added:
        print("Added value_info for outputs:", added)

    cleaned = "model_cleaned_for_tf.onnx"
    onnx.save(model, cleaned)
    print("Saved cleaned model as", cleaned)

    # run checker
    try:
        print("Running ONNX checker...")
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print("ONNX checker failure:", e)
        open("diagnostics.txt", "w").write(str(e))
        sys.exit(1)

    # find missing inputs
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

    # attempt conversion
    try:
        print("Preparing onnx-tf representation (prepare)...")
        tf_rep = prepare(model, strict=False)
        print("Exporting SavedModel -> 'saved_model' ...")
        tf_rep.export_graph("saved_model")
    except Exception as e:
        print("ONNX->TensorFlow conversion failed (export stage):", e)
        # isolate failing node
        info = isolate_failing_node(model)
        if info:
            print("Isolation produced diagnostics.txt and failing_node_*.onnx. Attach them or paste here for help.")
        else:
            open("error.txt", "w").write(str(e))
            print("No isolation info; see error.txt")
        sys.exit(1)

    # convert saved_model to tflite
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
    main()        if not any(safe_attr_name(a) == "axis" for a in node.attribute):
            node.attribute.append(helper.make_attribute("axis", 0))
    # ReduceMean/ReduceSum/etc -> keepdims=1 if missing
    if node.op_type.startswith("Reduce"):
        if not any(safe_attr_name(a) == "keepdims" for a in node.attribute):
            node.attribute.append(helper.make_attribute("keepdims", 1))
        # if axes missing, let it be (some graphs use input axes)
    # other small defaults can be added here

def add_value_info_for_outputs(model):
    """Add value_info entries for outputs that are referenced later but do not have value_info.
       Use heuristic dtype/shape of [1] to satisfy converter.
    """
    existing = {vi.name for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
    added = []
    for node in model.graph.node:
        dtype = op_preferred_dtype(node.op_type, node)
        for out in node.output:
            if not out:
                continue
            if out not in existing:
                # create a safe value_info entry
                vi = helper.make_tensor_value_info(out, dtype, [1])
                model.graph.value_info.append(vi)
                existing.add(out)
                added.append((out, node.op_type))
    return added

def gather_present_names(model):
    names = set()
    for init in model.graph.initializer:
        names.add(init.name)
    for vi in model.graph.value_info:
        names.add(vi.name)
    for inp in model.graph.input:
        names.add(inp.name)
    for out in model.graph.output:
        names.add(out.name)
    for node in model.graph.node:
        for o in node.output:
            if o:
                names.add(o)
    return names

def find_missing_inputs(model):
    present = gather_present_names(model)
    missing = set()
    for node in model.graph.node:
        for inp in node.input:
            if not inp:
                continue
            if inp not in present:
                missing.add(inp)
    return sorted(missing)

def inject_dummy_initializer(model, name, dtype=ONNX_FLOAT):
    """Inject a 1-element initializer and corresponding input value_info (safe dummy)."""
    # ensure a safe name (no weird chars)
    safe_name = name
    # create tiny array
    arr = np.zeros((1,), dtype=np.float32 if dtype == ONNX_FLOAT else np.int64)
    tensor = numpy_helper.from_array(arr, name=safe_name)
    model.graph.initializer.append(tensor)
    # also add an input/vi describing it:
    try:
        vi = helper.make_tensor_value_info(safe_name, dtype, [1])
        model.graph.input.append(vi)
    except Exception:
        pass
    return safe_name

def clean_node_attributes(model):
    for node in model.graph.node:
        new_attrs = []
        for a in node.attribute:
            if a is None:
                continue
            # defensive fixes
            if hasattr(a, "ints") and a.ints is None:
                a.ints[:] = []
            if hasattr(a, "floats") and a.floats is None:
                a.floats[:] = []
            if hasattr(a, "strings") and a.strings is None:
                a.strings[:] = []
            if hasattr(a, "s") and a.s is None:
                a.s = b""
            if hasattr(a, "i") and a.i is None:
                a.i = 0
            # keep attr
            new_attrs.append(a)
        del node.attribute[:]
        node.attribute.extend(new_attrs)
        # fill safe defaults for known ops
        ensure_attr_defaults(node)

def minimal_model_for_node(model, node_index):
    node = model.graph.node[node_index]
    # copy initializers referenced by this node
    init_names = {init.name for init in model.graph.initializer}
    new_inits = []
    new_inputs = []
    for inp in node.input:
        if not inp:
            continue
        if inp in init_names:
            new_inits.append(next((it for it in model.graph.initializer if it.name == inp), None))
            if new_inits[-1] is not None:
                new_inputs.append(helper.make_tensor_value_info(inp, new_inits[-1].data_type, list(new_inits[-1].dims)))
        else:
            new_inputs.append(helper.make_tensor_value_info(inp, TensorProto.FLOAT, [1]))
    new_outputs = []
    for out in node.output:
        if out:
            new_outputs.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, [1]))
    new_graph = helper.make_graph([node], f"node_{node_index}_graph", new_inputs, new_outputs, initializer=[i for i in new_inits if i])
    nm = helper.make_model(new_graph, producer_name="minimal_debug")
    return nm

def isolate_failing_node(model):
    print("Running per-node isolation to find failing node (may be slow)...")
    issues = []
    for idx, node in enumerate(model.graph.node):
        try:
            small = minimal_model_for_node(model, idx)
            try:
                onnx.checker.check_model(small)
            except Exception:
                pass
            prepare(small, strict=False)
        except Exception as e:
            issue = {
                "index": idx,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": [safe_attr_name(a) for a in node.attribute],
                "error": repr(e)
            }
            issues.append(issue)
            fn = f"failing_node_{idx}.onnx"
            onnx.save(small, fn)
            open("diagnostics.txt", "w").write(json.dumps(issue, indent=2))
            print(f"Found failing node #{idx}: op={node.op_type} name='{node.name}' (saved {fn})")
            return issues
    print("No single-node repro failed.")
    return issues

# Main flow
def main():
    AUTO_FIX = os.environ.get("AUTO_FIX", "0") in ("1", "true", "True")
    infile = os.environ.get("INPUT_ONNX", "model_simplified.onnx")
    print("Loading ONNX:", infile)
    model = onnx.load(infile)

    # 1) try shape inference
    print("Running shape_inference...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print("shape_inference failed (continuing):", e)

    # 2) clean attributes
    print("Cleaning node attributes...")
    clean_node_attributes(model)

    # 3) ensure outputs have value_info (so onnx-tf won't choke on missing info)
    print("Adding missing value_info for outputs (heuristic)...")
    added_vinfo = add_value_info_for_outputs(model)
    if added_vinfo:
        print("Added value_info for outputs:", added_vinfo)

    # save cleaned file
    cleaned_name = "model_cleaned_for_onnx_tf.onnx"
    onnx.save(model, cleaned_name)
    print("Saved cleaned model:", cleaned_name)

    # 4) run ONNX checker
    print("Checking model validity...")
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print("ONNX checker error:", e)
        open("diagnostics.txt", "w").write(str(e))
        sys.exit(1)

    # 5) find missing inputs and optionally auto-fix
    missing = find_missing_inputs(model)
    if missing:
        print("Missing inputs referenced by nodes:", missing)
        if AUTO_FIX:
            print("AUTO_FIX enabled: injecting dummy initializers for missing inputs...")
            added = []
            for m in missing:
                added.append(inject_dummy_initializer(model, m))
            print("Injected dummies:", added)
            onnx.save(model, "model_cleaned_fixed.onnx")
        else:
            open("diagnostics.txt", "w").write(json.dumps({"missing_inputs": missing}, indent=2))
            print("Missing inputs found; re-run with AUTO_FIX=1 to auto-inject dummy initializers for debugging.")
            sys.exit(1)
    else:
        print("No missing inputs found.")

    # 6) Try conversion
    try:
        print("Preparing onnx-tf representation...")
        tf_rep = prepare(model, strict=False)
        print("Exporting SavedModel...")
        tf_rep.export_graph("saved_model")
    except Exception as e:
        print("ONNX->TensorFlow conversion failed:", e)
        # try to isolate failing node
        issues = isolate_failing_node(model)
        if issues:
            print("Isolation saved failing_node_*.onnx and diagnostics.txt")
        else:
            open("error.txt", "w").write(str(e))
            print("Could not isolate single node; see error.txt")
        sys.exit(1)

    # 7) convert SavedModel -> tflite
    try:
        print("Converting SavedModel -> TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        tflite = converter.convert()
        with open("model.tflite", "wb") as f:
            f.write(tflite)
        print("Wrote model.tflite")
    except Exception as e:
        print("SavedModel->TFLite conversion failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
