import os, sys, json
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

# Helpers
def safe_attr_name(a): return getattr(a, "name", None)

ONNX_FLOAT = TensorProto.FLOAT
ONNX_INT64 = TensorProto.INT64

def op_preferred_dtype(op_type, node):
    """Heuristic: pick dtype for an op's outputs."""
    if op_type in ("Shape",):
        return ONNX_INT64
    if op_type in ("Range", "ArgMax", "ArgMin"):
        return ONNX_INT64
    if op_type in ("Cast",):
        # find 'to' attr if present
        for a in node.attribute:
            if safe_attr_name(a) == "to":
                return a.i if hasattr(a, "i") else ONNX_FLOAT
        return ONNX_FLOAT
    # default: float
    return ONNX_FLOAT

def ensure_attr_defaults(node):
    """Set common missing attributes to safe defaults."""
    # Cast -> to=float
    if node.op_type == "Cast":
        if not any(safe_attr_name(a) == "to" for a in node.attribute):
            node.attribute.append(helper.make_attribute("to", ONNX_FLOAT))
    # Concat -> axis=0
    if node.op_type == "Concat":
        if not any(safe_attr_name(a) == "axis" for a in node.attribute):
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
