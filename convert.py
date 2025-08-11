#!/usr/bin/env python3
import os, sys, json
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

# ---------- helpers ----------
def clean_onnx_model(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        new_attrs = []
        for attr in node.attribute:
            if attr is None:
                continue
            if hasattr(attr, "ints"):
                if attr.ints is None:
                    attr.ints[:] = []
                else:
                    attr.ints[:] = [0 if i is None else i for i in attr.ints]
            if hasattr(attr, "floats"):
                if attr.floats is None:
                    attr.floats[:] = []
                else:
                    attr.floats[:] = [0.0 if f is None else f for f in attr.floats]
            if hasattr(attr, "strings"):
                if attr.strings is None:
                    attr.strings[:] = []
                else:
                    attr.strings[:] = [b"" if s is None else s for s in attr.strings]
            if hasattr(attr, "s") and attr.s is None:
                attr.s = b""
            if ((not hasattr(attr, "ints") or len(attr.ints) > 0)
                or (not hasattr(attr, "floats") or len(attr.floats) > 0)
                or (not hasattr(attr, "strings") or len(attr.strings) > 0)
                or (hasattr(attr, "s") and attr.s != b"")):
                new_attrs.append(attr)
        del node.attribute[:]
        node.attribute.extend(new_attrs)
        if node.op_type == "Cast":
            if not any(getattr(a, "name", None) == "to" for a in node.attribute):
                node.attribute.append(helper.make_attribute("to", TensorProto.FLOAT))
        if node.op_type == "Concat":
            if not any(getattr(a, "name", None) == "axis" for a in node.attribute):
                node.attribute.append(helper.make_attribute("axis", 0))
        for i, inp in enumerate(node.input):
            if inp is None:
                node.input[i] = ""
        for i, outp in enumerate(node.output):
            if outp is None:
                node.output[i] = ""

def gather_graph_names(model: onnx.ModelProto):
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
        for out in node.output:
            if out:
                names.add(out)
    return names

def find_missing_inputs(model: onnx.ModelProto):
    present = gather_graph_names(model)
    missing = set()
    for node in model.graph.node:
        for inp in node.input:
            if not inp:
                continue
            if inp not in present:
                missing.add(inp)
    return sorted(missing)

def inject_dummy_initializers(model: onnx.ModelProto, missing_names):
    added = []
    for i, name in enumerate(missing_names):
        # make a tiny 1-element float32 tensor
        arr = np.zeros((1,), dtype=np.float32)
        tensor = numpy_helper.from_array(arr, name=name if name.isidentifier() else f"__dummy_{i}")
        # if ONNX name contains slashes or odd chars, rename to safe dummy and replace inputs
        if tensor.name != name:
            dummy_name = tensor.name
            # replace occurrences
            for node in model.graph.node:
                for idx, inp in enumerate(node.input):
                    if inp == name:
                        node.input[idx] = dummy_name
            name_used = dummy_name
        else:
            name_used = name
        model.graph.initializer.append(tensor)
        # also add an input value_info for shape hints
        vi = helper.make_tensor_value_info(name_used, TensorProto.FLOAT, [1])
        model.graph.input.append(vi)
        added.append(name_used)
    return added

# ---------- main ----------
def main():
    in_file = os.environ.get("INPUT_ONNX", "model_simplified.onnx")
    auto_fix = os.environ.get("AUTO_FIX", "0") in ("1", "true", "True")

    print("Loading ONNX:", in_file)
    model = onnx.load(in_file)

    print("Running ONNX shape inference (may fill value_info)...")
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        model = inferred
    except Exception as e:
        print("shape_inference failed (continuing):", e)

    print("Cleaning model attributes...")
    clean_onnx_model(model)
    onnx.save(model, "model_cleaned.onnx")

    print("Running ONNX checker...")
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: model is valid.")
    except Exception as e:
        print("ONNX checker failed:", e)
        # still continue to attempt diagnostics
        # write diagnostics and exit
        open("diagnostics.txt", "w").write(str(e))
        sys.exit(1)

    missing = find_missing_inputs(model)
    print("Missing input names found:", missing)
    if missing and auto_fix:
        print("AUTO_FIX enabled — injecting dummy initializers for missing inputs (heuristic).")
        added = inject_dummy_initializers(model, missing)
        print("Added dummy initializers:", added)
        onnx.save(model, "model_cleaned_and_fixed.onnx")
    elif missing:
        print("Missing inputs exist. Re-run with AUTO_FIX=1 to inject dummy tensors for testing (risky).")
        # write diagnostics
        open("diagnostics.txt", "w").write(json.dumps({"missing_inputs": missing}, indent=2))
        sys.exit(1)

    # attempt onnx-tf prepare/export
    try:
        print("Converting ONNX -> TensorFlow (prepare)...")
        tf_rep = prepare(model, strict=False)
        print("Exporting SavedModel ...")
        tf_rep.export_graph("saved_model")
    except Exception as e:
        print("ONNX->TensorFlow conversion failed:", e)
        # save brief diagnostics
        open("error.txt", "w").write(str(e))
        # try to isolate failing node (quick)
        present = gather_graph_names(model)
        for idx, node in enumerate(model.graph.node):
            for inp in node.input:
                if inp and inp not in present:
                    info = {
                        "failing_node_index": idx,
                        "op_type": node.op_type,
                        "name": node.name,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "attrs": [getattr(a, "name", None) for a in node.attribute],
                        "error": str(e)
                    }
                    onnx.save(onnx.helper.make_model(onnx.helper.make_graph([node], "single", [], [])),
                              f"failing_node_{idx}.onnx")
                    open("diagnostics.txt", "w").write(json.dumps(info, indent=2))
                    print("Saved minimal failing node as failing_node_{}.onnx".format(idx))
                    print("Wrote diagnostics.txt")
                    sys.exit(1)
        print("Could not isolate missing-input node easily. See error.txt")
        sys.exit(1)

    # SavedModel -> TFLite
    try:
        print("Converting SavedModel -> TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        tflite_model = converter.convert()
        with open("model.tflite", "wb") as f:
            f.write(tflite_model)
        print("Wrote model.tflite")
    except Exception as e:
        print("SavedModel->TFLite conversion failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
