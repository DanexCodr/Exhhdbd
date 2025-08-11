#!/usr/bin/env python3
import os, sys, json
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
from typing import List, Dict, Any

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

def minimal_model_for_node(model: onnx.ModelProto, node_index: int) -> onnx.ModelProto:
    """Create a minimal ONNX model containing only the given node and
       any initializers referenced by its inputs (heuristic)."""
    node = model.graph.node[node_index]
    new_graph = helper.make_graph(
        nodes=[node],
        name=f"node_only_graph_{node_index}",
        inputs=[],
        outputs=[],
        initializer=[]
    )

    # Add inputs for all node inputs (if they aren't initializers)
    init_names = {init.name for init in model.graph.initializer}
    for inp in node.input:
        if not inp:
            continue
        if inp in init_names:
            # copy initializer
            init = next((it for it in model.graph.initializer if it.name == inp), None)
            if init is not None:
                new_graph.initializer.append(init)
                # also expose as input value_info so checker doesn't choke
                new_graph.input.append(helper.make_tensor_value_info(inp, init.data_type, list(init.dims)))
        else:
            # unknown input -> add a scalar float input to keep ONNX checker happy
            new_graph.input.append(helper.make_tensor_value_info(inp, TensorProto.FLOAT, [1]))

    # For outputs, add a generic output value info from the node output names
    for out in node.output:
        if out:
            new_graph.output.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, [1]))

    new_model = helper.make_model(new_graph, producer_name="minimal_for_debug")
    return new_model

def diagnose_and_find_bad_node(full_model_path: str, model: onnx.ModelProto, tf_rep):
    print("Running per-node isolation to find failing node (this may take a bit)...")
    issues = []
    for idx, node in enumerate(model.graph.node):
        try:
            small = minimal_model_for_node(model, idx)
            # run ONNX checker on the small model (quick)
            try:
                onnx.checker.check_model(small)
            except Exception:
                # checker may complain; still try prepare()
                pass
            # try prepare() on the small model
            prepare(small, strict=False)
        except Exception as e:
            issue = {
                "index": idx,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": [{ "name": getattr(a, "name", None), "type": a.type, "repr": str(a) } for a in node.attribute],
                "error": repr(e)
            }
            issues.append(issue)
            # save the failing single-node model for inspection
            fn = f"failing_node_{idx}.onnx"
            onnx.save(small, fn)
            print(f"Found failing node #{idx}: op={node.op_type} name='{node.name}'")
            print("Inputs:", node.input)
            print("Outputs:", node.output)
            print("Attrs summary:", [(getattr(a, "name", None)) for a in node.attribute])
            print("Saved minimal repro ONNX ->", fn)
            # write diagnostics
            with open("diagnostics.txt", "w", encoding="utf-8") as f:
                json.dump(issue, f, indent=2)
            # stop at first failing node (we want the primary offender)
            return issues
    # if nothing found
    print("No single-node repro failed. The issue may be multi-node or depends on graph context.")
    return issues

def main():
    in_file = os.environ.get("INPUT_ONNX", "model_simplified.onnx")
    print("Loading ONNX:", in_file)
    model = onnx.load(in_file)

    print("Cleaning model...")
    clean_onnx_model(model)
    onnx.save(model, "model_cleaned.onnx")

    print("Checking model validity...")
    try:
        onnx.checker.check_model(model)
        print("ONNX checker OK.")
    except Exception as ex:
        print("ONNX checker failure:", ex)
        sys.exit(1)

    # Try full conversion
    try:
        print("Preparing onnx-tf representation (prepare)...")
        tf_rep = prepare(model, strict=False)
        print("Exporting SavedModel (this is where the earlier failure happened)...")
        tf_rep.export_graph("saved_model")
    except Exception as e:
        print("ONNX->TensorFlow conversion failed (export stage):", e)
        # attempt to isolate failing node by per-node prepare on minimal models
        issues = diagnose_and_find_bad_node(in_file, model, None)
        if issues:
            print("Diagnostics written to diagnostics.txt and failing_node_*.onnx saved.")
            print("Paste the diagnostics or attach the failing_node_*.onnx and I'll suggest a patch.")
            sys.exit(1)
        else:
            print("Could not isolate a single-node failure. Next steps:")
            print("- try AUTO_FIX=1 (inject dummies) or")
            print("- share model_cleaned.onnx or a small failing_node_*.onnx if produced.")
            sys.exit(1)

    # If we reach here, export_graph succeeded
    print("Exported SavedModel. Converting to TFLite...")
    try:
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
