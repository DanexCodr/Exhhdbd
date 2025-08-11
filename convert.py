# convert.py
import sys
import onnx
from onnx import helper, TensorProto
from onnx_tf.backend import prepare
import tensorflow as tf
import json
from typing import List, Dict, Any

def clean_onnx_model(model: onnx.ModelProto) -> None:
    """Fix many common None/missing-attribute issues in-place."""
    for node in model.graph.node:
        new_attrs = []
        for attr in node.attribute:
            if attr is None:
                continue

            # Fix repeated fields to empty lists if None or containing None
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

            # Fix scalar bytes string field
            if hasattr(attr, "s") and attr.s is None:
                attr.s = b""

            # Keep only attributes that actually carry something, else drop
            if ((not hasattr(attr, "ints") or len(attr.ints) > 0)
                or (not hasattr(attr, "floats") or len(attr.floats) > 0)
                or (not hasattr(attr, "strings") or len(attr.strings) > 0)
                or (hasattr(attr, "s") and attr.s != b"")):
                new_attrs.append(attr)

        # Replace attributes with cleaned list
        del node.attribute[:]
        node.attribute.extend(new_attrs)

        # Add required attributes for some ops if missing
        if node.op_type == "Cast":
            if not any(getattr(a, "name", None) == "to" for a in node.attribute):
                node.attribute.append(helper.make_attribute("to", TensorProto.FLOAT))

        if node.op_type == "Concat":
            if not any(getattr(a, "name", None) == "axis" for a in node.attribute):
                node.attribute.append(helper.make_attribute("axis", 0))

        # Replace None inputs/outputs with empty string to avoid None propagation
        for i, inp in enumerate(node.input):
            if inp is None:
                node.input[i] = ""
        for i, outp in enumerate(node.output):
            if outp is None:
                node.output[i] = ""

def gather_graph_names(model: onnx.ModelProto) -> set:
    names = set()
    for init in model.graph.initializer:
        names.add(init.name)
    for vi in model.graph.value_info:
        names.add(vi.name)
    for inp in model.graph.input:
        names.add(inp.name)
    for out in model.graph.output:
        names.add(out.name)
    # also add nodes' outputs to set (names created by nodes)
    for node in model.graph.node:
        for out in node.output:
            if out:
                names.add(out)
    return names

def diagnose_model(model: onnx.ModelProto) -> List[Dict[str, Any]]:
    """Produce a list of suspicious nodes (missing attrs/inputs/None fields)."""
    issues = []
    present_names = gather_graph_names(model)

    for idx, node in enumerate(model.graph.node):
        node_issues = []
        # Check attributes for None-containing fields
        for attr in node.attribute:
            # ListFields returns (field_descriptor, value)
            for fd, value in attr.ListFields():
                if value is None:
                    node_issues.append({
                        "type": "attr_field_none",
                        "attr_name": getattr(attr, "name", "<unknown>"),
                        "field": fd.name
                    })
                elif isinstance(value, (list, tuple)):
                    if any(v is None for v in value):
                        node_issues.append({
                            "type": "attr_list_contains_none",
                            "attr_name": getattr(attr, "name", "<unknown>"),
                            "field": fd.name
                        })

        # Check inputs referencing missing names
        for inp in node.input:
            if inp and inp not in present_names:
                node_issues.append({"type": "missing_input", "input_name": inp})

        # Check if op-specific required attrs obviously missing
        if node.op_type == "Cast" and not any(getattr(a, "name", None) == "to" for a in node.attribute):
            node_issues.append({"type": "missing_required_attr", "attr": "to", "op": "Cast"})
        if node.op_type == "Concat" and not any(getattr(a, "name", None) == "axis" for a in node.attribute):
            node_issues.append({"type": "missing_required_attr", "attr": "axis", "op": "Concat"})

        if node_issues:
            issues.append({
                "index": idx,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "issues": node_issues
            })
    return issues

def save_diagnostics(issues: List[Dict[str, Any]], filename: str = "diagnostics.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        if not issues:
            f.write("No issues found.\n")
            return
        f.write(json.dumps(issues, indent=2, ensure_ascii=False))
    print(f"Diagnostics written to {filename}")

def main():
    in_file = "model_simplified.onnx"
    cleaned_out = "model_cleaned.onnx"
    print("Loading simplified ONNX model...")
    model = onnx.load(in_file)

    print("Cleaning ONNX model attributes...")
    clean_onnx_model(model)

    print(f"Saving cleaned ONNX -> {cleaned_out}")
    onnx.save(model, cleaned_out)

    print("Running ONNX checker...")
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: model is valid.")
    except Exception as e:
        print("ONNX checker failed:", e)
        print("Running diagnostics to find suspicious nodes...")
        issues = diagnose_model(model)
        save_diagnostics(issues)
        print("Model failed validation after cleaning. See diagnostics.txt for details.")
        sys.exit(1)

    # Convert with onnx-tf
    try:
        print("Converting ONNX to TensorFlow (onnx-tf)...")
        tf_rep = prepare(model, strict=False)
        saved_model_dir = "saved_model"
        print(f"Exporting TensorFlow SavedModel to '{saved_model_dir}'...")
        tf_rep.export_graph(saved_model_dir)
    except Exception as e:
        print("ONNX->TensorFlow conversion failed:", e)
        print("Running diagnostics to find suspicious nodes...")
        issues = diagnose_model(model)
        save_diagnostics(issues)
        print("Conversion aborted. See diagnostics.txt for suspected nodes.")
        sys.exit(1)

    # Convert to TFLite
    try:
        print("Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        output_file = "model.tflite"
        with open(output_file, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model -> {output_file}")
    except Exception as e:
        print("SavedModel -> TFLite conversion failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
