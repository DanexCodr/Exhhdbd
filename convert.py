#!/usr/bin/env python3
import os
import sys
import json
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
from typing import List, Dict, Any

# ---------- Cleaning utilities ----------
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

# ---------- Diagnostics ----------
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
    for node in model.graph.node:
        for out in node.output:
            if out:
                names.add(out)
    return names

def diagnose_model(model: onnx.ModelProto) -> List[Dict[str, Any]]:
    issues = []
    present_names = gather_graph_names(model)
    for idx, node in enumerate(model.graph.node):
        node_issues = []
        # Attributes checks
        for attr in node.attribute:
            # Use reflection to find lists with None items
            if hasattr(attr, "ints"):
                if attr.ints is None:
                    node_issues.append({"type": "attr_none", "field": "ints", "attr_name": getattr(attr, "name", None)})
                elif any(i is None for i in attr.ints):
                    node_issues.append({"type": "attr_list_contains_none", "field": "ints", "attr_name": getattr(attr, "name", None)})
            if hasattr(attr, "floats"):
                if attr.floats is None:
                    node_issues.append({"type": "attr_none", "field": "floats", "attr_name": getattr(attr, "name", None)})
                elif any(f is None for f in attr.floats):
                    node_issues.append({"type": "attr_list_contains_none", "field": "floats", "attr_name": getattr(attr, "name", None)})
            if hasattr(attr, "strings"):
                if attr.strings is None:
                    node_issues.append({"type": "attr_none", "field": "strings", "attr_name": getattr(attr, "name", None)})
                elif any(s is None for s in attr.strings):
                    node_issues.append({"type": "attr_list_contains_none", "field": "strings", "attr_name": getattr(attr, "name", None)})
            if hasattr(attr, "s") and attr.s is None:
                node_issues.append({"type": "attr_none", "field": "s", "attr_name": getattr(attr, "name", None)})

        # Missing inputs referencing names not present in graph
        for inp in node.input:
            if inp and inp not in present_names:
                node_issues.append({"type": "missing_input", "input_name": inp})

        # Required op attrs
        if node.op_type == "Cast" and not any(getattr(a, "name", None) == "to" for a in node.attribute):
            node_issues.append({"type": "missing_required_attr", "op": "Cast", "attr": "to"})
        if node.op_type == "Concat" and not any(getattr(a, "name", None) == "axis" for a in node.attribute):
            node_issues.append({"type": "missing_required_attr", "op": "Concat", "attr": "axis"})

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

# ---------- Auto-fix (optional & heuristic) ----------
def auto_fix_issues(model: onnx.ModelProto, issues: List[Dict[str, Any]]) -> int:
    """
    Heuristically patch missing_input issues by adding small dummy initializers
    and replacing missing input names with the new initializer names.
    Returns number of dummy initializers added.
    WARNING: this may change model semantics; use only for testing/unblocking.
    """
    present = gather_graph_names(model)
    added = 0
    for it in issues:
        for iss in it["issues"]:
            if iss["type"] == "missing_input":
                missing_name = iss["input_name"]
                # if a dummy already created with this missing_name, skip
                # create a unique dummy name
                dummy_name = f"__onnx_dummy_{added}"
                # create a tiny tensor (scalar float)
                arr = numpy_helper.from_array((0.0).astype("float32") if hasattr(__import__("numpy"), "float32") else (0.0), name=dummy_name)
                # fallback safe creation if numpy_helper fails:
                try:
                    import numpy as _np
                    arr = numpy_helper.from_array(_np.array([0.0], dtype=_np.float32), name=dummy_name)
                    model.graph.initializer.append(arr)
                except Exception:
                    # as last resort add a TensorProto manually (1 element float)
                    t = helper.make_tensor(name=dummy_name, data_type=TensorProto.FLOAT, dims=[1], vals=[0.0])
                    model.graph.initializer.append(t)
                # also add a graph input so name appears in gather_graph_names
                vi = helper.make_tensor_value_info(dummy_name, TensorProto.FLOAT, [1])
                model.graph.input.append(vi)

                # replace occurrences in node inputs
                node = model.graph.node[it["index"]]
                for i, inp in enumerate(node.input):
                    if inp == missing_name:
                        node.input[i] = dummy_name

                added += 1
    return added

# ---------- Main flow ----------
def main():
    infile = os.environ.get("INPUT_ONNX", "model_simplified.onnx")
    cleaned_out = "model_cleaned.onnx"
    auto_fix = os.environ.get("AUTO_FIX", "0") in ("1", "true", "True")

    print("Loading ONNX:", infile)
    model = onnx.load(infile)

    print("Running initial clean...")
    clean_onnx_model(model)

    print("Saving cleaned ONNX ->", cleaned_out)
    onnx.save(model, cleaned_out)

    print("Running ONNX checker...")
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: model is valid.")
    except Exception as e:
        print("ONNX checker failed:", e)
        issues = diagnose_model(model)
        save_diagnostics(issues)
        # print top issues to console
        print("--- Top diagnostics (first 20) ---")
        for idx, it in enumerate(issues[:20]):
            print(f"[{idx}] node #{it['index']} name='{it['name']}' op={it['op_type']} issues={it['issues']}")
        sys.exit(1)

    # Try conversion
    try:
        print("Converting ONNX -> TensorFlow (onnx-tf)...")
        tf_rep = prepare(model, strict=False)
        saved_model_dir = "saved_model"
        print("Exporting SavedModel:", saved_model_dir)
        tf_rep.export_graph(saved_model_dir)
    except Exception as e:
        print("ONNX->TensorFlow conversion failed:", e)
        # run diagnostics
        issues = diagnose_model(model)
        save_diagnostics(issues)
        print("--- Top diagnostics (first 50) ---")
        for idx, it in enumerate(issues[:50]):
            print(f"[{idx}] node #{it['index']} name='{it['name']}' op={it['op_type']} issues={it['issues']}")
        if auto_fix and issues:
            print("AUTO_FIX enabled — attempting heuristic fixes (may alter semantics)")
            added = auto_fix_issues(model, issues)
            print(f"Added {added} dummy initializer(s). Saving to {cleaned_out} and retrying conversion.")
            onnx.save(model, cleaned_out)
            try:
                tf_rep = prepare(model, strict=False)
                tf_rep.export_graph("saved_model")
            except Exception as e2:
                print("Retry after auto-fix still failed:", e2)
                sys.exit(1)
        else:
            print("To try automatic fixes set environment AUTO_FIX=1 and re-run (risky).")
            sys.exit(1)

    # Convert SavedModel -> TFLite
    try:
        print("Converting SavedModel -> TFLite...")
        tflite_out = "model.tflite"
        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        tflite_model = converter.convert()
        with open(tflite_out, "wb") as f:
            f.write(tflite_model)
        print("Wrote", tflite_out)
    except Exception as e:
        print("SavedModel -> TFLite conversion failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
