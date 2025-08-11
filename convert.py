#!/usr/bin/env python3
import os
import sys
import onnx
from onnx import helper
import tensorflow as tf
from onnx_tf.backend import prepare
import uuid
import tempfile
import shutil

FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64

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
                node.attribute.append(helper.make_attribute("to", FLOAT))

def fix_concat_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Concat":
            if not any(getattr(a, "name", None) == "axis" for a in node.attribute):
                node.attribute.append(helper.make_attribute("axis", -1))

def fix_slice_nodes(model):
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

            data_input = node.input[0]
            node.input[:] = [data_input]

            for name_base, vals in (("starts", starts), ("ends", ends), ("axes", axes), ("steps", steps)):
                const_name = unique_name(f"{node.name or 'Slice'}_{name_base}_const")
                tensor = helper.make_tensor(
                    name=const_name,
                    data_type=INT64,
                    dims=[len(vals)] if len(vals) > 0 else [0],
                    vals=vals
                )
                append_initializer_if_missing(model, tensor)
                node.input.append(const_name)

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
                    axes_tensor = helper.make_tensor(
                        name=axes_name,
                        data_type=INT64,
                        dims=[1],
                        vals=[0]
                    )
                    append_initializer_if_missing(model, axes_tensor)
                    node.input.append(axes_name)

def fix_reduce_nodes(model):
    reduce_ops = ("ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin")
    for node in model.graph.node:
        if node.op_type in reduce_ops:
            if not any(getattr(a, "name", None) == "keepdims" for a in node.attribute):
                node.attribute.append(helper.make_attribute("keepdims", 1))

def clean_node_attributes_and_inputs(model):
    for node in model.graph.node:
        # Clean attributes: remove those with None or fields containing None
        new_attrs = []
        for attr in node.attribute:
            if attr is None:
                continue
            fields = attr.ListFields()
            if any(field_value is None or
                   (hasattr(field_value, '__iter__') and
                    any(v is None for v in field_value if v is not None))
                   for _, field_value in fields):
                # Skip attribute with None inside
                continue
            new_attrs.append(attr)
        del node.attribute[:]
        node.attribute.extend(new_attrs)

        # Clean inputs: remove empty or None inputs
        new_inputs = [i for i in node.input if i and i.strip() != ""]
        if len(new_inputs) != len(node.input):
            print(f"Cleaned empty inputs from node '{node.name or node.op_type}'")
        del node.input[:]
        node.input.extend(new_inputs)
        
def autopatch_model(model):
    fix_reduce_nodes(model)
    fix_cast_nodes(model)
    fix_concat_nodes(model)
    fix_slice_nodes(model)
    fix_unsqueeze_nodes(model)
    clean_node_attributes_and_inputs(model)  # add this at the end to fix any None attribute fields

def check_none_attributes(model):
    for i, node in enumerate(model.graph.node):
        for attr in node.attribute:
            if attr is None:
                print(f"[Node {i} - {node.name or node.op_type}] has None attribute!")
            else:
                fields = attr.ListFields()
                for name, val in fields:
                    if val is None:
                        print(f"[Node {i} - {node.name or node.op_type}] attribute '{attr.name}' field '{name}' is None!")
        for idx, inp in enumerate(node.input):
            if inp is None or (isinstance(inp, str) and inp.strip() == ""):
                print(f"[Node {i} - {node.name or node.op_type}] input[{idx}] is empty or None!")

def onnx_to_tflite(input_onnx, output_tflite):
    if not os.path.exists(input_onnx):
        raise FileNotFoundError(f"Input ONNX model not found: {input_onnx}")

    print(f"Loading ONNX model: {input_onnx}")
    model = onnx.load(input_onnx)

    print("Inspecting ONNX nodes for None attributes or inputs...")
    check_none_attributes(model)

    print("Patching ONNX model...")
    autopatch_model(model)

    # Save patched ONNX temporarily
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        onnx.save(model, tmp_path)

    print("Converting patched ONNX to TensorFlow...")
    tf_rep = prepare(model, strict=False)

    # Use temp directory for TF SavedModel
    tmp_tf_dir = tempfile.mkdtemp(prefix="tf_model_")
    try:
        tf_rep.export_graph(tmp_tf_dir)
        print("Converting TensorFlow SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_tf_dir)
        tflite_model = converter.convert()

        with open(output_tflite, "wb") as f:
            f.write(tflite_model)

        print(f"✅ TFLite model saved to: {output_tflite}")
    finally:
        # Clean temp files
        os.remove(tmp_path)
        shutil.rmtree(tmp_tf_dir)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input_model.onnx output_model.tflite")
        return

    input_onnx = sys.argv[1]
    output_tflite = sys.argv[2]

    try:
        onnx_to_tflite(input_onnx, output_tflite)
    except Exception as e:
        print("Error during conversion:", e)

if __name__ == "__main__":
    main()
