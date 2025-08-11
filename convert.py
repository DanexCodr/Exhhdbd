import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
from onnx import helper, TensorProto

def fix_reduce_nodes(model):
    """Patch Reduce ops with None axes to default [0]."""
    for node in model.graph.node:
        if node.op_type.startswith("Reduce"):
            has_axes = any(attr.name == "axes" for attr in node.attribute)
            if not has_axes:
                print(f"Fixing {node.name or node.op_type}: setting axes=[0]")
                node.attribute.append(helper.make_attribute("axes", [0]))

def fix_cast_nodes(model):
    """Add missing 'to' attribute for Cast ops."""
    for node in model.graph.node:
        if node.op_type == "Cast":
            has_to = any(attr.name == "to" for attr in node.attribute)
            if not has_to:
                print(f"Fixing {node.name or 'Cast'}: setting to=FLOAT")
                node.attribute.append(helper.make_attribute("to", TensorProto.FLOAT))

def clean_model(model):
    """Remove empty attributes and sanitize fields."""
    for node in model.graph.node:
        new_attrs = []
        for attr in node.attribute:
            if hasattr(attr, 'ints') and attr.ints is None:
                attr.ints[:] = []
            if hasattr(attr, 'floats') and attr.floats is None:
                attr.floats[:] = []
            if hasattr(attr, 'strings') and attr.strings is None:
                attr.strings[:] = []
            if hasattr(attr, 's') and attr.s is None:
                attr.s = b''
            if ((not hasattr(attr, 'ints') or len(attr.ints) > 0) or
                (not hasattr(attr, 'floats') or len(attr.floats) > 0) or
                (not hasattr(attr, 'strings') or len(attr.strings) > 0) or
                (hasattr(attr, 's') and attr.s != b'')):
                new_attrs.append(attr)
        del node.attribute[:]
        node.attribute.extend(new_attrs)

def main():
    input_path = "model_simplified.onnx"
    output_path = "model.tflite"

    print(f"Loading ONNX model from: {input_path}")
    onnx_model = onnx.load(input_path)

    print("Cleaning model...")
    clean_model(onnx_model)

    print("Fixing Reduce ops...")
    fix_reduce_nodes(onnx_model)

    print("Fixing Cast ops...")
    fix_cast_nodes(onnx_model)

    print("Saving patched ONNX...")
    onnx.save(onnx_model, "model_fixed.onnx")

    print("Converting ONNX → TensorFlow...")
    tf_rep = prepare(onnx_model, strict=False)
    tf_rep.export_graph("saved_model")

    print("Converting TensorFlow → TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
    tflite_model = converter.convert()

    print(f"Saving TFLite model as {output_path}")
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print("✅ Conversion complete!")

if __name__ == "__main__":
    main()
