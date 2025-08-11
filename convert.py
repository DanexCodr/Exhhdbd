import onnx
from onnx import helper, TensorProto
from onnx_tf.backend import prepare
import tensorflow as tf

def clean_onnx_model(model):
    for node in model.graph.node:
        new_attrs = []
        for attr in node.attribute:
            if attr is None:
                continue

            if hasattr(attr, 'ints') and (attr.ints is None or any(i is None for i in attr.ints)):
                attr.ints[:] = [i if i is not None else 0 for i in attr.ints] if attr.ints else []
            if hasattr(attr, 'floats') and (attr.floats is None or any(f is None for f in attr.floats)):
                attr.floats[:] = [f if f is not None else 0.0 for f in attr.floats] if attr.floats else []
            if hasattr(attr, 'strings') and (attr.strings is None or any(s is None for s in attr.strings)):
                attr.strings[:] = [s if s is not None else b'' for s in attr.strings] if attr.strings else []

            if hasattr(attr, 's') and attr.s is None:
                attr.s = b''

            if ((not hasattr(attr, 'ints') or len(attr.ints) > 0) or
                (not hasattr(attr, 'floats') or len(attr.floats) > 0) or
                (not hasattr(attr, 'strings') or len(attr.strings) > 0) or
                (hasattr(attr, 's') and attr.s != b'')):
                new_attrs.append(attr)

        del node.attribute[:]
        node.attribute.extend(new_attrs)

        # Add missing 'to' attribute for Cast nodes
        if node.op_type == "Cast":
            if not any(attr.name == "to" for attr in node.attribute):
                to_attr = helper.make_attribute("to", TensorProto.FLOAT)
                node.attribute.append(to_attr)

        # Add missing 'axis' attribute for Concat nodes
        if node.op_type == "Concat":
            if not any(attr.name == "axis" for attr in node.attribute):
                axis_attr = helper.make_attribute("axis", 0)
                node.attribute.append(axis_attr)

        # Fix None inputs by replacing with empty string
        for i, inp in enumerate(node.input):
            if inp is None:
                node.input[i] = ""

        # Fix None outputs by replacing with empty string
        for i, outp in enumerate(node.output):
            if outp is None:
                node.output[i] = ""

def main():
    print("Loading simplified ONNX model...")
    onnx_model = onnx.load("model_simplified.onnx")

    print("Cleaning ONNX model attributes...")
    clean_onnx_model(onnx_model)

    print("Checking model validity...")
    onnx.checker.check_model(onnx_model)

    print("Converting ONNX to TensorFlow...")
    tf_rep = prepare(onnx_model, strict=False)

    saved_model_dir = "saved_model"
    print(f"Exporting TensorFlow SavedModel to '{saved_model_dir}'...")
    tf_rep.export_graph(saved_model_dir)

    print("Converting SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    output_file = "model.tflite"
    print(f"Saving TFLite model to '{output_file}'...")
    with open(output_file, "wb") as f:
        f.write(tflite_model)

    print("Conversion complete!")

if __name__ == "__main__":
    main()
