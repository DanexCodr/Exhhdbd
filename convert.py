import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def clean_onnx_model(model):
    for node in model.graph.node:
        new_attrs = []
        for attr in node.attribute:
            if attr is None:
                continue

            # Fix repeated fields to empty lists if None
            if hasattr(attr, 'ints') and attr.ints is None:
                attr.ints[:] = []
            if hasattr(attr, 'floats') and attr.floats is None:
                attr.floats[:] = []
            if hasattr(attr, 'strings') and attr.strings is None:
                attr.strings[:] = []

            # Fix scalar bytes string field
            if hasattr(attr, 's') and attr.s is None:
                attr.s = b''

            # Keep only attributes with meaningful data
            if ((not hasattr(attr, 'ints') or len(attr.ints) > 0) or
                (not hasattr(attr, 'floats') or len(attr.floats) > 0) or
                (not hasattr(attr, 'strings') or len(attr.strings) > 0) or
                (hasattr(attr, 's') and attr.s != b'')):
                new_attrs.append(attr)
            # else drop empty attribute

        del node.attribute[:]
        node.attribute.extend(new_attrs)

def main():
    print("Loading simplified ONNX model...")
    onnx_model = onnx.load("model_simplified.onnx")

    print("Cleaning ONNX model attributes...")
    clean_onnx_model(onnx_model)

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
