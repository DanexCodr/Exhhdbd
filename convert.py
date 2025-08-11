import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def clean_onnx_model(model):
    for node in model.graph.node:
        # Filter out None attributes
        filtered_attrs = [attr for attr in node.attribute if attr is not None]
        del node.attribute[:]
        node.attribute.extend(filtered_attrs)

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
