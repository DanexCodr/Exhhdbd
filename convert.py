import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def main():
    print("Loading simplified ONNX model...")
    onnx_model = onnx.load("model_simplified.onnx")  # Use simplified model here

    print("Converting ONNX to TensorFlow...")
    tf_rep = prepare(onnx_model, strict=False)  # Disable strict mode to avoid None attribute errors

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
