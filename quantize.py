import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "model.onnx"
output_model = "model_quant.onnx"

quantize_dynamic(input_model, output_model, weight_type=QuantType.QInt8)

print("Quantization complete.")
