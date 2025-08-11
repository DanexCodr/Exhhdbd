import onnx
from onnx import helper
import tensorflow as tf
from onnx_tf.backend import prepare
import os

# ====== Default constants ======
FLOAT = onnx.TensorProto.FLOAT
INPUT_ONNX = "model_simplified.onnx"
PATCHED_ONNX = "model_patched.onnx"
TF_MODEL_DIR = "tf_model"
OUTPUT_TFLITE = "model.tflite"

# ====== Fix Functions ======
def fix_cast_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Cast":
            has_to = any(attr.name == "to" for attr in node.attribute)
            if not has_to:
                print(f"Fixing {node.name or 'Cast'}: setting to=FLOAT")
                node.attribute.append(helper.make_attribute("to", FLOAT))

def fix_concat_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Concat":
            has_axis = any(attr.name == "axis" for attr in node.attribute)
            if not has_axis:
                print(f"Fixing {node.name or 'Concat'}: setting axis=-1")
                node.attribute.append(helper.make_attribute("axis", -1))

def fix_slice_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Slice":
            needed_attrs = ["starts", "ends", "axes", "steps"]
            for attr_name in needed_attrs:
                if not any(attr.name == attr_name for attr in node.attribute):
                    print(f"Fixing {node.name or 'Slice'}: adding {attr_name}=[]")
                    node.attribute.append(helper.make_attribute(attr_name, []))

def fix_unsqueeze_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Unsqueeze":
            has_axes = any(attr.name == "axes" for attr in node.attribute)
            if not has_axes:
                print(f"Fixing {node.name or 'Unsqueeze'}: setting axes=[0]")
                node.attribute.append(helper.make_attribute("axes", [0]))

def fix_reduce_nodes(model):
    reduce_ops = ["ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin"]
    for node in model.graph.node:
        if node.op_type in reduce_ops:
            has_keepdims = any(attr.name == "keepdims" for attr in node.attribute)
            if not has_keepdims:
                print(f"Fixing {node.name or node.op_type}: setting keepdims=1")
                node.attribute.append(helper.make_attribute("keepdims", 1))

# ====== Autopatcher ======
def autopatch_model(model):
    print("Cleaning model...")
    fix_reduce_nodes(model)
    print("Fixing Cast ops...")
    fix_cast_nodes(model)
    print("Fixing Concat ops...")
    fix_concat_nodes(model)
    print("Fixing Slice ops...")
    fix_slice_nodes(model)
    print("Fixing Unsqueeze ops...")
    fix_unsqueeze_nodes(model)

# ====== Main pipeline ======
def main():
    print(f"Loading ONNX model from: {INPUT_ONNX}")
    onnx_model = onnx.load(INPUT_ONNX)

    autopatch_model(onnx_model)

    print(f"Saving patched ONNX to: {PATCHED_ONNX}")
    onnx.save(onnx_model, PATCHED_ONNX)

    print("Converting ONNX → TensorFlow...")
    tf_rep = prepare(onnx_model, strict=False)
    if os.path.exists(TF_MODEL_DIR):
        import shutil
        shutil.rmtree(TF_MODEL_DIR)
    tf_rep.export_graph(TF_MODEL_DIR)

    print("Converting TensorFlow → TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_DIR)
    tflite_model = converter.convert()

    with open(OUTPUT_TFLITE, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Conversion complete! Saved TFLite model as {OUTPUT_TFLITE}")

if __name__ == "__main__":
    main()
