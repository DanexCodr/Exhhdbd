import os
import onnx
from onnx import helper, numpy_helper
import tensorflow as tf
from onnx_tf.backend import prepare

# ====== Default constants ======
FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64
INPUT_ONNX = "model_simplified.onnx"
PATCHED_ONNX = "model_patched.onnx"
TF_MODEL_DIR = "tf_model"
OUTPUT_TFLITE = "model.tflite"

def get_opset_version(model):
    # pick the ai.onnx or default opset import
    for oi in model.opset_import:
        if oi.domain == "" or oi.domain == "ai.onnx":
            return oi.version
    return max((oi.version for oi in model.opset_import), default=0)

# ====== Fix Functions ======
def fix_cast_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Cast":
            has_to = any(getattr(attr, "name", None) == "to" for attr in node.attribute)
            if not has_to:
                print(f"Fixing {node.name or 'Cast'}: setting to=FLOAT")
                node.attribute.append(helper.make_attribute("to", FLOAT))

def fix_concat_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Concat":
            has_axis = any(getattr(attr, "name", None) == "axis" for attr in node.attribute)
            if not has_axis:
                print(f"Fixing {node.name or 'Concat'}: setting axis=-1")
                node.attribute.append(helper.make_attribute("axis", -1))

def fix_slice_nodes(model):
    """
    For opset >= 10: remove old-style starts/ends/axes/steps attributes (invalid)
    and ensure these are provided as input initializers (5 inputs total).
    For opset < 10: keep old attribute-style (add empty lists if missing).
    """
    opset_v = get_opset_version(model)
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Slice":
            continue

        attr_names = [getattr(a, "name", None) for a in node.attribute]

        if opset_v < 10:
            # attribute mode: ensure attributes exist (old behaviour)
            needed_attrs = ["starts", "ends", "axes", "steps"]
            for attr_name in needed_attrs:
                if attr_name not in attr_names:
                    print(f"Fixing {node.name or 'Slice'}: adding attribute {attr_name}=[] (opset < 10)")
                    node.attribute.append(helper.make_attribute(attr_name, []))
            continue

        # opset >= 10: attributes are invalid; convert attributes -> initializers/inputs
        # First collect any attribute-provided lists (if model incorrectly used attributes)
        attr_map = {getattr(a, "name", None): a for a in list(node.attribute)}
        # remove any starts/ends/axes/steps attributes to avoid "Unrecognized attribute" errors
        node.attribute[:] = [a for a in node.attribute if getattr(a, "name", None) not in ("starts", "ends", "axes", "steps")]

        # we expect up to 5 inputs: data, starts, ends, axes, steps
        # If some of those inputs are missing, create int64 initializers and append
        # Determine which inputs already exist (node.input[0] is data)
        existing_inputs = list(node.input)
        # determine names for missing inputs
        needed = ["starts", "ends", "axes", "steps"]
        # count how many "parameter" inputs currently present after data
        num_param_inputs = max(0, len(existing_inputs) - 1)
        # build list of parameter inputs present (if any)
        present_param_names = existing_inputs[1:]

        for i, key in enumerate(needed):
            if i < len(present_param_names):
                # param input exists; nothing to add for this param
                continue
            # create a unique name for initializer
            const_name = f"{node.name or 'Slice'}_{key}_const_{idx}"
            # choose default values:
            if key == "starts":
                vals = [0]
            elif key == "ends":
                vals = [2**30]  # a large sentinel (will be clipped by runtime)
            elif key == "axes":
                vals = [0]
            else:  # steps
                vals = [1]
            print(f"Fixing {node.name or 'Slice'}: injecting input {key}={vals} as initializer '{const_name}' (opset >= 10)")
            tensor = helper.make_tensor(name=const_name, data_type=INT64, dims=[len(vals)], vals=vals)
            model.graph.initializer.append(tensor)
            # also add a value_info so the name becomes known as output/input
            try:
                vi = helper.make_tensor_value_info(const_name, INT64, [len(vals)])
                model.graph.input.append(vi)
            except Exception:
                pass
            node.input.append(const_name)

def fix_unsqueeze_nodes(model):
    """
    Unsqueeze historically had 'axes' as an attribute. Newer opsets expect axes as input.
    - If opset < 13: ensure axes attribute exists (attribute mode).
    - If opset >= 13: move axes attribute into initializer/input if attribute present,
      or add initializer if missing.
    """
    opset_v = get_opset_version(model)
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Unsqueeze":
            continue

        attr_names = [getattr(a, "name", None) for a in node.attribute]

        if opset_v < 13:
            # attribute mode: ensure axes attribute exists
            if "axes" not in attr_names:
                print(f"Fixing {node.name or 'Unsqueeze'}: setting axes=[0] (attribute mode, opset < 13)")
                node.attribute.append(helper.make_attribute("axes", [0]))
            continue

        # opset >= 13: axes must be input tensor(s)
        # If there's an axes attribute, convert it to initializer + append to node.input
        axes_attr = None
        for a in list(node.attribute):
            if getattr(a, "name", None) == "axes":
                axes_attr = a
                break
        if axes_attr is not None:
            # remove attribute and create initializer for its ints
            node.attribute.remove(axes_attr)
            ints = list(axes_attr.ints) if hasattr(axes_attr, "ints") else ([int(axes_attr.i)] if hasattr(axes_attr, "i") else [0])
            const_name = f"{node.name or 'Unsqueeze'}_axes_const_{idx}"
            print(f"Fixing {node.name or 'Unsqueeze'}: converting attribute axes={ints} -> initializer '{const_name}' (opset >= 13)")
            tensor = helper.make_tensor(name=const_name, data_type=INT64, dims=[len(ints)], vals=ints)
            model.graph.initializer.append(tensor)
            try:
                vi = helper.make_tensor_value_info(const_name, INT64, [len(ints)])
                model.graph.input.append(vi)
            except Exception:
                pass
            node.input.append(const_name)
        else:
            # no attr; if only data input present, inject a default axes initializer
            if len(node.input) == 1:
                const_name = f"{node.name or 'Unsqueeze'}_axes_const_{idx}"
                print(f"Fixing {node.name or 'Unsqueeze'}: injecting default axes=[0] as initializer '{const_name}' (opset >= 13)")
                tensor = helper.make_tensor(name=const_name, data_type=INT64, dims=[1], vals=[0])
                model.graph.initializer.append(tensor)
                try:
                    vi = helper.make_tensor_value_info(const_name, INT64, [1])
                    model.graph.input.append(vi)
                except Exception:
                    pass
                node.input.append(const_name)

def fix_reduce_nodes(model):
    reduce_ops = ["ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin"]
    for node in model.graph.node:
        if node.op_type in reduce_ops:
            has_keepdims = any(getattr(attr, "name", None) == "keepdims" for attr in node.attribute)
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
    print("Fixing Slice ops (opset-aware)...")
    fix_slice_nodes(model)
    print("Fixing Unsqueeze ops (opset-aware)...")
    fix_unsqueeze_nodes(model)

# ====== Main pipeline ======
def main():
    print(f"Loading ONNX model from: {INPUT_ONNX}")
    onnx_model = onnx.load(INPUT_ONNX)

    autopatch_model(onnx_model)

    print(f"Saving patched ONNX to: {PATCHED_ONNX}")
    onnx.save(onnx_model, PATCHED_ONNX)

    print("Running ONNX checker...")
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX checker: OK")
    except Exception as e:
        print("ONNX checker failed after patch:", e)
        # still attempt conversion, but user should inspect patched model
        # exit to encourage fix if checker fails:
        raise

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
