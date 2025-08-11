#!/usr/bin/env python3
"""
Robust ONNX -> TensorFlow -> TFLite conversion helper.

This version is defensive: it deep-copies the model before patching,
avoids forcing opset edits (which caused assignment errors), downgrades
Unsqueeze nodes to attribute form (opset 11-style) to avoid missing
handler implementations, and does cleaning/shape-inference where possible.

Usage:
    python convert.py input_model.onnx output_model.tflite
"""
import os
import sys
import uuid
import tempfile
import shutil
import copy
import traceback

import onnx
from onnx import helper, shape_inference, numpy_helper
from onnx import TensorProto
import onnx.checker
import tensorflow as tf
from onnx_tf.backend import prepare

# Constants
FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64


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
                # conservative defaults
                starts = [0]
                ends = [9223372036854775807]
                axes = [0]
                steps = [1]

            # preserve only data input and append starts/ends/axes/steps constants
            if not node.input:
                # weird node with no inputs: skip
                print(f"Warning: Slice node '{node.name or node.op_type}' has no inputs, skipping fix")
                continue
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

            # remove attribute-style fields
            del node.attribute[:]
        else:
            for attr_name in ("starts", "ends", "axes", "steps"):
                if not any(getattr(a, "name", None) == attr_name for a in node.attribute):
                    node.attribute.append(helper.make_attribute(attr_name, []))


def fix_unsqueeze_nodes(model):
    # keep this in case old-opset unsqueeze attributes are required elsewhere
    opset_version = get_opset_version(model)
    for node in model.graph.node:
        if node.op_type == "Unsqueeze":
            if opset_version < 13:
                if not any(getattr(a, "name", None) == "axes" for a in node.attribute):
                    node.attribute.append(helper.make_attribute("axes", [0]))
            else:
                # if model is still opset>=13 and only has single input, add axes const as a fallback
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


def downgrade_unsqueeze_nodes(model):
    """
    Convert v13-style Unsqueeze (axes as input) into attribute-style (v11)
    so onnx-tf can find a handler for older versions. We do NOT modify model opset_imports
    to avoid assignment errors; we only change nodes.
    """
    for node in model.graph.node:
        if node.op_type == "Unsqueeze":
            # Remove axes input(s) if present (v13 style)
            if len(node.input) > 1:
                # keep data input only
                data = node.input[0]
                node.input[:] = [data]
            # remove axes attribute if present (avoid duplicates)
            node.attribute[:] = [a for a in node.attribute if getattr(a, "name", None) != "axes"]
            # add axes attribute as fallback
            node.attribute.append(helper.make_attribute("axes", [0]))


def fix_reduce_nodes(model):
    reduce_ops = ("ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin")
    for node in model.graph.node:
        if node.op_type in reduce_ops:
            if not any(getattr(a, "name", None) == "keepdims" for a in node.attribute):
                node.attribute.append(helper.make_attribute("keepdims", 1))


def clean_node_attributes_and_inputs(model):
    """
    Remove node attributes or attribute-fields that contain None values,
    and remove empty/None inputs. Defensive against UPB repeated containers.
    """
    for node in model.graph.node:
        new_attrs = []
        for attr in list(node.attribute):
            if attr is None:
                print(f"Removed None attribute from node '{node.name or node.op_type}'")
                continue
            try:
                fields = attr.ListFields()
            except Exception:
                print(f"Skipping attribute (ListFields failed) on node '{node.name or node.op_type}'")
                continue

            bad = False
            for _, field_value in fields:
                try:
                    if field_value is None:
                        bad = True
                        break
                    if hasattr(field_value, "__iter__") and not isinstance(field_value, (bytes, str)):
                        for v in field_value:
                            if v is None:
                                bad = True
                                break
                        if bad:
                            break
                except Exception:
                    bad = True
                    break

            if bad:
                print(f"Removed attribute '{getattr(attr, 'name', None)}' from node '{node.name or node.op_type}' because it contained None")
                continue

            new_attrs.append(attr)

        # replace attributes
        del node.attribute[:]
        node.attribute.extend(new_attrs)

        # Clean inputs: remove empty strings or None
        new_inputs = []
        for inp in node.input:
            if inp is None:
                print(f"Removed None input from node '{node.name or node.op_type}'")
                continue
            if isinstance(inp, str) and inp.strip() == "":
                print(f"Removed empty-string input from node '{node.name or node.op_type}'")
                continue
            new_inputs.append(inp)

        if len(new_inputs) != len(node.input):
            print(f"Cleaned inputs for node '{node.name or node.op_type}': {len(node.input)} -> {len(new_inputs)}")
        del node.input[:]
        node.input.extend(new_inputs)


def replace_shape_nodes(model):
    """
    Replace Shape nodes with Constant nodes when the shape can be
    determined from value_info/inputs. Uses safe clearing of model.graph.node.
    """
    shape_map = {}
    # collect shapes from value_info and inputs (conservative)
    for tensor in list(model.graph.value_info) + list(model.graph.input):
        try:
            shape = tensor.type.tensor_type.shape
            shape_dims = [dim.dim_value if (hasattr(dim, "dim_value") and dim.dim_value and dim.dim_value > 0) else 1 for dim in shape.dim]
            shape_map[tensor.name] = shape_dims
        except Exception:
            continue

    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "Shape":
            if node.input and node.input[0] in shape_map:
                shape_val = shape_map[node.input[0]]
                const_name = node.output[0]
                tensor = helper.make_tensor(
                    name=const_name + "_val",
                    data_type=INT64,
                    dims=[len(shape_val)],
                    vals=shape_val
                )
                const_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[const_name],
                    name=(node.name + "_replaced") if node.name else unique_name("Shape_replaced"),
                    value=tensor
                )
                new_nodes.append(const_node)
                print(f"Replaced Shape node {node.name or '(unnamed)'} with Constant -> {const_name}")
                continue
        new_nodes.append(node)

    # Safely replace nodes list
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)


def fix_dynamic_dims(model):
    for input_tensor in model.graph.input:
        try:
            shape = input_tensor.type.tensor_type.shape
            for dim in shape.dim:
                # treat non-positive or unknown dims as dynamic; set to 1
                if not hasattr(dim, "dim_value") or dim.dim_value <= 0:
                    dim.dim_value = 1
                    print(f"Fixed dynamic dimension in input {input_tensor.name} to 1")
        except Exception:
            continue


def check_none_attributes(model):
    """Prints any nodes/inputs/attributes that still contain None/empty strings."""
    found = False
    for i, node in enumerate(model.graph.node):
        for attr in node.attribute:
            if attr is None:
                print(f"[Node {i} - {node.name or node.op_type}] has a None attribute object")
                found = True
            else:
                try:
                    fields = attr.ListFields()
                    for name, val in fields:
                        if val is None:
                            print(f"[Node {i} - {node.name or node.op_type}] attribute '{getattr(attr, 'name', None)}' field '{name}' is None")
                            found = True
                        elif hasattr(val, "__iter__") and not isinstance(val, (bytes, str)):
                            for v in val:
                                if v is None:
                                    print(f"[Node {i} - {node.name or node.op_type}] attribute '{getattr(attr, 'name', None)}' contains None element")
                                    found = True
                                    break
                except Exception:
                    print(f"[Node {i} - {node.name or node.op_type}] attribute.ListFields() failed (could be UPB); please inspect")
                    found = True

        for idx, inp in enumerate(node.input):
            if inp is None or (isinstance(inp, str) and inp.strip() == ""):
                print(f"[Node {i} - {node.name or node.op_type}] input[{idx}] is empty or None")
                found = True

    if not found:
        print("No None attributes/inputs found in model nodes (basic check).")


def ensure_unsqueeze_axes_inputs(model):
    """
    Ensure Unsqueeze nodes that require an 'axes' input actually have one.
    If an Unsqueeze node has only the data input, append an INT64 constant
    initializer with value taken from an 'axes' attribute (if present) or
    fallback to [0].
    """
    for node in model.graph.node:
        if node.op_type != "Unsqueeze":
            continue

        # If axes already present as a second input, skip
        if len(node.input) > 1 and node.input[1] not in (None, ""):
            continue

        # Try to read axes from an attribute if present
        axes_vals = None
        for a in node.attribute:
            if getattr(a, "name", None) == "axes":
                try:
                    # attribute may be repeated ints
                    axes_vals = list(a.ints)
                except Exception:
                    axes_vals = None
                break

        if axes_vals is None or len(axes_vals) == 0:
            axes_vals = [0]  # conservative fallback

        # If node has no data input, skip with a warning
        if not node.input:
            print(f"Warning: Unsqueeze node '{node.name or node.op_type}' has no inputs; skipping injection")
            continue

        # Create unique initializer name & append if missing
        const_name = unique_name((node.name or "Unsqueeze") + "_axes_const")
        axes_tensor = helper.make_tensor(
            name=const_name,
            data_type=INT64,
            dims=[len(axes_vals)],
            vals=axes_vals
        )
        append_initializer_if_missing(model, axes_tensor)

        # Replace inputs to be [data_input, axes_const]
        data_input = node.input[0]
        node.input[:] = [data_input, const_name]

        # Remove any axes attribute to avoid ambiguity
        node.attribute[:] = [attr for attr in node.attribute if getattr(attr, "name", None) != "axes"]

        print(f"Injected axes tensor {const_name}={axes_vals} into Unsqueeze node '{node.name or node.op_type}'")


def fallback_unsqueeze_to_v11(model, default_axes=[0]):
    """
    Minimal fallback: convert Unsqueeze nodes that use input-based 'axes' (opset >=13)
    into attribute-based axes (opset 11 style) when possible.

    - If the second input (axes) is an initializer/constant, read it and turn it into an attribute.
    - If it's not a constant, add a conservative attribute default to allow conversion to proceed,
      while warning that this may be incorrect for dynamically computed axes.
    This function does NOT modify model.opset_import to avoid assignment/opsset errors.
    """
    changed = 0
    init_map = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type != "Unsqueeze":
            continue

        # if axes passed as second input
        if len(node.input) >= 2:
            axes_input_name = node.input[1]
            axes_vals = None

            if axes_input_name in init_map:
                try:
                    arr = numpy_helper.to_array(init_map[axes_input_name])
                    axes_vals = [int(x) for x in arr.flatten().tolist()]
                except Exception as e:
                    print(f"Warning: failed to read axes initializer '{axes_input_name}' for node '{node.name or node.op_type}': {e}")
                    axes_vals = None

            if axes_vals is None:
                # non-constant axes -> fallback to default, but warn
                print(f"Warning: Unsqueeze node '{node.name or node.op_type}' has non-constant axes input '{axes_input_name}'. "
                      f"Injecting default axes={default_axes} to let conversion proceed (may be incorrect).")
                axes_vals = list(default_axes)

            # downgrade node to attribute-style: remove axes input and set attribute
            node.input[:] = [node.input[0]]  # keep data input only
            # remove any existing axes attribute to avoid duplicates
            node.attribute[:] = [a for a in node.attribute if getattr(a, "name", None) != "axes"]
            node.attribute.append(helper.make_attribute("axes", axes_vals))
            changed += 1
        else:
            # no axes input; ensure an axes attribute exists
            if not any(getattr(a, "name", None) == "axes" for a in node.attribute):
                node.attribute.append(helper.make_attribute("axes", list(default_axes)))
                changed += 1

    if changed:
        print(f"fallback_unsqueeze_to_v11: patched {changed} Unsqueeze nodes.")
    return changed


def autopatch_model(model):
    fix_reduce_nodes(model)
    fix_cast_nodes(model)
    fix_concat_nodes(model)
    fix_slice_nodes(model)

    # Apply the fallback that converts input-based Unsqueeze axes into attributes (v11-style).
    try:
        fallback_unsqueeze_to_v11(model)
    except Exception as e:
        print(f"fallback_unsqueeze_to_v11 failed (continuing): {e}")

    # Ensure any Unsqueeze nodes have axes input if some code expects it
    try:
        ensure_unsqueeze_axes_inputs(model)
    except Exception as e:
        print(f"ensure_unsqueeze_axes_inputs failed (continuing): {e}")

    # run cleaning after other smaller fixes
    clean_node_attributes_and_inputs(model)


def onnx_to_tflite(input_onnx, output_tflite):
    if not os.path.exists(input_onnx):
        raise FileNotFoundError(f"Input ONNX model not found: {input_onnx}")

    print(f"Loading ONNX model: {input_onnx}")
    model_orig = onnx.load(input_onnx)

    # work on a deepcopy to avoid mutating shared/upstream objects that may be read-only
    model = copy.deepcopy(model_orig)

    print("Running shape inference...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Shape inference failed (continuing): {e}")

    print("Fixing dynamic dimensions...")
    fix_dynamic_dims(model)

    print("Replacing Shape nodes (where possible)...")
    try:
        # attempt to infer shapes again to help replace Shape nodes
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    try:
        replace_shape_nodes(model)
    except Exception as e:
        print(f"Failed to replace Shape nodes (continuing): {e}")

    print("Patching ONNX model (autopatches + cleaning)...")
    try:
        autopatch_model(model)
    except Exception as e:
        print("Autopatch step raised an exception:", e)
        traceback.print_exc()

    print("Cleaning node attributes and inputs once more...")
    clean_node_attributes_and_inputs(model)

    print("Checking for None attributes/inputs (diagnostics)...")
    check_none_attributes(model)

    print("Running final shape inference (will continue if it fails)...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Final shape inference failed (non-fatal): {e}")

    # Save patched ONNX temporarily
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        onnx.save(model, tmp_path)
        print(f"Saved cleaned model as: {tmp_path}")

    print("Validating ONNX model with checker...")
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print(f"ONNX checker reported issues (continuing): {e}")

    print("Converting patched ONNX to TensorFlow (onnx-tf prepare)...")
    try:
        tf_rep = prepare(model, strict=False)
    except Exception as e:
        print("ONNX->TF prepare failed:")
        traceback.print_exc()
        # re-raise so CI can see stack and diagnostics
        raise

    # Use temp directory for TF SavedModel
    tmp_tf_dir = tempfile.mkdtemp(prefix="tf_model_")
    try:
        print(f"Exporting SavedModel -> '{tmp_tf_dir}' ...")
        tf_rep.export_graph(tmp_tf_dir)
        print("Converting TensorFlow SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_tf_dir)
        tflite_model = converter.convert()

        with open(output_tflite, "wb") as f:
            f.write(tflite_model)

        print(f"✅ TFLite model saved to: {output_tflite}")
    finally:
        # Clean up temporary files
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_tf_dir)
        except Exception:
            pass


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
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()    try:
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
                # conservative defaults
                starts = [0]
                ends = [9223372036854775807]
                axes = [0]
                steps = [1]

            # preserve only data input and append starts/ends/axes/steps constants
            if not node.input:
                # weird node with no inputs: skip
                print(f"Warning: Slice node '{node.name or node.op_type}' has no inputs, skipping fix")
                continue
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

            # remove attribute-style fields
            del node.attribute[:]
        else:
            for attr_name in ("starts", "ends", "axes", "steps"):
                if not any(getattr(a, "name", None) == attr_name for a in node.attribute):
                    node.attribute.append(helper.make_attribute(attr_name, []))


def fix_unsqueeze_nodes(model):
    # keep this in case old-opset unsqueeze attributes are required elsewhere
    opset_version = get_opset_version(model)
    for node in model.graph.node:
        if node.op_type == "Unsqueeze":
            if opset_version < 13:
                if not any(getattr(a, "name", None) == "axes" for a in node.attribute):
                    node.attribute.append(helper.make_attribute("axes", [0]))
            else:
                # if model is still opset>=13 and only has single input, add axes const as a fallback
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


def downgrade_unsqueeze_nodes(model):
    """
    Convert v13-style Unsqueeze (axes as input) into attribute-style (v11)
    so onnx-tf can find a handler for older versions. We do NOT modify model opset_imports
    to avoid assignment errors; we only change nodes.
    """
    for node in model.graph.node:
        if node.op_type == "Unsqueeze":
            # Remove axes input(s) if present (v13 style)
            if len(node.input) > 1:
                # keep data input only
                data = node.input[0]
                node.input[:] = [data]
            # remove axes attribute if present (avoid duplicates)
            node.attribute[:] = [a for a in node.attribute if getattr(a, "name", None) != "axes"]
            # add axes attribute as fallback
            node.attribute.append(helper.make_attribute("axes", [0]))


def fix_reduce_nodes(model):
    reduce_ops = ("ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin")
    for node in model.graph.node:
        if node.op_type in reduce_ops:
            if not any(getattr(a, "name", None) == "keepdims" for a in node.attribute):
                node.attribute.append(helper.make_attribute("keepdims", 1))


def clean_node_attributes_and_inputs(model):
    """
    Remove node attributes or attribute-fields that contain None values,
    and remove empty/None inputs. Defensive against UPB repeated containers.
    """
    for node in model.graph.node:
        new_attrs = []
        for attr in list(node.attribute):
            if attr is None:
                print(f"Removed None attribute from node '{node.name or node.op_type}'")
                continue
            try:
                fields = attr.ListFields()
            except Exception:
                print(f"Skipping attribute (ListFields failed) on node '{node.name or node.op_type}'")
                continue

            bad = False
            for _, field_value in fields:
                try:
                    if field_value is None:
                        bad = True
                        break
                    if hasattr(field_value, "__iter__") and not isinstance(field_value, (bytes, str)):
                        for v in field_value:
                            if v is None:
                                bad = True
                                break
                        if bad:
                            break
                except Exception:
                    bad = True
                    break

            if bad:
                print(f"Removed attribute '{getattr(attr, 'name', None)}' from node '{node.name or node.op_type}' because it contained None")
                continue

            new_attrs.append(attr)

        # replace attributes
        del node.attribute[:]
        node.attribute.extend(new_attrs)

        # Clean inputs: remove empty strings or None
        new_inputs = []
        for inp in node.input:
            if inp is None:
                print(f"Removed None input from node '{node.name or node.op_type}'")
                continue
            if isinstance(inp, str) and inp.strip() == "":
                print(f"Removed empty-string input from node '{node.name or node.op_type}'")
                continue
            new_inputs.append(inp)

        if len(new_inputs) != len(node.input):
            print(f"Cleaned inputs for node '{node.name or node.op_type}': {len(node.input)} -> {len(new_inputs)}")
        del node.input[:]
        node.input.extend(new_inputs)


def replace_shape_nodes(model):
    """
    Replace Shape nodes with Constant nodes when the shape can be
    determined from value_info/inputs. Uses safe clearing of model.graph.node.
    """
    shape_map = {}
    # collect shapes from value_info and inputs (conservative)
    for tensor in list(model.graph.value_info) + list(model.graph.input):
        try:
            shape = tensor.type.tensor_type.shape
            shape_dims = [dim.dim_value if (hasattr(dim, "dim_value") and dim.dim_value and dim.dim_value > 0) else 1 for dim in shape.dim]
            shape_map[tensor.name] = shape_dims
        except Exception:
            continue

    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "Shape":
            if node.input and node.input[0] in shape_map:
                shape_val = shape_map[node.input[0]]
                const_name = node.output[0]
                tensor = helper.make_tensor(
                    name=const_name + "_val",
                    data_type=INT64,
                    dims=[len(shape_val)],
                    vals=shape_val
                )
                const_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[const_name],
                    name=(node.name + "_replaced") if node.name else unique_name("Shape_replaced"),
                    value=tensor
                )
                new_nodes.append(const_node)
                print(f"Replaced Shape node {node.name or '(unnamed)'} with Constant -> {const_name}")
                continue
        new_nodes.append(node)

    # Safely replace nodes list
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)


def fix_dynamic_dims(model):
    for input_tensor in model.graph.input:
        try:
            shape = input_tensor.type.tensor_type.shape
            for dim in shape.dim:
                # treat non-positive or unknown dims as dynamic; set to 1
                if not hasattr(dim, "dim_value") or dim.dim_value <= 0:
                    dim.dim_value = 1
                    print(f"Fixed dynamic dimension in input {input_tensor.name} to 1")
        except Exception:
            continue


def check_none_attributes(model):
    """Prints any nodes/inputs/attributes that still contain None/empty strings."""
    found = False
    for i, node in enumerate(model.graph.node):
        for attr in node.attribute:
            if attr is None:
                print(f"[Node {i} - {node.name or node.op_type}] has a None attribute object")
                found = True
            else:
                try:
                    fields = attr.ListFields()
                    for name, val in fields:
                        if val is None:
                            print(f"[Node {i} - {node.name or node.op_type}] attribute '{getattr(attr, 'name', None)}' field '{name}' is None")
                            found = True
                        elif hasattr(val, "__iter__") and not isinstance(val, (bytes, str)):
                            for v in val:
                                if v is None:
                                    print(f"[Node {i} - {node.name or node.op_type}] attribute '{getattr(attr, 'name', None)}' contains None element")
                                    found = True
                                    break
                except Exception:
                    print(f"[Node {i} - {node.name or node.op_type}] attribute.ListFields() failed (could be UPB); please inspect")
                    found = True

        for idx, inp in enumerate(node.input):
            if inp is None or (isinstance(inp, str) and inp.strip() == ""):
                print(f"[Node {i} - {node.name or node.op_type}] input[{idx}] is empty or None")
                found = True

    if not found:
        print("No None attributes/inputs found in model nodes (basic check).")

def ensure_unsqueeze_axes_inputs(model):
    """
    Ensure Unsqueeze nodes that require an 'axes' input actually have one.
    If an Unsqueeze node has only the data input, append an INT64 constant
    initializer with value taken from an 'axes' attribute (if present) or
    fallback to [0].
    """
    for node in model.graph.node:
        if node.op_type != "Unsqueeze":
            continue

        # If axes already present as a second input, skip
        if len(node.input) > 1 and node.input[1] not in (None, ""):
            continue

        # Try to read axes from an attribute if present
        axes_vals = None
        for a in node.attribute:
            if getattr(a, "name", None) == "axes":
                try:
                    # attribute may be repeated ints
                    axes_vals = list(a.ints)
                except Exception:
                    axes_vals = None
                break

        if axes_vals is None or len(axes_vals) == 0:
            axes_vals = [0]  # conservative fallback

        # If node has no data input, skip with a warning
        if not node.input:
            print(f"Warning: Unsqueeze node '{node.name or node.op_type}' has no inputs; skipping injection")
            continue

        # Create unique initializer name & append if missing
        const_name = unique_name((node.name or "Unsqueeze") + "_axes_const")
        axes_tensor = helper.make_tensor(
            name=const_name,
            data_type=INT64,
            dims=[len(axes_vals)],
            vals=axes_vals
        )
        append_initializer_if_missing(model, axes_tensor)

        # Replace inputs to be [data_input, axes_const]
        data_input = node.input[0]
        node.input[:] = [data_input, const_name]

        # Remove any axes attribute to avoid ambiguity
        node.attribute[:] = [attr for attr in node.attribute if getattr(attr, "name", None) != "axes"]

        print(f"Injected axes tensor {const_name}={axes_vals} into Unsqueeze node '{node.name or node.op_type}'")
        
def autopatch_model(model):
    fix_reduce_nodes(model)
    fix_cast_nodes(model)
    fix_concat_nodes(model)
    fix_slice_nodes(model)
    # Downgrade v13-style unsqueeze if you still do that
    # downgrade_unsqueeze_nodes(model)
    # Ensure any Unsqueeze nodes missing axes input get one
    ensure_unsqueeze_axes_inputs(model)
    # run cleaning after other smaller fixes
    clean_node_attributes_and_inputs(model)

def onnx_to_tflite(input_onnx, output_tflite):
    if not os.path.exists(input_onnx):
        raise FileNotFoundError(f"Input ONNX model not found: {input_onnx}")

    print(f"Loading ONNX model: {input_onnx}")
    model_orig = onnx.load(input_onnx)

    # work on a deepcopy to avoid mutating shared/upstream objects that may be read-only
    model = copy.deepcopy(model_orig)

    print("Running shape inference...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Shape inference failed (continuing): {e}")

    print("Fixing dynamic dimensions...")
    fix_dynamic_dims(model)

    print("Replacing Shape nodes (where possible)...")
    try:
        # attempt to infer shapes again to help replace Shape nodes
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    try:
        replace_shape_nodes(model)
    except Exception as e:
        print(f"Failed to replace Shape nodes (continuing): {e}")

    print("Patching ONNX model (autopatches + cleaning)...")
    try:
        autopatch_model(model)
    except Exception as e:
        print("Autopatch step raised an exception:", e)
        traceback.print_exc()

    print("Cleaning node attributes and inputs once more...")
    clean_node_attributes_and_inputs(model)

    print("Checking for None attributes/inputs (diagnostics)...")
    check_none_attributes(model)

    print("Running final shape inference (will continue if it fails)...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Final shape inference failed (non-fatal): {e}")

    # Save patched ONNX temporarily
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        onnx.save(model, tmp_path)
        print(f"Saved cleaned model as: {tmp_path}")

    print("Validating ONNX model with checker...")
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print(f"ONNX checker reported issues (continuing): {e}")

    print("Converting patched ONNX to TensorFlow (onnx-tf prepare)...")
    try:
        tf_rep = prepare(model, strict=False)
    except Exception as e:
        print("ONNX->TF prepare failed:")
        traceback.print_exc()
        # re-raise so CI can see stack and diagnostics
        raise

    # Use temp directory for TF SavedModel
    tmp_tf_dir = tempfile.mkdtemp(prefix="tf_model_")
    try:
        print(f"Exporting SavedModel -> '{tmp_tf_dir}' ...")
        tf_rep.export_graph(tmp_tf_dir)
        print("Converting TensorFlow SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_tf_dir)
        tflite_model = converter.convert()

        with open(output_tflite, "wb") as f:
            f.write(tflite_model)

        print(f"✅ TFLite model saved to: {output_tflite}")
    finally:
        # Clean up temporary files
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_tf_dir)
        except Exception:
            pass


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
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
