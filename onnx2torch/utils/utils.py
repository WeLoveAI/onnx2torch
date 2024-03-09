from typing import Dict, List, Optional

import numpy as np

import onnx
import torch


def onnx_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    import onnx.mapping as mapping

    return np.dtype(mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype])


def gen_onnxruntime_input_data(
    model: onnx.ModelProto, model_check_inputs: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    input_info = {}
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            elif dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(None)
        dtype = onnx_dtype_to_numpy(input_tensor.type.tensor_type.elem_type)

        input_info[name] = {"shape": shape, "dtype": dtype}

    if model_check_inputs:
        for model_check_input in model_check_inputs:
            key, value = model_check_input.rsplit(":", 1)
            if value.endswith(".npy"):
                data = np.load(value)
                input_info[key] = {"data": data}
            else:
                values_list = [int(val) for val in value.split(",")]
                if key in input_info:
                    input_info[key]["shape"] = values_list
                else:
                    raise Exception(
                        f"model_check_input name:{key} not found in model, available keys: {' '.join(input_info.keys())}"
                    )

    input_data_dict = {}
    for name, info in input_info.items():
        if "data" in info:
            input_data_dict[name] = info["data"]
        else:
            shapes = [
                shape if (shape != -1 and not isinstance(shape, str)) else 1
                for shape in info["shape"]
            ]
            shapes = shapes if shapes else [1]
            dtype = info["dtype"]

            if dtype in [np.int32, np.int64]:
                random_data = np.random.randint(10, size=shapes).astype(dtype)
            else:
                random_data = np.random.rand(*shapes).astype(dtype)
            input_data_dict[name] = random_data

    return input_data_dict


def onnxruntime_inference(
    model: onnx.ModelProto, input_data: dict
) -> Dict[str, np.array]:
    import onnxruntime as rt

    sess = rt.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    onnx_output = sess.run(None, input_data)

    output_names = [output.name for output in sess.get_outputs()]
    onnx_output = dict(zip(output_names, onnx_output))

    return onnx_output


numpy_to_torch_dtype_dict = {
    np.dtype(bool): torch.bool,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.complex64): torch.complex64,
    np.dtype(np.complex128): torch.complex128,
}


def numpy_dtype_to_torch(scalar_type):
    return numpy_to_torch_dtype_dict[scalar_type]
