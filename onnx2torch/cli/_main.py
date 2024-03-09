from typing import Union

import onnx


def convert(
    model: Union[str, onnx.ModelProto],
    output_model: str = None,
    check: bool = False,
    model_check_inputs: str = None,
):
    from onnx2torch.core.parser import OnnxPytorchParser

    onnx2torch = OnnxPytorchParser(model)
    onnx2torch.convert()

    if check:
        onnx2torch.check(model_check_inputs)

    if not output_model:
        return onnx2torch.pytorch_graph_module
    else:
        onnx2torch.save(output_model)


def main():
    import argparse

    from loguru import logger

    import onnx2torch

    parser = argparse.ArgumentParser(
        description="onnx2torch: Onnx to Pytorch Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_model", help="input onnx model")
    parser.add_argument(
        "output_model", nargs="?", default=None, help="output pytorch graph module"
    )

    parser.add_argument("--check", action="store_true", help="enable model check")
    parser.add_argument(
        "-v", "--version", action="version", version=onnx2torch.__version__
    )

    # Model Check Inputs
    parser.add_argument(
        "--model_check_inputs",
        nargs="+",
        type=str,
        help="Input shape of the model or numpy data path, INPUT_NAME:SHAPE or INPUT_NAME:DATAPATH, "
        "e.g. x:1,3,224,224 or x1:1,3,224,224 x2:data.npy. Useful when input shapes are dynamic.",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        logger.error(f"unrecognized options: {unknown}")
        return 1

    convert(
        args.input_model,
        args.output_model,
        args.check,
        args.model_check_inputs,
    )

    return 0
