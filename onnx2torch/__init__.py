import os
import warnings

from .cli import convert
from .version import __version__


if os.path.dirname(os.path.realpath(__file__)) == os.path.join(
    os.path.realpath(os.getcwd()), "o2t"
):
    message = (
        "You are importing o2t within its own root folder ({}). "
        "This is not expected to work and may give errors. Please exit the "
        "o2t project source and relaunch your python interpreter."
    )
    warnings.warn(message.format(os.getcwd()))
