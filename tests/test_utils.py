import inspect
import os
from test_config import TestsConfig


def file_and_test_annotation() -> str:
    """
    Gets the calling file and test name which is used as an annotation in plots to make it easier to trace the origin of
    the plot.

    :return: A formatted string containing the calling file and test name.
    """
    frame: inspect.FrameInfo = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    callers_file_name: str = os.path.basename(module.__file__)
    callers_function_name: str = inspect.stack()[1][3]
    annotation: str = \
        f'File: {callers_file_name}\n' \
        f'Test: {callers_function_name}' \
        if TestsConfig.show_test_location \
        else None

    return annotation
