"""This module provides classes and functions for JSON encoding of numpy data types."""

import json
import base64
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encodes numpy data types into JSON-compatible formats.

    Extends the standard json.JSONEncoder to handle numeric, array, and bytes
    objects from numpy.
    """

    def default(self, obj):
        """Returns a JSON-serializable version of the given object.

        If the object is a numpy integer or float, it is converted to the
        appropriate Python type (int or float). If the object is a numpy array,
        it is converted to a list. If the object is a bytes object, it is
        Base64-encoded. Otherwise, the default json.JSONEncoder method is called.

        Args:
            obj (Any): The object to serialize.

        Returns:
            Any: A JSON-serializable representation of the given object.
        """
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        return json.JSONEncoder.default(self, obj)