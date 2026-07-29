import base64
from io import BytesIO

import numpy as np
import orjson
from PIL import Image


def default(obj):
    if isinstance(obj, Image.Image):
        with BytesIO() as out:
            obj.save(out, format="PNG")
            png_string = out.getvalue()
            return base64.b64encode(png_string).decode("utf-8")
    if isinstance(obj, np.ndarray):
        # `OPT_SERIALIZE_NUMPY` defers an array here for one of two reasons.
        if not obj.flags["C_CONTIGUOUS"]:
            # Not C-contiguous, e.g. a truncated embedding (`a[:, :256]`). Copy into a
            # contiguous buffer so it goes back through orjson's native path, which keeps the
            # float32 formatting; `.tolist()` would widen to float64 and undo the size win.
            return np.ascontiguousarray(obj)
        # Contiguous, so the dtype is one orjson cannot handle. Hand back the elements and let
        # them be serialized individually, or fail here if they cannot be.
        return obj.tolist()
    raise TypeError


class Jsoner:
    @staticmethod
    def deserialize(body):
        return orjson.loads(body)

    @staticmethod
    def serialize(body, accept=None):
        return orjson.dumps(body, option=orjson.OPT_SERIALIZE_NUMPY, default=default)


# class _JSONEncoder(json.JSONEncoder):
#     """
#     custom `JSONEncoder` to make sure float and int64 ar converted
#     """

#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, datetime.datetime):
#             return obj.__str__()
#         elif isinstance(obj, Image.Image):
#             with BytesIO() as out:
#                 obj.save(out, format="PNG")
#                 png_string = out.getvalue()
#                 return base64.b64encode(png_string).decode("utf-8")
#         else:
#             return super(_JSONEncoder, self).default(obj)
