import orjson
from io import BytesIO
from PIL import Image
import base64


def default(obj):
    if isinstance(obj, Image.Image):
        with BytesIO() as out:
            obj.save(out, format="PNG")
            png_string = out.getvalue()
            return base64.b64encode(png_string).decode("utf-8")
    raise TypeError


class Jsoner:
    @staticmethod
    def deserialize(body):
        return orjson.loads(body)

    @staticmethod
    def serialize(body):
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
