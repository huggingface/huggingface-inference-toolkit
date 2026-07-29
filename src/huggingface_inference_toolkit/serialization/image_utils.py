from io import BytesIO

from PIL import Image

# A media type is not a PIL format name. PIL registers "JPEG", never "JPG", and `image/x-image`
# names no format at all, so deriving the format from the subtype produced a `KeyError` for two
# of the types we advertise as supported. Keys are bare, lowercased media types, which is what
# `ContentType.resolve_accept` hands back.
MEDIA_TYPE_TO_PIL_FORMAT = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/tiff": "TIFF",
    "image/bmp": "BMP",
    "image/gif": "GIF",
    "image/webp": "WEBP",
    # Not a real format: it means "an image", so answer with the lossless one.
    "image/x-image": "PNG",
}

# JPEG has no alpha channel and no palette, so PIL refuses those modes outright.
_JPEG_SAFE_MODES = {"RGB", "L", "CMYK"}


class Imager:
    @staticmethod
    def deserialize(body):
        image = Image.open(BytesIO(body)).convert("RGB")
        return {"inputs": image}

    @staticmethod
    def serialize(image, accept=None):
        if not isinstance(image, Image.Image):
            raise ValueError(f"Can only serialize PIL.Image.Image, got {type(image)}")

        # `accept` reaches us already resolved to a bare media type, but tolerate a raw header
        # so a direct caller is not silently given the wrong format.
        media_type = (accept or "image/png").split(";")[0].strip().lower()
        if media_type not in MEDIA_TYPE_TO_PIL_FORMAT:
            raise ValueError(
                f'Cannot serialize an image as "{accept}". Supported types are: '
                f"{', '.join(MEDIA_TYPE_TO_PIL_FORMAT)}"
            )
        image_format = MEDIA_TYPE_TO_PIL_FORMAT[media_type]

        if image_format == "JPEG" and image.mode not in _JPEG_SAFE_MODES:
            # Masks come back as "P" and generated images can carry alpha; both are valid
            # answers to a JPEG request, so flatten rather than fail.
            image = image.convert("RGB")

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image_format)
        return img_byte_arr.getvalue()
