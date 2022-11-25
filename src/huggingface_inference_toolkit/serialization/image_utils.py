from io import BytesIO

from PIL import Image


class Imager:
    @staticmethod
    def deserialize(body):
        image = Image.open(BytesIO(body)).convert("RGB")
        return {"inputs": image}

    @staticmethod
    def serialize(image, accept=None):
        if isinstance(image, Image.Image):
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format=accept.split("/")[-1].upper())
            img_byte_arr = img_byte_arr.getvalue()
            return img_byte_arr
        else:
            raise ValueError(f"Can only serialize PIL.Image.Image, got {type(image)}")
