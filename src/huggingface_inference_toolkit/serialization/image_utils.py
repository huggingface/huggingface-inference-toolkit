from io import BytesIO

from PIL import Image


class Imager:
    @staticmethod
    def deserialize(body):
        image = Image.open(BytesIO(body)).convert("RGB")
        return {"inputs": image}

    @staticmethod
    def serialize(body):
        raise NotImplementedError("Image serialization not implemented")
