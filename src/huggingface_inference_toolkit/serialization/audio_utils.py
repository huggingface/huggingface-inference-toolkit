class Audioer:
    @staticmethod
    def deserialize(body):
        return {"inputs": bytes(body)}

    @staticmethod
    def serialize(body):
        raise NotImplementedError("Audio serialization not implemented")
