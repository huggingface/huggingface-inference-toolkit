class PreTrainedPipeline:
    def __init__(self, path):
        self.path = path

    def __call__(self, data):
        return data
