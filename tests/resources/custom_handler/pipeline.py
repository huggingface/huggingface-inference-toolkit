from custom_utils import test_method
class PreTrainedPipeline:
    def __init__(self, path):
        self.path = path

    def __call__(self, data):
        res = test_method(data)
        return res
