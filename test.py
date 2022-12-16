import time
from time import perf_counter
from huggingface_inference_toolkit.utils import get_pipeline


# importing the library
from memory_profiler import profile

# instantiating the decorator
@profile
# code for which memory has to
# be monitored
def my_func():
    start_time = perf_counter()
    x = get_pipeline("text-classification", "yoshitomo-matsubara/bert-large-uncased-sst2")
    latency = perf_counter() - start_time
    print(f"loading the model took {latency} seconds")
    return x


if __name__ == "__main__":
    my_func()
    time.sleep(10)
