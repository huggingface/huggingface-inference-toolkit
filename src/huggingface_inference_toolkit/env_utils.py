import os


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to True or False booleans.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises:
        ValueError: if 'val' is anything else.

    Note:
        Function `strtobool` copied and adapted from `distutils`, as it's deprecated from Python 3.10 onwards.

    References:
        - https://github.com/python/cpython/blob/48f9d3e3faec5faaa4f7c9849fecd27eae4da213/Lib/distutils/util.py#L308-L321
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(
        f"Invalid truth value, it should be a string but {val} was provided instead."
    )


def api_inference_compat() -> bool:
    """
    Whether to answer with the response shapes the Inference API / Hub widgets expect.

    Read on every call rather than cached at import: tests toggle it, and it costs nothing.
    """
    return strtobool(os.getenv("API_INFERENCE_COMPAT", "false"))


def ignore_custom_handler() -> bool:
    """
    Whether to ignore a `handler.py` shipped in the model repository and serve the model with the
    default pipeline instead.
    """
    return strtobool(os.getenv("IGNORE_CUSTOM_HANDLER", "false"))


def task_route_enabled() -> bool:
    """
    Whether to expose /pipeline/{task}, letting a caller name the pipeline to serve a request with.

    Defaults to api_inference_compat(): the Inference API needs the route, so defaulting to it means
    adopting this code takes no deployment change. It is a separate variable because the two are
    orthogonal — one picks a pipeline, the other reshapes responses — and because the route lets a
    caller cause a pipeline to be built per task, each with its own copy of the weights, so it is
    worth being able to say no to independently.

    Set ENABLE_TASK_ROUTE to have the route without the compat response shapes, or to keep it off
    while they are on.
    """
    value = os.getenv("ENABLE_TASK_ROUTE")
    return api_inference_compat() if value is None else strtobool(value)
