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
