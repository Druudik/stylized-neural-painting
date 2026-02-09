from collections.abc import Callable, Iterable
from typing import Any


def flatten(arr: Iterable[Any]) -> list[Any]:
    """Recursively flatten nested iterables into a single list.

    Args:
        arr: An iterable that may contain nested lists, tuples, or sets.

    Returns:
        A flat list containing all elements from the nested structure.
    """
    result: list[Any] = []
    for item in arr:
        if isinstance(item, (list, tuple, set)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def optional_wrap(obj: Any, wrapper: Callable | None, *args: Any) -> Any:
    """Conditionally wrap an object with a wrapper function.

    Args:
        obj: The object to potentially wrap.
        wrapper: A callable that takes obj and *args, or None to skip wrapping.
        *args: Additional arguments to pass to the wrapper.

    Returns:
        The wrapped object if wrapper is provided, otherwise the original object.
    """
    if wrapper is None:
        return obj
    else:
        return wrapper(obj, *args)
