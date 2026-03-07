"""Utilities for marking dev modules with usage-time warnings."""
import functools
import warnings


def _warn_dev(name, module):
    warnings.warn(
        f"`{name}` from `{module}` is under development and may be "
        f"incorrect. Use with caution.",
        UserWarning,
        stacklevel=3,
    )


def _wrap_function(func, module):
    """Wrap a function to emit a dev warning on first call."""
    warned = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal warned
        if not warned:
            _warn_dev(func.__name__, module)
            warned = True
        return func(*args, **kwargs)

    return wrapper


def _wrap_class(cls, module):
    """Wrap a class so its __init__ emits a dev warning on first use."""
    original_init = cls.__init__
    warned = False

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        nonlocal warned
        if not warned:
            _warn_dev(cls.__name__, module)
            warned = True
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


def mark_dev_exports(namespace, module_name):
    """Wrap all public classes and functions in a namespace dict with dev
    warnings. Call this at the end of a dev ``__init__.py``::

        mark_dev_exports(globals(), __name__)
    """
    for name, obj in list(namespace.items()):
        if name.startswith("_"):
            continue
        if isinstance(obj, type):
            namespace[name] = _wrap_class(obj, module_name)
        elif callable(obj):
            namespace[name] = _wrap_function(obj, module_name)
