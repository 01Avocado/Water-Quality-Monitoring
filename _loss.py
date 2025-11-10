"""
Compatibility shim for legacy scikit-learn pickles.

Some serialized estimators reference the private module `_loss` at the
top-level (a packaging quirk of certain scikit-learn versions). When the
module cannot be resolved, loading fails with `ModuleNotFoundError: No
module named '_loss'`. To retain forward compatibility on deployment
targets, we expose that name and delegate to the canonical implementation
within scikit-learn.
"""

import importlib


def _expose(module_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False

    for name in dir(module):
        if not name.startswith("_"):
            globals()[name] = getattr(module, name)
    return True


if not (
    _expose("sklearn._loss._loss")
    or _expose("sklearn._loss")
    or _expose("sklearn.metrics._loss")
):
    raise ImportError("Unable to expose scikit-learn loss functions for legacy pickle compatibility.")


