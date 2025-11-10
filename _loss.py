"""
Compatibility shim for legacy scikit-learn pickles.

Some serialized estimators reference the private module `_loss` at the
top-level (a packaging quirk of certain scikit-learn versions). When the
module cannot be resolved, loading fails with `ModuleNotFoundError: No
module named '_loss'`. To retain forward compatibility on deployment
targets, we expose that name and delegate to the canonical implementation
within scikit-learn.
"""

try:
    # Newer scikit-learn versions expose compiled losses under sklearn._loss._loss
    from sklearn._loss._loss import *  # type: ignore # noqa: F401,F403
except ImportError:  # pragma: no cover
    try:
        from sklearn._loss import *  # type: ignore # noqa: F401,F403
    except ImportError:
        # Fallback for very old builds.
        from sklearn.metrics._loss import *  # type: ignore # noqa: F401,F403


