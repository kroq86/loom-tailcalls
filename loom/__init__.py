"""Async tail-call optimization primitives for Python."""

__version__ = "0.2.0"

from .tailcalls import TailCallError, explain_tailcalls, tailrec, tailstream

__all__ = ["TailCallError", "__version__", "explain_tailcalls", "tailrec", "tailstream"]
