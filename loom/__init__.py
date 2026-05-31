"""Async tail-call optimization primitives for Python."""

from .tailcalls import TailCallError, explain_tailcalls, tailrec, tailstream

__all__ = ["TailCallError", "explain_tailcalls", "tailrec", "tailstream"]
