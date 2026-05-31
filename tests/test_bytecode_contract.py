import asyncio
import dis
import sys
import unittest
from collections.abc import Callable

from loom import tailrec, tailstream


@tailrec
async def bytecode_countdown(n: int, acc: int = 0) -> int:
    if n <= 0:
        return acc
    return await bytecode_countdown(n - 1, acc + 1)


@tailstream
async def bytecode_stream(n: int):
    if n <= 0:
        yield ("final", n)
        return
    yield ("token", n)
    async for item in bytecode_stream(n - 1):
        yield item
    return


def opcode_names(fn: Callable[..., object]) -> list[str]:
    return [instruction.opname for instruction in dis.get_instructions(fn)]


def opcode_argvals(fn: Callable[..., object]) -> list[object]:
    return [instruction.argval for instruction in dis.get_instructions(fn)]


def has_argval(fn: Callable[..., object], value: object) -> bool:
    for argval in opcode_argvals(fn):
        if argval == value:
            return True
        if isinstance(argval, tuple) and value in argval:
            return True
    return False


def traced_opcodes(fn: Callable[..., object], run: Callable[[], object]) -> list[str]:
    offset_to_opname = {
        instruction.offset: instruction.opname for instruction in dis.get_instructions(fn)
    }
    seen: list[str] = []

    def tracer(frame, event, arg):
        if frame.f_code is fn.__code__:
            frame.f_trace_opcodes = True
            if event == "opcode":
                opname = offset_to_opname.get(frame.f_lasti)
                if opname is not None:
                    seen.append(opname)
        return tracer

    old_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        run()
    finally:
        sys.settrace(old_trace)
    return seen


class TestBytecodeContract(unittest.TestCase):
    def test_tailrec_bytecode_has_loop_not_self_call(self) -> None:
        names = opcode_names(bytecode_countdown)
        argvals = opcode_argvals(bytecode_countdown)

        self.assertIn("JUMP_BACKWARD", names)
        self.assertTrue(has_argval(bytecode_countdown, "__loom_next_0"))
        self.assertTrue(has_argval(bytecode_countdown, "__loom_next_1"))
        self.assertNotIn("__loom_bind", argvals)
        self.assertNotIn("__loom_signature", argvals)
        self.assertNotIn("bytecode_countdown", argvals)

    def test_tailstream_bytecode_has_loop_and_remains_async_generator(self) -> None:
        names = opcode_names(bytecode_stream)
        argvals = opcode_argvals(bytecode_stream)

        self.assertIn("JUMP_BACKWARD", names)
        self.assertIn("YIELD_VALUE", names)
        self.assertTrue(has_argval(bytecode_stream, "__loom_next_0"))
        self.assertNotIn("__loom_bind", argvals)
        self.assertNotIn("bytecode_stream", argvals)

    def test_runtime_opcode_trace_hits_loop_and_return(self) -> None:
        seen = traced_opcodes(
            bytecode_countdown,
            lambda: asyncio.run(bytecode_countdown(3)),
        )

        self.assertIn("JUMP_BACKWARD", seen)
        self.assertIn("RETURN_VALUE", seen)


if __name__ == "__main__":
    unittest.main()
