# LinkedIn post (RU) — copy/paste ready

Use the block below as the main post. Attach a terminal screenshot from
`python3 examples/deep_agent_loop_comparison.py` (see `docs/linkedin-comment-demo.txt`).

---

Python-агенты ломаются не на LLM. Они ломаются на цикле.

Ты пишешь agent loop так, как он *должен* выглядеть:

state → await step → next state → … → result

В коде это естественно читается как tail-recursion:

```python
from loom import tailrec

@tailrec
async def agent_loop(state):
    if state.done:
        return state
    event = await run_next_step(state)
    return await agent_loop(state.apply(event))
```

Красиво. Явно. Как state machine.

Проблема: в CPython async-рекурсия — это реальный call stack. 10 шагов — ок. 10 000 — RecursionError. Для agent/workflow/runtime loops это не edge case, это вторник.

Loom (loom-tailcalls на PyPI) — маленькая библиотека, которая на этапе декорации переписывает tail-position `return await fn(...)` в while с O(1) stack. Семантика сохраняется; неподдерживаемые формы отклоняются, а не «оптимизируются наугад».

pip install loom-tailcalls

Это не LangGraph, не Temporal, не ещё один agent framework. Это слой ниже: stack-safety compiler для async tail-recursive state machines. Плюс @tailstream для streaming-агентов.

Benchmark на 100k шагов: ~1.1–1.2× overhead vs ручной while. Мы не обещаем, что будем быстрее цикла. Мы обещаем, что не съедим стек, пока ты пишешь код в правильной форме.

GitHub: https://github.com/kroq86/loom-tailcalls
PyPI: https://pypi.org/project/loom-tailcalls/

---

А теперь — для тех, кто дочитал до сюда и всё ещё уверен, что «понял».

Loom — это не «декоратор с while». Это semantics-preserving program transformation над узким фрагментом L_supported ⊂ AsyncPython.

Пусть конфигурация машины:

Σ = Env × Store × C,  где C = A₁ × … × Aₙ

Один шаг — частичная transition function:

δ : Σ ⇀ E* × (R + Σ + X)

где E* — trace наблюдаемых эффектов, R — результат, X — исключения.

• δ(σ) = (τ, in_Σ(σ′))  → tail self-call
• δ(σ) = (τ, in_R(r))    → return
• δ(σ) = (τ, in_X(x))    → raise

Рекурсивная operational semantics F:

F(σ) =
  (τ, r)           если δ(σ) = (τ, in_R(r))
  (τ, raise x)     если δ(σ) = (τ, in_X(x))
  τ · F(σ′)        если δ(σ) = (τ, in_Σ(σ′))

Трансформ T порождает loop semantics G: bind(signature_f, v_args, v_kwargs) + continue, с инвариантом

Env(новый рекурсивный вызов) = Env(после rebinding + continue)

Теорема корректности (soundness):

∀P ∈ L_supported, ∀σ ∈ Σ:  sem(P)(σ) = sem(T(P))(σ)

Доказательство — индукция по числу tail transitions K:
• base: δ(σ) сразу в R или X — F и G совпадают
• step: δ(σ) = (τ, in_Σ(σ′)) — F(σ) = τ · F(σ′), G делает τ, σ ← σ′, по IH остаток совпадает

Completeness относительна: P ∈ L_supported ⇒ accept(P), но valid программы вне фрагмента reject'ятся (TailCallError) — лучше отказ, чем неверная оптизация.

Compile-time: O(N) по AST
Runtime: O(K · step_cost), stack O(1), rebinding O(n + kwargs) на transition

Для @tailstream — отдельный фрагмент: terminal pattern `async for … yield …; return`, сохранение async-generator kind даже при исчезновении observable yields.

Контракт: domain / trace / state / rejection tests + optional Ollama fuzzing только по trusted templates (модель не исполняет сгенерированный Python).

Formal core: https://github.com/kroq86/loom-tailcalls/blob/main/docs/formal-core.md

---

Если после этого абзаца ты чувствуешь лёгкую тошноту и желание вернуться к while True — добро пожаловать в целевую аудиторию.

Остальным:

pip install loom-tailcalls

#Python #AsyncIO #AIAgents #OpenSource #PyPI

---

## First comment (paste under the post)

Demo — plain recursion vs Loom at 100 000 steps:

```
plain recursion failed: RecursionError
loom recursion survived: AgentState(remaining=0, total=100000)
```

Run locally:

python3 examples/deep_agent_loop_comparison.py
