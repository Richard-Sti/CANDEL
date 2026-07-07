Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


Most important
--------------

1. After you complete a task, review and report the changes.

2. When there is a series of independent tasks to be completed, always deploy a series of agents.

3. After completing each major step, update this file's "Current status" section with: what was done, current state, and next steps. Remove steps that are no longer relevant.

4. Note that the instructions below apply only to Python. But the sense of it applies to other languages too.


Basics
------

1. Write clean, idiomatic Python that follows PEP8 style guidelines unless explicitly told otherwise. 

2. Prioritize efficiency and vectorization using NumPy, SciPy, or JAX where appropriate.

3. Minimize memory overhead by avoiding unnecessary copies and using in-place operations when safe.

4. Use informative variable names but keep them short when they're in math-heavy or array-heavy code (e.g. bf, r, z).

5. Add comments only where clarification is needed. Don't comment obvious things. Assume the reader is a domain expert.

6. Avoid boilerplate unless explicitly asked. 

7. For plotting:
   - Don’t add titles or colorbars unless explicitly asked.
   - Label axes with units in LaTeX if known.

8. For shell scripting:
    - Use #!/bin/bash -l shebang for SLURM jobs.
    - Prefer awk and grep for parsing files.
    - Always print useful status info unless in silent mode.

9. When given raw array data:
    - Assume it's in physical units unless told otherwise.
    - If in cosmological context, assume comoving Mpc/h and Msun/h units.

10. When unsure of a parameter or behavior, ask for clarification instead of guessing.

11. In the function signature, do not describe the variable types. Python does not use that anyway.

12. Never use conda. Always use the local venv of the package.
