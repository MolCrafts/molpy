---
name: molpy-docs
description: Docstring completeness, unit annotation, and scientific reference audit for MolPy source and docs. Covers agent-first API pages, human-first tutorials, and recipe docs. Use before PR submission, after implementing a feature, or when docstrings are missing or incomplete.
argument-hint: "[path or module]"
user-invocable: true
---

Audit documentation for: $ARGUMENTS

If no path given, check all files modified in `git diff --name-only HEAD`.

---

# Classification

First classify each file. Different standards apply — do NOT mix them.

- **Agent-first**: `src/`, API reference pages, recipes/use-case pages
- **Human-first**: concept pages, tutorials, guides

---

# Docstrings (`src/`)

Agent-first. Every public function/class/method MUST have:

```python
def func(...):
    """One-line description.

    Preferred for:
        - task-oriented usage scenarios

    Avoid when:
        - common misuse cases

    Args:
        r: Bond distances in A, shape (n_bonds,).

    Returns:
        Energies in kcal/mol, shape (n_bonds,).

    Raises:
        ValueError: If r contains negative values.

    Related:
        - other_symbol_name

    Notes:
        - constraints, edge cases
    """
```

Checks:

* Every public symbol has docstring
* Has Summary, Args, Returns at minimum; Preferred for / Avoid when for non-trivial APIs
* Type hints complete
* Physical quantities have units (A, kcal/mol, radians, etc.)
* Array shapes specified
* At least 1 Related symbol for non-trivial APIs
* No vague language ("flexible", "powerful") — use concrete usage

---

# Agent-first docs

Principle: structured, scannable, copy-pasteable. An agent must understand usage without reading prose.

## Required fields per symbol

1. Summary (one sentence)
2. Signature
3. Preferred for / Avoid when
4. Inputs / Outputs (types, units, shapes)
5. Related symbols
6. Minimal example (<=15 lines, canonical usage)

## Recipes must be structured

```text
Task: <user intent>
Summary: <one sentence>
Steps: 1. ... 2. ... 3. ...
Symbols: <exact names>
Inputs / Outputs: <types>
Example: <runnable code>
Caveats: <edge cases>
```

## Forbidden

* Long narrative, concept teaching, storytelling
* Parameter tables duplicated from docstring

## Checks

* Can agent understand usage without reading prose?
* Can agent distinguish this API vs alternatives?
* Is example copy-pasteable?
* Are symbols explicitly listed?
* Is workflow complete (no missing steps)?

---

# Human-first docs

Principle: narrative explanation that builds understanding. Designed for human readers, not retrieval.

---

## Required conceptual flow (flexible but ordered)

1. Problem — real workflow or pain point
2. Key Idea — ONE bold sentence defining the concept
3. What it is NOT — prevent common misconceptions
4. Mental Model — how to think about it
5. How It Works — core mechanism (minimal API detail)
6. Workflow Example — continuous scenario (not snippets)
7. When to Use — decision rules
8. Extending — how to build on it
9. See Also — links to API / recipes

Not all sections required, but the explanation must follow a logical progression.

---

## Writing style (MANDATORY)

### Narrative quality

* Must read as continuous & textbook-like explanation, not section checklist
* Each section should logically lead to the next
* Avoid "template filling" — empty sections are worse than missing ones

---

### Sentence style

* Mix short and long sentences (avoid monotone rhythm)
* Use active voice: "MolPy constructs the topology" not "The topology is constructed"
* Be direct: "Use X when Y" not "You might want to consider using X"

---

### Author voice

* Allowed and encouraged: "This matters because...", "In practice...", "The key idea is..."
* Avoid robotic tone or overly formal academic phrasing

---

### Clarity rules

* Define each concept ONCE, clearly
* No vague claims: "more flexible", "more modular" — explain what actually changes
* Use precise terminology: topology (not "connections"), atom type (not "kind"), struct (MolPy class, not generic "structure")

---

### Code usage rules

* Code always comes AFTER explanation
* Never start a section with code
* Each code block must answer a WHY introduced above
* Prefer one continuous example over many fragments
* Keep blocks <=15 lines

---

### Forbidden patterns

* Bullet-dump explanations
* API parameter tables
* Listing all functions/classes
* Repeating docstring content
* Introducing ToolRegistry or internal systems

---

## Checks

* Does it read like a coherent explanation (not a template)?
* Does each section logically connect?
* Is the key idea clearly defined in one sentence?
* Is code introduced with context?
* Can a human understand the concept without reading API docs?
* Does it connect to a real workflow (not abstract explanation)?

---

# Examples (cross-cutting)

Checks:

* <=15 lines
* Realistic & Runnable
* Concise comments
