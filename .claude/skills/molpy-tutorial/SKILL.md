---
name: molpy-tutorial
description: Write textbook-style tutorial documentation for human readers. Use when creating User Guide pages, concept explanations, or worked-example notebooks.
argument-hint: "<module, concept, or existing doc path>"
user-invocable: true
---

Write a tutorial page for: $ARGUMENTS

Target audience: **human MolPy users** — chemists and simulation engineers learning the library.
Output location: `docs/user-guide/` as `.md` or `.ipynb`.

---

## Writing Rules (mandatory)

Follow the narrative arc — not all sections are required, but the order is fixed:

1. **Problem** — open with the real-world workflow pain point. Why does this concept exist?
2. **Key Idea** — one bolded definition sentence: **"A X is ..."**. No implementation detail.
3. **What it is NOT** — one short paragraph to cut obvious misconceptions.
4. **Mental Model** — an analogy. How should the reader think about this?
5. **How It Works** — explain the essential mechanism. Minimal API detail.
6. **Workflow Example** — one continuous end-to-end scenario, not isolated snippets.
7. **When to Use** — decision rules: this vs. alternatives.
8. **Extending** — briefly show how users can build on the concept.
9. **See Also** — links to API reference, related guide pages.

**Critical style rules** (violations are WRONG, fix immediately):

- Document reads as continuous narrative, NOT an outline or bullet list
- Section titles are statements of insight, not labels: "Reactions preserve atom identity" not "Reactions"
- Every code block is preceded by an explanation of WHY. Never open a section with code.
- No parameter tables in the narrative — those belong in `docs/api/`
- No tool/class catalogues — "which tools exist" is API reference territory
- No ToolRegistry content — that is for agent internals, never user docs
- Voice: technical-authoritative, active voice, vary sentence length
- Avoid: "simply", "just", "easily", "now let's", monotonous parallel structure, emoji

**Code example conventions:**
- Top-level import: `import molpy as mp`
- Submodule: `from molpy.reacter import Reacter`
- Expected output as comment: `# -> [Atom: C, Atom: O]`
- Max 15 lines per block
- Consistent names: `mol`, `ff`, `frame`, `traj`, `builder`, `rxn`

---

## Steps

**Step 1 — Understand the target**

Read the source module and any existing doc pages for `$ARGUMENTS`. Use `mcp__molpy__get_source` and `mcp__molpy__list_symbols` to inspect the API.

**Step 2 — Draft outline (internal only)**

Map the narrative arc sections to actual content. Identify: the core pain point, one anchor analogy, the best single end-to-end example scenario.

**Step 3 — Write the page**

Use the `molpy-documenter` agent to draft. Pass it the narrative arc, the style rules above, and the key example scenario.

**Step 4 — Self-review checklist**

Before returning the doc, verify:
- [ ] Opens with a problem, not a definition or code
- [ ] One bolded definition sentence exists and is standalone
- [ ] No section is a bulleted list of API items
- [ ] Every code block is preceded by prose explaining why
- [ ] Section titles read as insight statements
- [ ] No ToolRegistry or agent-internal content

**Output**: the written doc file path + a one-line summary of what the page teaches.
