# Policy on Critical Bugs and Scientific Correctness

## Scope and Motivation

MolPy is developed as a **scientific modeling framework**.
Its primary responsibility is to support **physically meaningful, reproducible, and methodologically sound simulations**.

This policy concerns the identification and handling of **critical defects** that may compromise:

* **Scientific correctness**
  (e.g. incorrect physical models, flawed algorithms, invalid assumptions)

* **Numerical validity**
  (e.g. unstable schemes, hidden unit inconsistencies, silent numerical errors)

* **Reproducibility**
  (e.g. undocumented sources of non-determinism, inconsistent defaults)

* **Data and topology integrity**
  (e.g. incorrect atom mappings, broken connectivity, invalid force-field assignments)

* **Interpretability of results**
  (e.g. misleading outputs that appear plausible but are scientifically incorrect)

This policy does **not** apply to feature requests, performance optimizations, or minor usability issues.

---

## Reporting Critical Bugs or Scientific Issues

**Please do not report critical scientific or methodological issues via public issue trackers.**

If you identify a **fatal bug, methodological flaw, or scientific inconsistency**, please contact the maintainers privately at:

📧 **[integrity@molcrafts.org](mailto:integrity@molcrafts.org)**

A report should include, where possible:

1. **Description of the issue**
   A clear explanation of what is incorrect and why it is scientifically or methodologically significant.

2. **Affected components**
   For example: structure builders, reaction templates, force-field handling, analysis workflows, or numerical routines.

3. **Reproduction or evidence**

   * Minimal scripts or configurations
   * Counterexamples
   * Analytical arguments
   * Comparisons with reference implementations or literature

4. **Observed vs expected behavior**
   Especially important for numerical or physical discrepancies.

5. **Scientific impact**
   Including whether:

   * Results may be quantitatively wrong
   * Qualitative trends may be misleading
   * Published or shared conclusions could be affected

6. **Suggested correction or references** (optional)
   Patches, alternative formulations, or relevant literature are welcome but not required.

---

## Assessment and Response

All reported issues will be evaluated with respect to:

* Severity and scope of scientific impact
* Whether the issue is conceptual, algorithmic, or implementation-related
* Potential implications for reproducibility and prior results

Depending on the outcome, responses may include:

* Code corrections or refactoring
* Explicit documentation of limitations or assumptions
* Deprecation of affected functionality
* Release notes or technical advisories describing the issue and its resolution

---

## Transparency and Disclosure

MolPy follows a **science-first disclosure principle**:

* Scientifically significant issues will be documented once understood
* Corrections will be released together with clear explanations
* If prior results may be affected, this will be stated explicitly

Our objective is not to obscure errors, but to ensure that users can **evaluate, reproduce, and trust** the results produced with the framework.

---

## Responsibility of Users

Users of MolPy are encouraged to:

* Validate results against physical intuition and reference methods
* Inspect assumptions and defaults used in workflows
* Treat automated outputs as scientific hypotheses, not ground truth
* Report suspected inconsistencies, even when uncertain

---

## Commitment to Scientific Integrity

MolPy is intended to be a **transparent and inspectable scientific tool**, not a black box.
We consider the identification and correction of critical bugs and scientific errors to be a core part of responsible research software development.

We appreciate the community’s role in maintaining scientific rigor and reproducibility.
