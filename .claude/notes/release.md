# Release

1. **molrs first** — master + tag `vX.Y.Z` + Publish green (PyPI must include Pyodide wheel if browsers matter).
2. Bump molpy to the same **major.minor**, pin `molcrafts-molrs>=X.Y.0,<X.(Y+1)`.
3. Tag molpy `vX.Y.Z` → Release workflow (trusted publishing).

No publish helper scripts; workflows only.
