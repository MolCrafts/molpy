"""Allow ``python -m molpy ...`` to invoke the CLI."""

from molpy.cli import main

raise SystemExit(main())
