"""Allow ``python -m carl`` to invoke the CLI.

When a user installs ``carl-loop`` via ``pip install --user`` the
``carl`` entry-point script lands in a directory (e.g.
``~/.local/bin`` or ``~/Library/Python/3.x/bin`` on macOS) that may not
be on ``PATH``. ``python -m carl`` always works regardless because
it goes through Python's module-import mechanism rather than a shell
binary. This file is the standard pattern.
"""

from __future__ import annotations

from carl.cli import main

if __name__ == "__main__":  # pragma: no cover
    main()
