"""Version information for molpy.

``version`` is the version of the native core molpy was built with
(``[workspace.package].version`` in the project's ``Cargo.toml``): the Python
package and ``molpy._core`` are one wheel, so they cannot drift apart.
"""

from molpy._core import __version__ as version

release_date = "2026-10-21"

__all__ = [
    "version",
    "release_date",
]
