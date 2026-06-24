import os

try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib

EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_toml_path = os.path.join(EXTENSION_DIR, "pyproject.toml")
with open(_toml_path, "rb") as f:
    try:
        extension_metadata = tomllib.load(f)
    except TypeError:
        import toml
        extension_metadata = toml.load(_toml_path)

__version__ = extension_metadata["project"]["version"]

from .tasks import *  # noqa: F401, F403
