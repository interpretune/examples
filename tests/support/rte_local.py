"""The RTE component's entrypoint as this repository's working tree has it, for checks of code not yet published.

``tests/support/rte_component.py`` resolves the entrypoint from the components cache, which holds the published
revision: a check of an entrypoint change would then run against the code the change replaces. These checks import
``components/rte/rte_boolq.py`` from the tree under a private module name, so they test the tree that a publish of
``components/rte`` would ship. Names resolve lazily (PEP 562), so ``from tests.support.rte_local import X`` works.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ENTRYPOINT = Path(__file__).resolve().parents[2] / "components" / "rte" / "rte_boolq.py"
RTE_COMPONENT = ENTRYPOINT.parent
_MODULE_NAME = "_examples_rte_boolq_working_tree"


@functools.cache
def rte_local_entrypoint() -> ModuleType:
    """Import ``components/rte/rte_boolq.py`` from the working tree, once per process."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(rte_local_entrypoint(), name)


def working_tree_class_paths(config):
    """``config`` with every ``rte_boolq.<Class>`` class path pointed at the working-tree entrypoint.

    A run configuration names the entrypoint's classes ``rte_boolq.<Class>``, which interpretune resolves through the
    components cache, so a check that instantiates the configuration would build the published classes rather than the
    ones under test.
    """
    if isinstance(config, dict):
        return {
            key: (
                f"{rte_local_entrypoint().__name__}.{value.removeprefix('rte_boolq.')}"
                if key == "class_path" and isinstance(value, str) and value.startswith("rte_boolq.")
                else working_tree_class_paths(value)
            )
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [working_tree_class_paths(value) for value in config]
    return config
