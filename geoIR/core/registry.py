"""Central registry inspired by ARCH.md."""
from collections import defaultdict
from typing import Callable, Dict


class Registry(defaultdict):
    """Nested dict: group -> name -> callable."""

    def __init__(self):
        super().__init__(dict)

    def register(self, group: str):
        """Decorator usage: `@registry.register("encoder")("default")`"""

        def _group_decorator(name: str):
            def _fn_decorator(fn: Callable):
                self[group][name] = fn
                return fn

            return _fn_decorator

        return _group_decorator

    def __repr__(self) -> str:  # pragma: no cover
        return f"Registry(groups={list(self.keys())})"


registry: "Registry[str, Dict[str, Callable]]" = Registry()
