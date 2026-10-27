"""Plug-in contract for every OneRun intelligence module.

To add a module:  1) subclass BaseModule (or PlannedModule for a placeholder)
                  2) implement render()
                  3) append an instance in onerun/modules/registry.py
The navigation, roadmap, home page and architecture page pick it up automatically.
"""
from __future__ import annotations


class BaseModule:
    key: str = ""                 # unique, url-safe
    title: str = ""
    icon: str = "🧩"
    stage: str = "Across the Lifecycle"     # Before / During / After the Deal / Across the Lifecycle
    phase: int = 1                # 1 = MVP, 2, 3, 4 = backlog
    status: str = "planned"       # live | foundation | planned
    question: str = ""            # the key business question
    engines: str = ""             # AI / ML engine(s)
    description: str = ""
    business_value: list = []

    @property
    def is_live(self) -> bool:
        return self.status == "live"

    def render(self) -> None:          # pragma: no cover - implemented by subclasses
        raise NotImplementedError
