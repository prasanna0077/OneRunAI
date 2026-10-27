"""Single source of truth for every module. Navigation, roadmap and architecture are built from this list."""
from __future__ import annotations

from onerun.modules.acquisition_module import AcquisitionModule
from onerun.modules.integration_module import IntegrationModule
from onerun.modules.planned import ExecutiveModule, planned_modules
from onerun.modules.retention_module import RetentionModule

MVP_MODULES = [AcquisitionModule(), IntegrationModule(), RetentionModule()]
PHASE_MODULES = [ExecutiveModule()] + planned_modules()
ALL_MODULES = MVP_MODULES + PHASE_MODULES


def by_key(key: str):
    return next(m for m in ALL_MODULES if m.key == key)


def by_phase(phase: int):
    return [m for m in ALL_MODULES if m.phase == phase]
