"""
Plugin Registry for extensible image enhancement methods.

Adapted verbatim from CellTracker. Pages and workers ask the registry
which plugins exist; plugins declare their parameters via `ParamSpec`
so the GUI can build forms automatically.

Usage:
    @EnhancementPlugin.register
    class CLAHEEnhancement(EnhancementPlugin):
        name = "CLAHE"
        description = "Contrast Limited Adaptive Histogram Equalization"
        def get_params(self): ...
        def execute(self, volume, params, progress_cb=None): ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np


@dataclass
class ParamSpec:
    """Describes a single parameter for a plugin.

    `param_type` is one of: "float", "int", "bool", "choice", "str".
    `choices` is required when param_type == "choice".
    """
    name: str
    label: str
    param_type: str = "float"
    default: Any = 0.0
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    choices: List[str] = field(default_factory=list)
    tooltip: str = ""


class PluginBase(ABC):
    """Base class for all ND2Studios plugins."""
    name: str = "Unnamed"
    description: str = ""
    category: str = "general"

    _registry: Dict[str, Type["PluginBase"]] = {}

    @classmethod
    def register(cls, plugin_cls):
        """Decorator: register a plugin subclass under its category:name key."""
        key = f"{plugin_cls.category}:{plugin_cls.name}"
        PluginBase._registry[key] = plugin_cls
        return plugin_cls

    @classmethod
    def get_plugins(cls, category: str) -> List[Type["PluginBase"]]:
        return [v for k, v in PluginBase._registry.items()
                if k.startswith(f"{category}:")]

    @classmethod
    def get_plugin(cls, category: str, name: str) -> Optional[Type["PluginBase"]]:
        return PluginBase._registry.get(f"{category}:{name}")

    @abstractmethod
    def get_params(self) -> List[ParamSpec]:
        ...

    @abstractmethod
    def execute(self, data: Any, params: Dict[str, Any],
                progress_cb=None) -> Any:
        ...


class EnhancementPlugin(PluginBase):
    """Base class for image-enhancement plugins.

    Input/output: (T, H, W) numpy array. Plugins MUST be pure: no Qt,
    no global state. They take data + params + optional `progress_cb`
    and return an array of the same shape (dtype may change).
    """
    category = "enhancement"

    @abstractmethod
    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        ...
