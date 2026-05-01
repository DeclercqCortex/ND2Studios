"""
Image-processing recipes: save/load a reusable preprocessing pipeline
(frame-mean normalization flag + ordered list of enhancement steps) as
a portable JSON file.

Crop/ROI and per-dataset state are intentionally excluded — a recipe is
meant to be applied to any dataset.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from nd2studios.core.plugin_registry import PluginBase


RECIPE_VERSION = 1
RECIPE_EXTENSION = ".nd2s_recipe.json"


def build_recipe(
    pipeline: List[Tuple[str, Dict[str, Any]]],
    normalized: bool,
    name: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Serialize the in-memory recipe into a JSON-ready dict."""
    return {
        "version": RECIPE_VERSION,
        "kind": "nd2studios.recipe",
        "name": name,
        "notes": notes,
        "normalized": bool(normalized),
        "pipeline": [
            {"name": step_name, "params": dict(params)}
            for step_name, params in pipeline
        ],
    }


def save_recipe(
    path: str,
    pipeline: List[Tuple[str, Dict[str, Any]]],
    normalized: bool,
    name: str = "",
    notes: str = "",
) -> None:
    """Write the recipe to ``path`` as indented JSON."""
    recipe = build_recipe(pipeline, normalized, name=name, notes=notes)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)


def load_recipe(path: str) -> Dict[str, Any]:
    """
    Read a recipe JSON file and return its dict.

    Raises ``ValueError`` with a readable message if the file doesn't look
    like a recipe produced by this app.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Recipe file is not a JSON object.")
    if data.get("kind") != "nd2studios.recipe":
        raise ValueError("File is not a CellTracker recipe.")
    if data.get("version", 0) > RECIPE_VERSION:
        raise ValueError(
            f"Recipe version {data.get('version')} is newer than this "
            f"build supports (max {RECIPE_VERSION})."
        )
    if not isinstance(data.get("pipeline", []), list):
        raise ValueError("Recipe 'pipeline' field must be a list.")
    return data


def validate_recipe_plugins(recipe: Dict[str, Any]) -> List[str]:
    """
    Return a list of plugin names referenced by the recipe that are NOT
    registered on the current build. Empty list means fully compatible.
    """
    missing = []
    for step in recipe.get("pipeline", []):
        name = step.get("name")
        if not name:
            continue
        if PluginBase.get_plugin("enhancement", name) is None:
            missing.append(name)
    return missing


def recipe_to_pipeline(
    recipe: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Convert a loaded recipe dict into the in-memory ``_pipeline`` shape."""
    return [
        (step["name"], dict(step.get("params", {})))
        for step in recipe.get("pipeline", [])
        if "name" in step
    ]
