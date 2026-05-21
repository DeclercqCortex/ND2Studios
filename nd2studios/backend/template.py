"""
Pipeline template save/load for ND2Studios Batch processing (V1.23).

A .nd2st.json template captures the full pipeline configuration
(import, recipe, analysis, results) without any file-specific data.
Use ``save_template`` + ``write_template`` to persist, ``load_template``
to restore.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

TEMPLATE_EXTENSION = ".nd2st.json"
_VERSION = "1.0"


def save_template(
    name: str,
    import_config: Dict[str, Any],
    recipe: List[Tuple[str, Dict[str, Any]]],
    recipe_normalized: bool,
    analysis_config: Dict[str, Any],
    results_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a template dict from pipeline configuration.

    Args:
        name:              human-readable template name
        import_config:     dict from exp.import_config  (z_mode, frame_stride…)
        recipe:            list of (plugin_name, params_dict) tuples
        recipe_normalized: whether frame-mean normalisation is applied
        analysis_config:   dict from exp.analysis_config (pipeline_name, params)
        results_config:    dict from exp.results_config  (image_format…)

    Returns:
        dict suitable for ``write_template``.
    """
    return {
        "nd2s_template_version": _VERSION,
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "pipeline": {
            "import": {
                "z_mode": import_config.get("z_projection", "max"),
                "frame_stride": int(import_config.get("t_stride", 1)),
            },
            "recipe": [
                {"name": n, "params": dict(p)} for n, p in recipe
            ],
            "recipe_normalized": bool(recipe_normalized),
            "analysis": {
                "pipeline_name": (
                    analysis_config.get("pipeline_name")
                    or analysis_config.get("pipeline", "")
                ),
                "params": dict(analysis_config.get("params", {})),
            },
            "results": {
                "image_format": results_config.get("image_format", "tiff"),
            },
        },
    }


def write_template(template: Dict[str, Any], filepath: str) -> str:
    """Write *template* to *filepath* (adds extension if missing).

    Returns the final path written.
    """
    if not filepath.endswith(TEMPLATE_EXTENSION):
        filepath += TEMPLATE_EXTENSION
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    return filepath


def load_template(filepath: str) -> Dict[str, Any]:
    """Load and validate a .nd2st.json template.

    Raises:
        ValueError  if the file is not a recognised ND2Studios template.
        FileNotFoundError if the path does not exist.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    version = data.get("nd2s_template_version")
    if version != _VERSION:
        raise ValueError(
            f"Unsupported template version: {version!r}  "
            f"(expected {_VERSION!r})"
        )
    return data
