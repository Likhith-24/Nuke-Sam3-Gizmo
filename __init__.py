# H2_SamViT_Gizmo — SAM3/SAM2 + MatAnyone2 Gizmo for Nuke
# Copyright (C) 2026
#
# AI-powered segmentation (SAM3/SAM2) and alpha matting (MatAnyone2)
# for production-quality alpha mattes inside Nuke.
#
# v3: MatAnyone2 replaces ViTMatte. Simple XY knobs for points/bbox.

__version__ = "3.0.0"
__author__ = "H2"

import os

# Get the directory where this package is installed
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
GIZMO_DIR = os.path.join(PACKAGE_DIR, "gizmos")
ICONS_DIR = os.path.join(PACKAGE_DIR, "icons")

# ── Bootstrap the virtual-environment packages into Nuke's Python ──
# This MUST happen before anything tries to import torch / transformers.
try:
    from . import env_bootstrap
    _env_ok = env_bootstrap.bootstrap(verbose=True)
    if _env_ok:
        env_bootstrap.check_packages(verbose=True)
    else:
        print("[H2_SamViT_Gizmo] ML packages unavailable — run install.py first.")
except Exception as _exc:
    print(f"[H2_SamViT_Gizmo] env_bootstrap failed: {_exc}")
    _env_ok = False

# ── Nuke registration ──
try:
    import nuke
    _IN_NUKE = True
except ImportError:
    _IN_NUKE = False


def register():
    """Register the gizmo with Nuke."""
    if not _IN_NUKE:
        return

    nuke.pluginAddPath(GIZMO_DIR)

    if os.path.exists(ICONS_DIR):
        nuke.pluginAddPath(ICONS_DIR)

    # Register knob-changed handlers (v3)
    from . import callbacks_v3  # noqa: F811

    print(f"[H2_SamViT_Gizmo] Registered v{__version__} (SAM3/SAM2 + MatAnyone2)")


def unregister():
    """Unregister the gizmo from Nuke."""
    print("[H2_SamViT_Gizmo] Unregistered")
