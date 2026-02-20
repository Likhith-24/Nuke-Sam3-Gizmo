# H2_SamViT_Gizmo - SAM3 + ViTMatte Gizmo for Nuke
# Copyright (C) 2026
#
# A comprehensive AI-powered segmentation and matting gizmo combining
# SAM3 (Segment Anything Model 3) with ViTMatte for high-quality alpha mattes.

__version__ = "1.1.0"
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

    # Register knob-changed handlers
    from . import callbacks  # noqa: F811

    # Install viewer click handler (Ctrl+Click to place points/bbox)
    try:
        from . import viewer_events
        viewer_events.install()
    except Exception as _e:
        print(f"[H2_SamViT_Gizmo] Viewer click handler not installed: {_e}")

    print(f"[H2_SamViT_Gizmo] Registered v{__version__}")


def unregister():
    """Unregister the gizmo from Nuke."""
    print("[H2_SamViT_Gizmo] Unregistered")
