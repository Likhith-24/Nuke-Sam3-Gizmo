# env_bootstrap.py - Virtual environment bootstrap for H2 SamViT
# Injects an external Python venv's site-packages into Nuke's sys.path
# so that torch, transformers, cv2, etc. become available inside Nuke.

import os
import sys
import glob
import json
import platform
from pathlib import Path

# Default venv location: <plugin_dir>/venv
PACKAGE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VENV_DIR = PACKAGE_DIR / "venv"
CONFIG_FILE = PACKAGE_DIR / "env_config.json"

IS_WINDOWS = platform.system() == "Windows"

# Packages importable inside Nuke's process. The torch family is NOT in this
# list on purpose: Nuke bundles its own libtorch (c10.dll — for Inference /
# CopyCat), so importing pip-torch in-process fails with WinError 1114. All
# torch work runs in worker.py via worker_client.py instead.
REQUIRED_MODULES = ["numpy", "cv2", "PIL", "scipy", "psutil"]

# Torch-family packages — verified inside the worker process, never here.
WORKER_MODULES = ["torch", "torchvision", "transformers", "sam2", "sam3", "einops"]


def _read_config() -> dict:
    """Read the environment config file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _write_config(cfg: dict) -> None:
    """Write the environment config file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[H2 SamViT] Warning: Could not write config: {e}")


def get_venv_path() -> Path:
    """
    Get the configured virtual environment path.
    Priority: env_config.json > H2_SAMVIT_VENV env var > default ./venv
    """
    # 1. Config file
    cfg = _read_config()
    if "venv_path" in cfg:
        p = Path(cfg["venv_path"])
        if p.exists():
            return p

    # 2. Environment variable
    env_path = os.environ.get("H2_SAMVIT_VENV")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # 3. Default
    return DEFAULT_VENV_DIR


def set_venv_path(path: str) -> None:
    """Set a custom venv path in the config file."""
    cfg = _read_config()
    cfg["venv_path"] = str(path)
    _write_config(cfg)
    print(f"[H2 SamViT] Venv path set to: {path}")


def _find_site_packages(venv_path: Path) -> list:
    """Find site-packages directories inside a venv."""
    candidates = []

    # Standard venv layout: venv/lib/pythonX.Y/site-packages
    patterns = [
        venv_path / "lib" / "python*" / "site-packages",
        venv_path / "Lib" / "site-packages",           # Windows
        venv_path / "lib" / "site-packages",            # Some layouts
    ]

    for pattern in patterns:
        matches = glob.glob(str(pattern))
        candidates.extend(matches)

    # Conda environment layout
    conda_sp = venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if conda_sp.exists() and str(conda_sp) not in candidates:
        candidates.append(str(conda_sp))

    return [p for p in candidates if os.path.isdir(p)]


def _inject_lib_dynload(venv_path: Path, verbose: bool = False) -> None:
    """Inject the base Python's lib-dynload into sys.path.

    Nuke's embedded Python may be missing C-extension modules like
    ``_lzma``, ``_sqlite3``, etc.  The venv was created from a system
    Python that ships them in its ``lib-dynload`` directory.

    We read ``pyvenv.cfg`` inside the venv to find the base Python
    home, then add its ``lib-dynload`` so those extensions become
    importable inside Nuke.
    """
    cfg_file = venv_path / "pyvenv.cfg"
    if not cfg_file.exists():
        return

    home_dir = None
    for line in cfg_file.read_text().splitlines():
        key_value = line.split("=", 1)
        if len(key_value) == 2 and key_value[0].strip().lower() == "home":
            home_dir = key_value[1].strip()
            break

    if not home_dir:
        return

    # home is e.g. /home/user/.pyenv/versions/3.11.8/bin
    # lib-dynload is at  ../lib/python3.XX/lib-dynload
    base = Path(home_dir).parent
    for libdir in sorted(base.glob("lib/python3.*/lib-dynload")):
        if libdir.is_dir() and str(libdir) not in sys.path:
            sys.path.append(str(libdir))
            if verbose:
                print(f"[H2 SamViT] Added lib-dynload: {libdir}")
            break


def bootstrap(verbose: bool = True) -> bool:
    """
    Inject package directories into sys.path so that ML packages
    (torch, transformers, etc.) become importable inside Nuke.

    Supports two layouts:
      • Linux / macOS — standard venv  (./venv)
      • Windows       — flat target dir (./python_packages)

    Call this once at startup (from __init__.py or init.py).

    Returns:
        True if packages were successfully injected, False otherwise.
    """
    site_packages: list[str] = []
    source_label = ""

    # Locate the uv-managed venv (install.py creates ./venv on every platform).
    venv_path = get_venv_path()
    if venv_path.exists():
        sp = _find_site_packages(venv_path)
        site_packages.extend(sp)
        source_label = str(venv_path)

    if not site_packages:
        if verbose:
            print(f"[H2 SamViT] No venv found at: {venv_path}")
            print("[H2 SamViT] Run the installer first (from a terminal, not Nuke):")
            print(f'[H2 SamViT]   cd "{PACKAGE_DIR}" && python install.py')
        return False

    # APPEND the venv site-packages to sys.path — do not prepend. torch runs
    # out-of-process now, so Nuke only needs the light packages (numpy, cv2,
    # PIL, psutil) from here, and appending means Nuke's own bundled packages
    # keep priority for every OTHER plugin in this shared .nuke; the venv only
    # fills the gaps. (Prepending globally shadowed Nuke's numpy for the whole
    # session — a cross-plugin hazard.)
    injected = 0
    for sp in site_packages:
        if sp not in sys.path:
            sys.path.append(sp)
            injected += 1
            if verbose:
                print(f"[H2 SamViT] Added to sys.path: {sp}")

    # NOTE: this used to also prepend the venv's bin dir to PATH and set
    # VIRTUAL_ENV. Both mutated the whole Nuke process for every other tool
    # (and a stale VIRTUAL_ENV breaks uv-based tooling); the worker builds
    # its own environment in worker_client, so neither is needed anymore.
    if venv_path.exists():
        # Nuke's Python may be missing C-extension modules (_lzma, etc.)
        # that live in the base Python's lib-dynload directory.
        _inject_lib_dynload(venv_path, verbose=verbose)

    if verbose and injected > 0:
        print(f"[H2 SamViT] Environment bootstrapped from: {source_label}")

    return True


def check_packages(verbose: bool = True) -> dict:
    """
    Check which required packages are importable.

    Returns:
        Dict mapping module name to (available: bool, version: str|None)
    """
    results = {}
    for mod_name in REQUIRED_MODULES:
        # Broad except: a clashing DLL raises OSError, not ImportError.
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "unknown")
            results[mod_name] = {"available": True, "version": version}
        except Exception:
            results[mod_name] = {"available": False, "version": None}

    if verbose:
        print("[H2 SamViT] Package check (in-process):")
        for name, info in results.items():
            if info["available"]:
                print(f"  ✓ {name} ({info['version']})")
            else:
                print(f"  ✗ {name} — NOT FOUND")
        print(
            "[H2 SamViT] torch/sam2/sam3 run out-of-process "
            "(verified by the inference worker)."
        )

    return results


def is_ready() -> bool:
    """Check if all required in-process packages are importable."""
    for mod_name in REQUIRED_MODULES:
        try:
            __import__(mod_name)
        except Exception:  # OSError on DLL clashes, not just ImportError
            return False
    return True


def get_status_message() -> str:
    """Get a human-readable status message for the UI."""
    venv_path = get_venv_path()

    if not venv_path.exists():
        return (
            f"Virtual environment not found.\n\n"
            f"Expected location:\n  {venv_path}\n\n"
            f"To set up, run in a terminal (not inside Nuke):\n"
            f'  cd "{PACKAGE_DIR}"\n'
            f"  python install.py\n\n"
            f"Or set a custom path via env var:\n"
            f"  H2_SAMVIT_VENV=/path/to/your/venv"
        )

    missing = []
    for mod_name in REQUIRED_MODULES:
        try:
            __import__(mod_name)
        except Exception:
            missing.append(mod_name)

    if missing:
        return (
            f"Environment found at:\n  {venv_path}\n\n"
            f"Missing packages: {', '.join(missing)}\n\n"
            f"To install them, run:\n"
            f'  cd "{PACKAGE_DIR}"\n'
            f"  python install.py"
        )

    return "Environment OK — all packages available."
