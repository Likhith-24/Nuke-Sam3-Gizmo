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

# On Windows, packages live in a flat target directory instead of a venv
PYTHON_PACKAGES_DIR = PACKAGE_DIR / "python_packages"

# Packages that MUST be importable for inference to work
REQUIRED_MODULES = ["torch", "transformers", "cv2", "PIL", "scipy", "psutil", "sam2", "sam3", "einops"]

# Nice-to-have but not blocking (torchvision may fail to import inside Nuke
# due to shared-library conflicts, yet our code never imports it directly)
OPTIONAL_MODULES = ["torchvision"]


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


def _find_bin_dir(venv_path: Path) -> str:
    """Find the bin/Scripts directory of the venv."""
    for name in ["bin", "Scripts"]:
        d = venv_path / name
        if d.exists():
            return str(d)
    return ""


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

    # 1. Check for a virtual environment (Linux / macOS primary path)
    venv_path = get_venv_path()
    if venv_path.exists():
        sp = _find_site_packages(venv_path)
        site_packages.extend(sp)
        source_label = str(venv_path)

    # 2. Check for a target-directory install (Windows / embedded path)
    if PYTHON_PACKAGES_DIR.exists():
        pkg_str = str(PYTHON_PACKAGES_DIR)
        if pkg_str not in site_packages:
            site_packages.append(pkg_str)
            if not source_label:
                source_label = pkg_str

    if not site_packages:
        if verbose:
            print("[H2 SamViT] No package environment found.")
            if IS_WINDOWS:
                print(f"[H2 SamViT] Expected target dir: {PYTHON_PACKAGES_DIR}")
            else:
                print(f"[H2 SamViT] Expected venv at: {venv_path}")
            py_cmd = "python install.py" if IS_WINDOWS else f'python3 "{PACKAGE_DIR / "install.py"}"'
            print(f"[H2 SamViT] Run the install script first:")
            print(f"[H2 SamViT]   {py_cmd}")
        return False

    # Inject site-packages into sys.path (at the front so they take priority)
    injected = 0
    for sp in site_packages:
        if sp not in sys.path:
            sys.path.insert(0, sp)
            injected += 1
            if verbose:
                print(f"[H2 SamViT] Added to sys.path: {sp}")

    # Also add the venv's bin directory to PATH for subprocess calls
    if venv_path.exists():
        bin_dir = _find_bin_dir(venv_path)
        if bin_dir:
            current_path = os.environ.get("PATH", "")
            if bin_dir not in current_path:
                os.environ["PATH"] = bin_dir + os.pathsep + current_path
        # Set environment variables that some packages need
        os.environ["VIRTUAL_ENV"] = str(venv_path)

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
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "unknown")
            results[mod_name] = {"available": True, "version": version}
        except ImportError:
            results[mod_name] = {"available": False, "version": None}

    if verbose:
        print("[H2 SamViT] Package check:")
        for name, info in results.items():
            if info["available"]:
                print(f"  ✓ {name} ({info['version']})")
            else:
                print(f"  ✗ {name} — NOT FOUND")
        # Optional packages (non-blocking)
        for mod_name in OPTIONAL_MODULES:
            try:
                mod = __import__(mod_name)
                v = getattr(mod, "__version__", "unknown")
                print(f"  ✓ {mod_name} ({v})  [optional]")
            except ImportError:
                print(f"  ⚠ {mod_name} — not found  [optional, non-blocking]")

    return results


def is_ready() -> bool:
    """Check if all required ML packages are importable."""
    for mod_name in REQUIRED_MODULES:
        try:
            __import__(mod_name)
        except ImportError:
            return False
    return True


def get_status_message() -> str:
    """Get a human-readable status message for the UI."""
    venv_path = get_venv_path()
    has_venv = venv_path.exists()
    has_pkgdir = PYTHON_PACKAGES_DIR.exists()
    py_cmd = "python install.py" if IS_WINDOWS else "python3 install.py"

    if not has_venv and not has_pkgdir:
        if IS_WINDOWS:
            return (
                f"Package environment not found.\n\n"
                f"Expected:\n  {PYTHON_PACKAGES_DIR}\n\n"
                f"To set up, run in a terminal:\n"
                f"  cd \"{PACKAGE_DIR}\"\n"
                f"  {py_cmd}"
            )
        return (
            f"Virtual environment not found.\n\n"
            f"Expected location:\n  {venv_path}\n\n"
            f"To set up, run in a terminal:\n"
            f"  cd \"{PACKAGE_DIR}\"\n"
            f"  {py_cmd}\n\n"
            f"Or set a custom path via env var:\n"
            f"  export H2_SAMVIT_VENV=/path/to/your/venv"
        )

    missing = []
    for mod_name in REQUIRED_MODULES:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(mod_name)

    if missing:
        env_label = str(PYTHON_PACKAGES_DIR) if (has_pkgdir and not has_venv) else str(venv_path)
        return (
            f"Environment found at:\n  {env_label}\n\n"
            f"Missing packages: {', '.join(missing)}\n\n"
            f"To install them, run:\n"
            f"  cd \"{PACKAGE_DIR}\"\n"
            f"  {py_cmd}"
        )

    return "Environment OK — all packages available."
