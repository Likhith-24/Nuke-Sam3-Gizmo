# install.py - Installation script for H2 SamViT Gizmo
# Creates a virtual environment matching Nuke's Python version and installs
# torch, transformers, opencv, and all other ML dependencies into it.
#
# USAGE — run from a regular terminal (NOT from inside Nuke):
#
#   cd <this_directory>
#
#   python3 install.py                  # auto-detect GPU, create ./venv
#   python3 install.py --cuda 12.6      # force a specific CUDA version
#   python3 install.py --cpu            # CPU-only, no CUDA
#   python3 install.py --venv /path     # custom venv location
#
# After installation, restart Nuke. The gizmo injects the venv at startup.

import os
import sys
import shutil
import subprocess
import argparse
import json
import platform
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
DEFAULT_VENV_DIR = PACKAGE_DIR / "venv"
PYTHON_PACKAGES_DIR = PACKAGE_DIR / "python_packages"
CONFIG_FILE = PACKAGE_DIR / "env_config.json"

# Nuke 16 ships Python 3.11 — the venv MUST use a matching minor version.
REQUIRED_PYTHON_MINOR = 11  # 3.11.x

IS_WINDOWS = platform.system() == "Windows"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _run(cmd, **kwargs):
    """Run a command, streaming output."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def find_python311() -> str:
    """
    Locate a system Python 3.11 interpreter.
    On Windows also checks the ``py`` launcher and Nuke's embedded Python.
    Raises RuntimeError if none is found.
    """
    candidates = ["python3.11", "python3", "python"] if not IS_WINDOWS else [
        "python3.11", "python3", "python"
    ]
    for name in candidates:
        path = shutil.which(name)
        if not path:
            continue
        try:
            out = subprocess.check_output(
                [path, "-c", "import sys; print(sys.version_info.minor)"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if int(out) == REQUIRED_PYTHON_MINOR:
                return path
        except Exception:
            continue

    # Windows: try the ``py`` launcher with a version flag
    if IS_WINDOWS:
        py = shutil.which("py")
        if py:
            try:
                real = subprocess.check_output(
                    [py, f"-3.{REQUIRED_PYTHON_MINOR}", "-c",
                     "import sys; print(sys.executable)"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                if real and os.path.exists(real):
                    return real
            except Exception:
                pass

        # Fallback: Nuke's own embedded Python
        nuke_py = _find_nuke_python()
        if nuke_py:
            return nuke_py

    raise RuntimeError(
        f"Could not find Python 3.{REQUIRED_PYTHON_MINOR} on this system.\n"
        f"Nuke 16 embeds Python 3.{REQUIRED_PYTHON_MINOR}, so the venv must match.\n\n"
        f"Install it via your package manager or pyenv:\n"
        f"  Linux : sudo apt install python3.11 python3.11-venv\n"
        f"  macOS : brew install python@3.11\n"
        f"  Windows: https://www.python.org/downloads/release/python-3110/\n"
        f"  pyenv : pyenv install 3.{REQUIRED_PYTHON_MINOR}"
    )


def _find_nuke_python() -> str:
    """Try to locate Nuke's embedded Python (Windows fallback)."""
    # Infer Nuke root from plugin directory:
    # <NukeDir>/plugins/H2_SamViT_Gizmo/  →  <NukeDir>
    nuke_root = PACKAGE_DIR.parent.parent
    for name in ("python.exe", "python3.exe", "python"):
        p = nuke_root / name
        if p.exists():
            try:
                out = subprocess.check_output(
                    [str(p), "-c",
                     "import sys; print(sys.version_info.minor)"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                if int(out) == REQUIRED_PYTHON_MINOR:
                    return str(p)
            except Exception:
                continue
    return ""


def detect_cuda() -> str:
    """Try to detect CUDA version via nvidia-smi / nvcc. Returns '' for CPU."""
    try:
        subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL
        )
    except Exception:
        return ""

    # nvidia-smi exists → GPU present. Try nvcc for exact version.
    try:
        nvcc_out = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL
        )
        for line in nvcc_out.splitlines():
            if "release" in line.lower():
                ver = line.split("release")[-1].strip().split(",")[0].strip()
                return ver
    except Exception:
        pass

    return "12.6"  # safe default for modern drivers


def _venv_executable(venv_dir: Path, name: str) -> str:
    """Return path to an executable inside the venv (cross-platform)."""
    if IS_WINDOWS:
        exe = venv_dir / "Scripts" / (name + ".exe")
        if not exe.exists():
            exe = venv_dir / "Scripts" / name
    else:
        exe = venv_dir / "bin" / name
    if not exe.exists():
        raise FileNotFoundError(f"{name} not found in {venv_dir}")
    return str(exe)


# ──────────────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────────────

def step_create_venv(python_bin: str, venv_dir: Path) -> None:
    """Create a fresh virtual environment."""
    print(f"\n{'='*60}")
    print("[1/5] Creating virtual environment")
    print(f"{'='*60}")
    print(f"  Location : {venv_dir}")

    if venv_dir.exists():
        resp = input("  Venv already exists. Recreate from scratch? [y/N]: ").strip().lower()
        if resp == "y":
            shutil.rmtree(venv_dir)
        else:
            print("  → Keeping existing venv.\n")
            return

    _run([python_bin, "-m", "venv", str(venv_dir)])
    print("  ✓ Virtual environment created.\n")


def step_install_torch(pip_bin: str, cuda_version: str) -> None:
    """Install PyTorch (with or without CUDA)."""
    print(f"\n{'='*60}")
    print("[2/5] Installing PyTorch")
    print(f"{'='*60}")

    if not cuda_version:
        print("  Mode: CPU-only\n")
        _run([pip_bin, "install", "--no-cache-dir",
              "torch>=2.0.0", "torchvision>=0.15.0",
              "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        # Map to the closest supported CUDA wheel tag
        cuda_stripped = cuda_version.replace(".", "")
        supported = {"124": "cu124", "126": "cu126", "128": "cu128"}
        tag = None
        for key in sorted(supported.keys(), reverse=True):
            if int(cuda_stripped) >= int(key):
                tag = supported[key]
                break
        tag = tag or "cu126"

        print(f"  Mode: CUDA {cuda_version} → wheels tag {tag}\n")
        _run([pip_bin, "install", "--no-cache-dir",
              "torch>=2.0.0", "torchvision>=0.15.0",
              "--index-url", f"https://download.pytorch.org/whl/{tag}"])

    print("  ✓ PyTorch installed.\n")


def step_install_packages(pip_bin: str) -> None:
    """Install transformers, opencv, sam2, and other ML packages."""
    print(f"\n{'='*60}")
    print("[3/5] Installing ML packages")
    print(f"{'='*60}\n")

    packages = [
        "transformers>=4.30.0",
        "opencv-python-headless>=4.8.0",   # headless avoids Qt conflicts with Nuke
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
        "psutil>=5.9.0",
        "hydra-core>=1.3.0",
        "iopath>=0.1.10",
    ]
    _run([pip_bin, "install", "--no-cache-dir", "--upgrade", "pip", "wheel", "setuptools"])
    _run([pip_bin, "install", "--no-cache-dir"] + packages)

    # sam2 must be installed with --no-build-isolation to avoid
    # re-downloading torch as a build dependency.
    print("  Installing SAM2 …")
    _run([pip_bin, "install", "--no-cache-dir", "--no-build-isolation", "sam2>=1.0.0"])

    # sam3 must be installed from GitHub (not on PyPI).
    print("  Installing SAM3 …")
    _run([pip_bin, "install", "--no-cache-dir", "--no-build-isolation",
          "git+https://github.com/facebookresearch/sam3.git"])

    # SAM3 extra dependencies
    print("  Installing SAM3 dependencies …")
    _run([pip_bin, "install", "--no-cache-dir",
          "einops>=0.8.0", "decord>=0.6.0", "pycocotools>=2.0.0",
          "timm>=1.0.0", "ftfy>=6.1.0"])

    print("  ✓ ML packages installed.\n")


# ──────────────────────────────────────────────────────────────────────
# Windows: target-directory install  (no venv, packages go into a flat dir)
# ──────────────────────────────────────────────────────────────────────

def step_install_to_target(python_bin: str, target_dir: Path,
                           cuda_version: str) -> None:
    """Install all packages into a flat target directory (Windows embedded)."""

    print(f"\n{'='*60}")
    print("[1/3] Preparing target package directory")
    print(f"{'='*60}")
    print(f"  Location : {target_dir}")

    if target_dir.exists():
        resp = (
            input("  Target dir exists. Reinstall from scratch? [y/N]: ")
            .strip().lower()
        )
        if resp == "y":
            shutil.rmtree(target_dir)
        else:
            print("  → Keeping existing packages.\n")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Upgrade pip itself (not into target)
    _run([python_bin, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "pip"])

    pip_target = [python_bin, "-m", "pip", "install", "--no-cache-dir",
                  "--target", str(target_dir)]

    # ── PyTorch ──
    print(f"\n{'='*60}")
    print("[2/3] Installing PyTorch + ML packages")
    print(f"{'='*60}")

    if not cuda_version:
        print("  Mode: CPU-only\n")
        _run(pip_target + [
            "torch>=2.0.0", "torchvision>=0.15.0",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])
    else:
        cuda_stripped = cuda_version.replace(".", "")
        supported = {"124": "cu124", "126": "cu126", "128": "cu128"}
        tag = None
        for key in sorted(supported.keys(), reverse=True):
            if int(cuda_stripped) >= int(key):
                tag = supported[key]
                break
        tag = tag or "cu126"
        print(f"  Mode: CUDA {cuda_version} → wheels tag {tag}\n")
        _run(pip_target + [
            "torch>=2.0.0", "torchvision>=0.15.0",
            "--index-url", f"https://download.pytorch.org/whl/{tag}",
        ])

    # ── Other ML packages ──
    packages = [
        "transformers>=4.30.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
        "psutil>=5.9.0",
        "hydra-core>=1.3.0",
        "iopath>=0.1.10",
    ]
    _run(pip_target + packages)

    # sam2 needs --no-build-isolation to avoid re-downloading torch
    _run([python_bin, "-m", "pip", "install", "--no-cache-dir",
          "--target", str(target_dir),
          "--no-build-isolation", "sam2>=1.0.0"])

    # sam3 from GitHub
    _run([python_bin, "-m", "pip", "install", "--no-cache-dir",
          "--target", str(target_dir),
          "--no-build-isolation",
          "git+https://github.com/facebookresearch/sam3.git"])

    # SAM3 extra dependencies
    _run([python_bin, "-m", "pip", "install", "--no-cache-dir",
          "--target", str(target_dir),
          "einops>=0.8.0", "decord>=0.6.0", "pycocotools>=2.0.0",
          "timm>=1.0.0", "ftfy>=6.1.0"])

    print("  ✓ All packages installed.\n")


def step_verify_target(python_bin: str, target_dir: Path) -> bool:
    """Verify packages installed in the target directory."""
    print(f"\n{'='*60}")
    print("[3/3] Verification")
    print(f"{'='*60}\n")

    check = (
        f"import sys; sys.path.insert(0, r'{target_dir}');\n"
        "mods = ['torch','torchvision','transformers','cv2',"
        "'PIL','scipy','psutil','numpy','sam2','sam3','einops']\n"
        "ok = True\n"
        "for m in mods:\n"
        "    try:\n"
        "        mod = __import__(m); v = getattr(mod,'__version__','?')\n"
        "        print(f'  ✓ {m} ({v})')\n"
        "    except ImportError:\n"
        "        print(f'  ✗ {m} — NOT FOUND'); ok = False\n"
        "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')\n"
        "if ok: print('OK')\n"
    )
    try:
        out = subprocess.check_output(
            [python_bin, "-c", check], text=True, stderr=subprocess.STDOUT,
        )
        print(out)
        return "OK" in out
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Verification failed:\n{e.output}")
        return False


def step_setup_nuke(plugin_path: Path) -> None:
    """Append loader lines to the user's .nuke/init.py and menu.py."""
    print(f"\n{'='*60}")
    print("[4/5] Configuring Nuke startup files")
    print(f"{'='*60}\n")

    home = Path.home()
    nuke_dir = home / ".nuke"
    nuke_dir.mkdir(exist_ok=True)

    parent_dir = plugin_path.parent  # folder that contains H2_SamViT_Gizmo/

    # ── init.py ──
    marker = "# >>> H2_SamViT_Gizmo >>>"
    init_block = (
        f"\n{marker}\n"
        f"import sys, nuke\n"
        f"sys.path.insert(0, r\"{parent_dir}\")\n"
        f"nuke.pluginAddPath(r\"{plugin_path / 'gizmos'}\")\n"
        f"# <<< H2_SamViT_Gizmo <<<\n"
    )
    _append_if_missing(nuke_dir / "init.py", marker, init_block, "init.py")

    # ── menu.py ──
    menu_marker = "# >>> H2_SamViT_Gizmo menu >>>"
    menu_block = (
        f"\n{menu_marker}\n"
        f"from H2_SamViT_Gizmo import menu\n"
        f"# <<< H2_SamViT_Gizmo menu <<<\n"
    )
    _append_if_missing(nuke_dir / "menu.py", menu_marker, menu_block, "menu.py")
    print()


def _append_if_missing(filepath: Path, marker: str, block: str, label: str) -> None:
    if filepath.exists():
        text = filepath.read_text()
        if marker in text:
            print(f"  ✓ {label} — already configured.")
            return
        filepath.write_text(text + block)
        print(f"  ✓ {label} — appended configuration.")
    else:
        filepath.write_text(block)
        print(f"  ✓ {label} — created.")


def step_verify(venv_dir: Path) -> bool:
    """Verify every critical package is importable inside the venv."""
    print(f"\n{'='*60}")
    print("[5/5] Verification")
    print(f"{'='*60}\n")

    python_bin = _venv_executable(venv_dir, "python")
    check = (
        "import sys; "
        "mods = ['torch','torchvision','transformers','cv2','PIL','scipy','psutil','numpy','sam2','sam3','einops']; "
        "ok = True\n"
        "for m in mods:\n"
        "    try:\n"
        "        mod = __import__(m); v = getattr(mod,'__version__','?')\n"
        "        print(f'  ✓ {m} ({v})')\n"
        "    except ImportError:\n"
        "        print(f'  ✗ {m} — NOT FOUND'); ok = False\n"
        "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')\n"
        "if ok: print('OK')\n"
    )
    try:
        out = subprocess.check_output(
            [python_bin, "-c", check], text=True, stderr=subprocess.STDOUT,
        )
        print(out)
        return "OK" in out
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Verification failed:\n{e.output}")
        return False


def print_model_info() -> None:
    """Print model download instructions."""
    models_dir = PACKAGE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print("MODEL CHECKPOINTS")
    print(f"{'='*60}\n")

    sam_path = models_dir / "sam3.pt"
    if sam_path.exists():
        mb = sam_path.stat().st_size / (1024 ** 2)
        print(f"  [SAM3]    ✓ Found ({mb:.0f} MB)")
    else:
        print(f"  [SAM3]    ✗ Download from:")
        print(f"            https://huggingface.co/facebook/sam3")
        print(f"            Save as: {sam_path}")

    # Also check SAM2 checkpoints
    for name in ("sam2_hiera_large.pt", "sam2.1_hiera_large.pt"):
        p = models_dir / name
        if p.exists():
            mb = p.stat().st_size / (1024 ** 2)
            print(f"  [SAM2]    ✓ {name} ({mb:.0f} MB)")

    print(f"  [ViTMatte]  Auto-downloaded from HuggingFace on first use")
    print(f"              (hustvl/vitmatte-small-composition-1k)\n")


def save_config(env_dir: Path) -> None:
    """Persist the environment path so the gizmo finds it at startup."""
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass
    if IS_WINDOWS:
        cfg["target_path"] = str(env_dir)
    else:
        cfg["venv_path"] = str(env_dir)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def verify_plugin_files() -> bool:
    """Check that all required plugin files are present."""
    required = [
        "__init__.py", "callbacks.py", "inference.py", "temporal.py",
        "cache.py", "filters.py", "vitmatte_refiner.py",
        "env_bootstrap.py", "gizmos/H2_SamViT.gizmo",
    ]
    ok = True
    for f in required:
        p = PACKAGE_DIR / f
        status = "✓" if p.exists() else "✗ MISSING"
        if not p.exists():
            ok = False
        print(f"  {status}  {f}")
    return ok


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H2 SamViT Gizmo — Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 install.py                  # auto GPU detect\n"
            "  python3 install.py --cuda 12.6      # force CUDA 12.6\n"
            "  python3 install.py --cpu            # CPU-only\n"
            "  python3 install.py --venv ~/h2venv  # custom venv path\n"
        ),
    )
    parser.add_argument("--venv", type=str, default=None,
                        help="Custom path for the virtual environment (Linux/macOS)")
    parser.add_argument("--cuda", type=str, default=None,
                        help="CUDA version to use (e.g. 12.1)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only PyTorch (no CUDA)")
    args = parser.parse_args()

    print("=" * 60)
    print("  H2 SamViT Gizmo — Installer")
    print("  SAM3 + ViTMatte for Nuke 16")
    print("=" * 60)

    # Locate Python 3.11
    python_bin = find_python311()
    print(f"\n  Python : {python_bin}")
    print(f"  OS     : {'Windows (target-dir mode)' if IS_WINDOWS else 'Linux / macOS (venv mode)'}")

    # Determine CUDA mode
    if args.cpu:
        cuda_ver = ""
        print("  Mode   : CPU-only (--cpu flag)")
    elif args.cuda:
        cuda_ver = args.cuda
        print(f"  Mode   : CUDA {cuda_ver} (user-specified)")
    else:
        cuda_ver = detect_cuda()
        print(f"  Mode   : {'CUDA ' + cuda_ver + ' (auto-detected)' if cuda_ver else 'CPU-only (no GPU detected)'}")

    if IS_WINDOWS:
        # ── Windows: flat target-directory install ──
        target_dir = PYTHON_PACKAGES_DIR
        print(f"  Target : {target_dir}\n")

        step_install_to_target(python_bin, target_dir, cuda_ver)

        # Save config (just record that we used target-dir mode)
        save_config(target_dir)

        # Configure Nuke startup
        step_setup_nuke(PACKAGE_DIR)

        # Verify
        passed = step_verify_target(python_bin, target_dir)
    else:
        # ── Linux / macOS: standard venv install ──
        venv_dir = Path(args.venv).resolve() if args.venv else DEFAULT_VENV_DIR
        print(f"  Venv   : {venv_dir}\n")

        # Step 1 — create venv
        step_create_venv(python_bin, venv_dir)
        pip_bin = _venv_executable(venv_dir, "pip")

        # Step 2 — install PyTorch
        step_install_torch(pip_bin, cuda_ver)

        # Step 3 — install other ML packages
        step_install_packages(pip_bin)

        # Save venv location
        save_config(venv_dir)

        # Step 4 — configure Nuke startup
        step_setup_nuke(PACKAGE_DIR)

        # Step 5 — verify
        passed = step_verify(venv_dir)

    # Plugin file check
    print(f"{'='*60}")
    print("Plugin files:")
    print(f"{'='*60}")
    verify_plugin_files()

    # Model instructions
    print()
    print_model_info()

    # Summary
    print("=" * 60)
    if passed:
        print("✓ Installation successful!")
    else:
        print("⚠ Completed with warnings — check messages above.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Download the SAM3 checkpoint (if not already done)")
    print("  2. Restart Nuke")
    print("  3. Create an H2_SamViT node from the H2 menu")
    print()


if __name__ == "__main__":
    main()
