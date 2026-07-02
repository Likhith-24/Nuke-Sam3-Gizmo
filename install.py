# install.py — H2_SamViT_Gizmo ML environment installer (uv-based).
#
# Provisions the dependency set declared in pyproject.toml into ./venv using
# uv. env_bootstrap.py injects that venv's site-packages into Nuke's sys.path
# at startup.
#
# USAGE — run from a normal terminal (NOT inside Nuke):
#
#   python install.py               # create/update ./venv (locks if needed)
#   python install.py --recreate    # delete ./venv and rebuild from scratch
#   python install.py --frozen      # require an up-to-date uv.lock (rollout/CI)
#
# uv is bootstrapped automatically (via pip) if it isn't already on PATH.
# The CUDA build and all versions are pinned in pyproject.toml / uv.lock.

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
VENV_DIR = PACKAGE_DIR / "venv"
CONFIG_FILE = PACKAGE_DIR / "env_config.json"
REQUIRED_PYTHON = "3.11"  # must match Nuke 16's embedded Python

REQUIRED_MODULES = [
    "torch", "torchvision", "transformers", "cv2", "PIL",
    "scipy", "psutil", "numpy", "sam2", "sam3", "einops",
]


def _run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_uv() -> str:
    """Return a path to the uv executable, bootstrapping it via pip if needed."""
    uv = shutil.which("uv")
    if uv:
        return uv
    print("uv not found — bootstrapping with pip ...")
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "uv"])
    # The uv wheel bundles the binary and exposes find_uv_bin().
    try:
        import uv as uv_module

        return uv_module.find_uv_bin()
    except Exception:
        uv = shutil.which("uv")
        if uv:
            return uv
    raise RuntimeError(
        "Failed to bootstrap uv. Install it manually: https://docs.astral.sh/uv/"
    )


def save_config(venv_dir: Path) -> None:
    """Record the venv location so env_bootstrap.py finds it at startup."""
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    cfg["venv_path"] = str(venv_dir)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def verify(venv_dir: Path) -> bool:
    """Import every required module inside the freshly built venv."""
    py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    check = "\n".join(
        [
            "import importlib",
            f"mods = {REQUIRED_MODULES!r}",
            "ok = True",
            "for m in mods:",
            "    try:",
            "        mod = importlib.import_module(m)",
            "        print('  ok', m, getattr(mod, '__version__', '?'))",
            "    except Exception as e:",
            "        print('  MISSING', m, '->', e); ok = False",
            "try:",
            "    import torch; print('  CUDA available:', torch.cuda.is_available())",
            "except Exception as e:",
            "    print('  torch load error:', e)",
            "print('VERIFY_OK' if ok else 'VERIFY_FAIL')",
        ]
    )
    out = subprocess.run([str(py), "-c", check], text=True, capture_output=True)
    print(out.stdout, end="")
    if out.stderr.strip():
        print(out.stderr, end="")
    return "VERIFY_OK" in out.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 SamViT Gizmo installer (uv)")
    parser.add_argument(
        "--recreate", action="store_true", help="Delete ./venv and rebuild"
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Require an up-to-date uv.lock (reproducible rollout)",
    )
    parser.add_argument(
        "--venv",
        type=str,
        default=None,
        help=(
            "Custom venv location (e.g. C:\\h2_samvit\\venv). Useful when the "
            "plugin lives on a network share: a machine-local venv imports "
            "torch much faster than over SMB. The location is recorded in "
            "env_config.json (per machine — rerun install.py on each "
            "workstation that wants a local venv)."
        ),
    )
    args = parser.parse_args()

    venv_dir = Path(args.venv).expanduser().resolve() if args.venv else VENV_DIR

    print("=" * 60)
    print("  H2 SamViT Gizmo — installer (uv)")
    print("=" * 60)

    uv = ensure_uv()
    print(f"  uv     : {uv}")
    print(f"  venv   : {venv_dir}")

    if args.recreate and venv_dir.exists():
        print("  Removing existing venv ...")
        shutil.rmtree(venv_dir)

    # `uv sync` creates the venv (downloading Python 3.11 if missing), resolves
    # + locks the dependency set, and installs it — building sam2/sam3 without
    # isolation against the torch it installs first.
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)  # place venv where env_bootstrap looks

    sync_cmd = [uv, "sync", "--python", REQUIRED_PYTHON]
    if args.frozen:
        sync_cmd.append("--frozen")
    _run(sync_cmd, cwd=str(PACKAGE_DIR), env=env)

    save_config(venv_dir)

    print("\n" + "=" * 60)
    print("  Verifying")
    print("=" * 60)
    passed = verify(venv_dir)

    print("\n" + "=" * 60)
    print("  Installed successfully" if passed else "  Completed with warnings — see above")
    print("=" * 60)
    print("\nNext:")
    print("  1. Commit the generated uv.lock for reproducible installs.")
    print("  2. Download the model checkpoints (see README).")
    print("  3. Restart Nuke and create an H2_SamViT node.\n")


if __name__ == "__main__":
    main()
