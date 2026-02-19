# install.py - Installation script for H2 SamViT Gizmo
# Run this script to set up the gizmo in Nuke

import os
import sys
import shutil
import subprocess
from pathlib import Path


def get_nuke_paths():
    """Find Nuke installation and user directory."""
    # Common Nuke user directories
    home = Path.home()
    
    nuke_dirs = []
    
    if sys.platform == "win32":
        # Windows
        nuke_dirs = list(home.glob(".nuke*"))
        if not nuke_dirs:
            nuke_dirs = [home / ".nuke"]
    elif sys.platform == "darwin":
        # macOS
        nuke_dirs = list(home.glob(".nuke*"))
        if not nuke_dirs:
            nuke_dirs = [home / ".nuke"]
    else:
        # Linux
        nuke_dirs = list(home.glob(".nuke*"))
        if not nuke_dirs:
            nuke_dirs = [home / ".nuke"]
    
    return nuke_dirs


def install_python_dependencies():
    """Install required Python packages."""
    print("Installing Python dependencies...")
    
    requirements = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements)
        ])
        print("✓ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def download_models():
    """Download required model checkpoints."""
    print("\nDownloading model checkpoints...")
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\n[SAM3 Model]")
    print("Please download SAM3 checkpoint from:")
    print("  https://github.com/facebookresearch/segment-anything-2")
    print(f"\nSave 'sam2_hiera_large.pt' to:")
    print(f"  {models_dir / 'sam2_hiera_large.pt'}")
    
    print("\n[ViTMatte Model]")
    print("The ViTMatte model will be downloaded automatically from HuggingFace")
    print("on first use (hustvl/vitmatte-small-composition-1k)")
    
    return True


def add_to_nuke_init():
    """Add plugin to Nuke's init.py."""
    nuke_dirs = get_nuke_paths()
    
    if not nuke_dirs:
        print("✗ Could not find Nuke user directory")
        return False
    
    nuke_dir = nuke_dirs[0]
    nuke_dir.mkdir(exist_ok=True)
    
    init_file = nuke_dir / "init.py"
    menu_file = nuke_dir / "menu.py"
    
    plugin_path = Path(__file__).parent.absolute()
    
    # Code to add to init.py
    init_code = f'''
# H2 SamViT Gizmo - SAM3 + ViTMatte
import sys
sys.path.insert(0, r"{plugin_path.parent}")
import nuke
nuke.pluginAddPath(r"{plugin_path / 'gizmos'}")
'''
    
    # Code to add to menu.py
    menu_code = f'''
# H2 SamViT Gizmo - Menu Registration
from H2_SamViT_Gizmo import menu
'''
    
    # Update init.py
    print(f"\nUpdating {init_file}...")
    if init_file.exists():
        existing = init_file.read_text()
        if "H2_SamViT_Gizmo" not in existing:
            init_file.write_text(existing + init_code)
            print("✓ Added to existing init.py")
        else:
            print("✓ Already present in init.py")
    else:
        init_file.write_text(init_code)
        print("✓ Created new init.py")
    
    # Update menu.py
    print(f"Updating {menu_file}...")
    if menu_file.exists():
        existing = menu_file.read_text()
        if "H2_SamViT_Gizmo" not in existing:
            menu_file.write_text(existing + menu_code)
            print("✓ Added to existing menu.py")
        else:
            print("✓ Already present in menu.py")
    else:
        menu_file.write_text(menu_code)
        print("✓ Created new menu.py")
    
    return True


def verify_installation():
    """Verify the installation is complete."""
    print("\n" + "=" * 50)
    print("Verifying installation...")
    print("=" * 50)
    
    plugin_path = Path(__file__).parent
    
    required_files = [
        "__init__.py",
        "callbacks.py",
        "inference.py",
        "temporal.py",
        "cache.py",
        "filters.py",
        "vitmatte_refiner.py",
        "gizmos/H2_SamViT.gizmo",
    ]
    
    all_present = True
    for f in required_files:
        path = plugin_path / f
        if path.exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} - MISSING")
            all_present = False
    
    # Check model
    model_path = plugin_path / "models" / "sam2_hiera_large.pt"
    if model_path.exists():
        print(f"  ✓ models/sam2_hiera_large.pt")
    else:
        print(f"  ⚠ models/sam2_hiera_large.pt - Download required")
    
    return all_present


def main():
    print("=" * 50)
    print("H2 SamViT Gizmo - Installation Script")
    print("SAM3 + ViTMatte for Nuke")
    print("=" * 50)
    print()
    
    # Step 1: Install Python dependencies
    if not install_python_dependencies():
        print("\n⚠ Warning: Some dependencies may be missing")
    
    # Step 2: Download models
    download_models()
    
    # Step 3: Add to Nuke
    if not add_to_nuke_init():
        print("\n⚠ Warning: Could not update Nuke configuration")
        print("You may need to manually add the plugin to init.py")
    
    # Step 4: Verify
    verify_installation()
    
    print("\n" + "=" * 50)
    print("Installation complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Download SAM3 checkpoint (see instructions above)")
    print("2. Restart Nuke")
    print("3. Find 'H2 SamViT' in the H2 menu or Image menu")
    print()


if __name__ == "__main__":
    main()
