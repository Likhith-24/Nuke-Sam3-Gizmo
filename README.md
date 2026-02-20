# H2 SamViT Gizmo

AI-powered segmentation and matting gizmo for **Nuke 16** combining **SAM 2 / SAM 3** (Segment Anything Model) with **ViTMatte** for production-quality alpha mattes.

All models run **fully locally** — no cloud APIs, no internet required after initial download.

## Features

### Dual Pipeline Modes

| Mode | How it works |
|------|-------------|
| **Point / Bbox** | Place foreground & background points and/or draw a bounding box interactively in the Viewer |
| **Text Prompt** | Describe an object in plain text — powered by Grounding DINO (cached locally after first download) |

### Interactive Viewer Controls

| Shortcut | Action |
|----------|--------|
| **Ctrl + Left Click** | Add foreground (positive) point |
| **Ctrl + Right Click** | Add background (negative) point |
| **Shift + Left Drag** | Draw bounding box |

- Points and bbox are drawn directly on the Viewer as **coloured overlays**
- Foreground points: green circle with **+**, Background points: red circle with **–**
- Bounding box: cyan dashed rectangle with corner handles
- Overlay colours are fully customisable via **FG Point Color**, **BG Point Color**, and **Bbox Color** knobs
- Shortcuts are only active when an H2 SamViT node is selected and *Enable Edit* is on

### Model Selection
- **SAM 2.0** — Tiny · Small · Base+ · Large
- **SAM 2.1** — Tiny · Small · Base+ · Large
- **SAM 3** — (when checkpoint is available)
- **Precision** — fp16 · bf16 · fp32
- Models are **auto-downloaded** on first use (or via the *Download Model* button)
- All models and weights are cached under `./models/` for **offline operation**

### Format Compatibility
- Supports **all** image and video formats that Nuke can read:
  EXR · TIFF · PNG · JPEG · DPX · MOV · MP4 · MXF and more
- Internally renders the current frame via Nuke's own pipeline, so any format Nuke decodes will work

### Mask Refinement (ViTMatte)
- Automatic trimap generation with independent **Erode** (0–200) and **Dilate** (0–200) radii
- **Show Trimap Overlay** with adjustable opacity for visual debugging
- High-quality soft alpha edges via [ViTMatte](https://huggingface.co/hustvl/vitmatte-small-composition-1k)

### Pre-Processing
- Binary threshold, **Black Point / White Point** levels
- Hole filling with configurable area
- Crop padding for bbox context

### Output
- **Output Alpha Mode**: Straight or Premultiplied
- Display modes: Overlay (coloured) or Matte (B/W)
- Shrink/Grow, Feather, Offset, Final Binary Sharp

### Temporal Consistency
- IoU-based mask selection to prevent frame-to-frame jumping
- Automatic resolution matching — cached masks from different resolutions are handled gracefully
- Optional temporal smoothing with edge-only mode

---

## Requirements

| Requirement | Details |
|-------------|---------|
| Nuke | **16.0** or later (Python 3.11 embedded) |
| GPU | CUDA-capable, 8 GB+ VRAM recommended |
| Disk | ~150 MB – 900 MB per model checkpoint |

---

## Installation

### Linux / macOS — venv mode

```bash
cd /path/to/Nuke16/plugins/H2_SamViT_Gizmo

# Auto-detect GPU, create ./venv, install packages
python3 install.py

# Or force a specific CUDA version / CPU-only
python3 install.py --cuda 12.6
python3 install.py --cpu
```

The installer:
1. Creates a Python 3.11 virtual environment in `./venv`
2. Installs PyTorch (with CUDA), Transformers, OpenCV, etc.
3. Configures `~/.nuke/init.py` and `menu.py` to load the gizmo on startup

### Windows — target-directory mode

On Windows, a flat `pip install --target` approach is used instead of a
venv, because Nuke's embedded Python on Windows does not support venvs
reliably.

```cmd
cd C:\path\to\Nuke16\plugins\H2_SamViT_Gizmo

python install.py              &:: auto-detect GPU
python install.py --cuda 12.6  &:: force CUDA version
python install.py --cpu        &:: CPU-only
```

Packages are installed into `./python_packages/`.

### How the bootstrap works

You do **not** need to manually activate the venv or run any script
before launching Nuke. The bootstrap is fully automatic:

1. Nuke loads `__init__.py` at startup (via `~/.nuke/init.py`)
2. `__init__.py` calls `env_bootstrap.bootstrap()`
3. `bootstrap()` finds `./venv/lib/python3.11/site-packages/` (Linux/macOS)
   or `./python_packages/` (Windows) and injects it into `sys.path`
4. All ML packages (`torch`, `transformers`, `cv2`, etc.) become importable
   inside Nuke's Python — no shell activation required

If packages are missing, a dialog appears in Nuke with instructions.

---

## Model Checkpoints

Models are stored in `./models/` and downloaded automatically the first
time you run inference (or via the **Download Model** button in the gizmo).

| Family | Version | Size | File | ~MB |
|--------|---------|------|------|----:|
| SAM 2 | 2.0 | Tiny | `sam2_hiera_tiny.pt` | 156 |
| SAM 2 | 2.0 | Small | `sam2_hiera_small.pt` | 185 |
| SAM 2 | 2.0 | Base+ | `sam2_hiera_base_plus.pt` | 324 |
| SAM 2 | 2.0 | Large | `sam2_hiera_large.pt` | 898 |
| SAM 2 | 2.1 | Tiny | `sam2.1_hiera_tiny.pt` | 156 |
| SAM 2 | 2.1 | Small | `sam2.1_hiera_small.pt` | 185 |
| SAM 2 | 2.1 | Base+ | `sam2.1_hiera_base_plus.pt` | 324 |
| SAM 2 | 2.1 | Large | `sam2.1_hiera_large.pt` | 898 |
| SAM 3 | — | — | `sam3.pt` | — |

ViTMatte (`hustvl/vitmatte-small-composition-1k`) is downloaded
automatically from HuggingFace on first use.

---

## Usage

### Basic Workflow

1. Connect an image to the **H2 SamViT** node's input
2. Choose a **Model** (e.g. SAM 2.1 Large fp16)
3. Select **Pipeline Mode** — *Point / Bbox* or *Text Prompt*
4. Add prompts:
   - **Point / Bbox**: use interactive Viewer shortcuts:
     - **Ctrl + Left Click** — add foreground point
     - **Ctrl + Right Click** — add background point
     - **Shift + Left Drag** — draw bounding box
   - **Text Prompt**: type a description (e.g. `person in red jacket`)
5. Click **Run Inference**
6. Fine-tune with Pre-Processing, Trimap, and Output controls
7. Use the alpha channel downstream for compositing

### Tool Behaviour

- Interactive shortcuts only work when the H2 SamViT node is **selected**
  and **Enable Edit** is turned on
- Points and bbox are shown as coloured overlays directly in the Viewer
- Customise overlay colours with the **FG Point Color**, **BG Point Color**,
  and **Bbox Color** knobs in the UI
- *Add FG Point* / *Add BG Point* / *Draw Box* toggle buttons also
  work — they are mutually exclusive (ONYX-style)
- Expand the **Points List** group to see all active handles

---

## Parameters Reference

### Model
| Parameter | Default | Description |
|-----------|---------|-------------|
| Model Family | SAM2 | SAM2 or SAM3 |
| SAM Version | 2.1 | 2.0 or 2.1 (SAM2 only) |
| Model Size | Large | Tiny · Small · Base+ · Large |
| Model Precision | fp16 | fp16 · bf16 · fp32 |
| Download Model | — | Download the selected checkpoint |

### Tools
| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable Edit | On | Master toggle for AI segmentation |
| Draw Box | Off | Click-drag to draw bounding box |
| Add FG Point | Off | Place a foreground (include) point |
| Add BG Point | Off | Place a background (exclude) point |
| Clear All Points | — | Remove all points |
| Clear Box | — | Clear the bounding box |
| FG Point Color | Green | Overlay colour for foreground points |
| BG Point Color | Red | Overlay colour for background points |
| Bbox Color | Cyan | Overlay colour for bounding box |

### Pre-Processing
| Parameter | Default | Description |
|-----------|---------|-------------|
| Binary Threshold | 100 | Threshold for coarse mask |
| Black Point | 0.0 | Levels — lift shadows |
| White Point | 1.0 | Levels — clip highlights |
| Fill Holes | On | Fill small holes in the mask |
| Fill Holes Area | 16 | Max hole size in pixels |
| Crop Padding % | 20 | Extra padding around bbox |

### VitMatte Trimap
| Parameter | Default | Description |
|-----------|---------|-------------|
| Show Trimap Overlay | Off | Display trimap visualisation |
| Trimap Overlay Opacity | 0.6 | Overlay opacity |
| Trimap Erode Radius | 3 | Erode definite-foreground (0–200) |
| Trimap Dilate Radius | 10 | Dilate unknown region (0–200) |

### Output
| Parameter | Default | Description |
|-----------|---------|-------------|
| Output Alpha Mode | Straight | Straight or Premultiplied |
| Display Mode | Overlay | Overlay or Matte |
| Show Mask Overlay | On | Overlay visibility |
| Overlay Color | red 50% | Colour & opacity of overlay |
| Mask Shrink/Grow | 0 | Expand (+) or contract (-) mask |
| Edge Feather | 0 | Blur mask edges |
| Offset Mask X / Y | 0 | Translate mask position |
| Final Binary Sharp | Off | Threshold to hard 0/1 mask |

### Temporal Mask Consistency
| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable Temporal Consistency | On | Prevent mask jumping |
| Temporal Weight % | 50 | IoU weight for mask selection |
| Suppression Threshold % | 30 | Jumping-region removal |

### Temporal Smoothing
| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable Temporal | Off | Frame-to-frame smoothing |
| Smoothing | 50 | Blend strength |
| Smooth Edges Only | Off | Only smooth edge regions |
| Edge Width | 8 | Width of edge band |

### Cache
| Parameter | Default | Description |
|-----------|---------|-------------|
| Cache Memory % | 25 | RAM percentage for mask cache |

---

## File Structure

```
H2_SamViT_Gizmo/
├── __init__.py          # Nuke package entry — calls bootstrap + register
├── env_bootstrap.py     # Injects venv/python_packages into sys.path
├── install.py           # One-command installer (venv on Linux, target-dir on Windows)
├── callbacks.py         # Knob-changed handlers, point/bbox management
├── inference.py         # SAM + ViTMatte inference engine
├── model_manager.py     # Model registry, download, and builder
├── vitmatte_refiner.py  # ViTMatte alpha refinement
├── ui_overlay.py        # Qt viewer overlay — interactive points/bbox + coloured drawing
├── cache.py             # LRU mask cache
├── temporal.py          # Temporal consistency / smoothing
├── filters.py           # Image processing filters
├── nuke_integration.py  # Nuke helper utilities
├── menu.py              # Adds gizmo to Nuke's Nodes menu
├── requirements.txt     # pip requirements
├── gizmos/
│   └── H2_SamViT.gizmo # Nuke gizmo definition
├── models/              # SAM checkpoints (auto-downloaded)
│   └── ...
└── venv/                # Virtual environment (Linux/macOS, created by install.py)
    └── ...
```

---

## Troubleshooting

### "packages not installed" dialog on startup
Run the installer from a terminal — **not** from inside Nuke:
```bash
cd /path/to/H2_SamViT_Gizmo
python3 install.py        # Linux / macOS
python install.py         # Windows
```

### "SAM checkpoint not found"
Click **Download Model** in the gizmo, or models will auto-download on
first inference. Ensure you have internet access.

### "CUDA out of memory"
- Switch to a smaller model (e.g. Tiny or Small)
- Use fp16 precision
- Lower Cache Memory %
- Close other GPU applications

### Mask quality issues
- Increase Trimap Dilate Radius for softer edges
- Add more foreground / background points
- Try adjusting Black Point / White Point

---

## Credits

- **SAM 2 / SAM 3** — Meta AI Research ([segment-anything-2](https://github.com/facebookresearch/segment-anything-2))
- **ViTMatte** — HUST Vision Lab ([vitmatte](https://huggingface.co/hustvl/vitmatte-small-composition-1k))
- **Grounding DINO** — IDEA Research ([grounding-dino](https://github.com/IDEA-Research/GroundingDINO))

## License

This gizmo is provided for educational and production use. SAM, ViTMatte,
and Grounding DINO have their own licences — please review them for
commercial use.

---

*H2 SamViT v1.1 — H2 Studios 2026*
