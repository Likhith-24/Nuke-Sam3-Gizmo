# H2 SamViT Gizmo

AI-powered segmentation and matting gizmo for **Nuke 16** combining **SAM 2 / SAM 3** (Segment Anything Model) with **ViTMatte** for production-quality alpha mattes.

All models run **fully locally** — no cloud APIs, no internet required after initial download.

## Features

### Dual Pipeline Modes

| Mode | How it works |
| -- | -- |
| **Point / Bbox** | Place foreground & background points and/or draw a bounding box interactively in the Viewer |
| **Text Prompt** | Describe an object in plain text — powered by Grounding DINO (cached locally after first download) |

### Interactive Viewer Controls (mode-based)

Prompt placement is **mode-based** — no modifier keys:

1. Click a placement button on the node (**Place FG Points**, **Place BG
   Points**, **Draw Bbox**, …) to enter that mode.
2. **Plain left-click in the Viewer** to place points / bbox corners.
   Right-click, middle-click, scroll and all modifier-clicks pass straight
   through to Nuke (context menu, pan, zoom… nothing is hijacked).
3. Click the same button again to exit the mode.

- **Fit the Viewer first (press `F`)** — click→pixel mapping assumes the
  frame is fitted; placing points while zoomed/panned lands them offset.
- Points and bbox are drawn directly on the Viewer as **coloured overlays**
- Foreground points: green circle with **+**, Background points: red circle with **–**
- Overlay colours are fully customisable via **FG Point Color**, **BG Point Color**, and **Bbox Color** knobs
- *Enable Edit* must be on; the node does not need to stay selected while clicking

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
| -- | -- |
| Nuke | **16.0** or later (Python 3.11 embedded) |
| GPU | CUDA-capable, 8 GB+ VRAM recommended |
| Disk | ~150 MB – 900 MB per model checkpoint |

---

## Installation

**Shared installs need no per-machine setup**: the `./venv` next to this
plugin is the default environment for every machine that loads it.
Cold worker start from the share costs ~15 s (torch import); after that the
model stays warm.

**Optional — faster cold starts on one machine** (~3 s): install a local
venv. Run from a **normal terminal — not inside Nuke**:

```powershell
cd <path-to>\plugins\H2_SamViT_Gizmo
py -3.11 install.py --venv C:\h2_samvit\venv     # local venv for THIS machine
py -3.11 install.py --frozen                     # rebuild the shared ./venv exactly from uv.lock
py -3.11 install.py --recreate                   # delete + rebuild from scratch
```

`install.py` is driven by [`uv`](https://docs.astral.sh/uv/):

1. Bootstraps `uv` (via `pip`) if it isn't already on `PATH`.
2. Runs `uv sync` — creates a Python 3.11 venv (downloading a matching
   interpreter if needed) and installs the exact set pinned in
   `pyproject.toml` / `uv.lock`: PyTorch (CUDA 12.6), Transformers, OpenCV,
   SAM2, SAM3, Triton, etc.
3. Records the venv path in `env_config.json` (shared file; entries only
   apply on machines where the recorded path exists — other machines fall
   back to the shared `./venv`).

**CUDA** is pinned to the 12.6 wheels in `pyproject.toml`; to target another
build, change the `pytorch-cu126` index there and re-run `uv lock`. Keep
`uv.lock` committed so every machine reproduces the same environment.

### Architecture: inference runs OUTSIDE Nuke (do not "simplify" this)

Two hard constraints shape how this plugin runs models — both were
diagnosed the hard way in July 2026:

1. **Nuke bundles its own libtorch** (`c10.dll`, `torch_cpu.dll` — for the
   Inference/CopyCat nodes), so pip-torch can never be imported inside
   Nuke's process.
2. **Endpoint-security (EDR) tools can block torch's `c10.dll` in any child process
   of Nuke.exe** (`WinError 1114`) — regardless of environment, PATH, or
   DLL location. The same venv loads torch fine from PowerShell or from a
   process outside Nuke's tree.

Therefore all torch work happens in `worker.py`, spawned **detached from
Nuke's process tree via WMI** (its parent is `WmiPrvSE.exe`) and reached
over a **token-protected localhost socket** advertised in a rendezvous
file (`%TEMP%\h2_samvit_rdv_<pid>.json`; worker log: same path + `.log`).
The worker keeps the model warm on the GPU between clicks and
**self-destructs after 30 min idle**. Nuke's process only imports the
light packages (numpy, cv2, PIL) from the venv — appended to `sys.path`
so Nuke's own bundled packages keep priority for other plugins.

---

## Model Checkpoints

Models are stored in `./models/` (on the share — downloaded once, used by
every machine). **Run Inference / Process Sequence are gated**: they refuse
with a clear message until the selected model's checkpoint is downloaded —
use the **Download Model** button on the node first.

| Family | Version | Size | File | ~MB |
| -- | -- | -- | -- | ---: |
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

1. **Save your Nuke script first** — computed masks are written to
   `h2_samvit_masks/<node>/mask.####.png` **next to the saved script** so
   they persist and travel with the comp (unsaved scripts fall back to
   `%TEMP%`, which gets cleaned).
2. Connect an image to the **H2 SamViT** node's input
3. Choose a **Model** (e.g. SAM 2.1 Large) and click **Download Model** if
   the Status line says it isn't downloaded yet
4. Select **Pipeline Mode** — *Point / Bbox* or *Text Prompt*
5. Add prompts:
   - **Point / Bbox**: press **`F`** over the Viewer to fit the frame, click
     **Place FG Points**, left-click the subject in the Viewer (repeat for
     BG points / bbox), click the button again to exit the mode
   - **Text Prompt**: type a description (e.g. `person in red jacket`)
6. Click **Run Inference** (current frame) or **Process Sequence** (whole
   input range, same prompts every frame)
7. The result lands in the **alpha channel** — press `A` over the Viewer to
   see the matte, or set **Display Mode** to *Matte*
8. Fine-tune with Pre-Processing, Trimap, and Output controls

### Tool Behaviour

- **Enable Edit** must be on; placement modes receive clicks even if the
  node loses selection
- The first inference of a session starts the worker (~15 s shared venv /
  ~3 s local) and loads the model once; later clicks reuse the warm model
- **Free VRAM** releases the model from GPU memory (worker stays alive;
  next inference reloads the model only)
- Cancelling the progress bar kills the worker; cancelling during
  **Process Sequence** stops the whole run
- Expand the **Points List** group to see all active handles

---

## Parameters Reference

### Model

| Parameter | Default | Description |
| -- | -- | -- |
| Model Family | SAM2 | SAM2 or SAM3 |
| SAM Version | 2.1 | 2.0 or 2.1 (SAM2 only) |
| Model Size | Large | Tiny · Small · Base+ · Large |
| Model Precision | fp16 | fp16 · bf16 · fp32 |
| Download Model | — | Download the selected checkpoint |

### Tools

| Parameter | Default | Description |
| -- | -- | -- |
| Enable Edit | On | Master toggle for AI segmentation |
| Place FG Points | — | Toggle mode: left-click Viewer to add include points |
| Place BG Points | — | Toggle mode: left-click Viewer to add exclude points |
| Draw Bbox | — | Toggle mode: click two corners in the Viewer |
| Delete Point | — | Toggle mode: click near a point to remove it |
| Clear All Points | — | Remove all points |
| Clear Box | — | Clear the bounding box |
| Free VRAM | — | Unload the model from GPU (reloads on next inference) |
| FG Point Color | Green | Overlay colour for foreground points |
| BG Point Color | Red | Overlay colour for background points |
| Bbox Color | Cyan | Overlay colour for bounding box |

### Pre-Processing

| Parameter | Default | Description |
| -- | -- | -- |
| Binary Threshold | 100 | Threshold for coarse mask |
| Black Point | 0.0 | Levels — lift shadows |
| White Point | 1.0 | Levels — clip highlights |
| Fill Holes | On | Fill small holes in the mask |
| Fill Holes Area | 16 | Max hole size in pixels |
| Crop Padding % | 20 | Extra padding around bbox |

### VitMatte Trimap

| Parameter | Default | Description |
| -- | -- | -- |
| Show Trimap Overlay | Off | Display trimap visualisation |
| Trimap Overlay Opacity | 0.6 | Overlay opacity |
| Trimap Erode Radius | 3 | Erode definite-foreground (0–200) |
| Trimap Dilate Radius | 10 | Dilate unknown region (0–200) |

### Output

| Parameter | Default | Description |
| -- | -- | -- |
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
| -- | -- | -- |
| Enable Temporal Consistency | On | Prevent mask jumping |
| Temporal Weight % | 50 | IoU weight for mask selection |
| Suppression Threshold % | 30 | Jumping-region removal |

### Temporal Smoothing

| Parameter | Default | Description |
| -- | -- | -- |
| Enable Temporal | Off | Frame-to-frame smoothing |
| Smoothing | 50 | Blend strength |
| Smooth Edges Only | Off | Only smooth edge regions |
| Edge Width | 8 | Width of edge band |

### Cache

| Parameter | Default | Description |
| -- | -- | -- |
| Cache Memory % | 25 | RAM percentage for mask cache |

---

## File Structure

```text
H2_SamViT_Gizmo/
├── init.py              # Nuke startup — registers ./gizmos, makes the package importable
├── __init__.py          # Python package entry — calls bootstrap + register
├── env_bootstrap.py     # Appends ./venv site-packages to Nuke's sys.path (light pkgs only)
├── install.py           # uv-based installer (--venv for a machine-local env)
├── worker.py            # THE ONLY place torch is imported — detached inference worker
├── worker_client.py     # Nuke side: WMI-detached spawn + localhost socket protocol
├── callbacks.py         # Buttons, knob-changed handlers, point/bbox management
├── inference.py         # Nuke-side broker: render frame → worker → apply mask
├── model_manager.py     # Model registry, download, and builder
├── viewer_events.py     # Qt event filter — mode-based click-to-place in the Viewer
├── viewer_overlay.py    # Qt overlay painting
├── ui_overlay.py        # PIL overlay — draws point/bbox guides into the gizmo
├── matanyone2_refiner.py# MatAnyone2 alpha refinement
├── cache.py             # LRU mask cache
├── temporal.py          # Temporal consistency / smoothing
├── filters.py           # Image processing filters
├── menu.py              # Nodes menu entry + installs the viewer click handler
├── pyproject.toml       # uv dependency set (torch/CUDA pin, SAM source builds)
├── uv.lock              # locked, reproducible dependency versions
├── env_config.json      # per-machine venv locations (paths only apply where they exist)
├── gizmos/
│   └── H2_SamViT.gizmo  # THE gizmo (v2/v3 variants are broken → gizmos_disabled/)
├── gizmos_disabled/     # quarantined upstream gizmo variants — do not re-enable as-is
├── models/              # SAM checkpoints (shared; downloaded via the node's button)
│   └── ...
└── venv/                # shared uv-managed ML environment (default for all machines)
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

### Inference fails — where to look

The worker's log is the first stop:
`%TEMP%\h2_samvit_rdv_<nuke-pid>.json.log` (error dialogs quote its tail).
It records torch import, model load, and per-request tracebacks.

### `WinError 1114` / "Error loading c10.dll"

Two known causes, both already engineered around — if you see this again,
something bypassed the architecture:

1. Torch was imported **inside Nuke's process** (Nuke bundles its own
   libtorch — never `import torch` in Nuke-side code).
2. The worker ran as a **child of Nuke** instead of detached — endpoint
   security tooling blocks `c10.dll` in any Nuke.exe descendant. The worker
   must be spawned via the WMI path in `worker_client.py` (parent:
   `WmiPrvSE.exe`). Verify torch itself is fine from PowerShell:
   `& <venv>\Scripts\python.exe -c "import torch; print(torch.__version__)"`

### "checkpoint is not downloaded yet"

Run Inference / Process Sequence are gated on the selected model's
checkpoint being on disk. Click **Download Model** on the node (internet
required once; checkpoints land in `./models/` on the share for everyone).

### Points don't land where I click

The Viewer must be **fitted** — press `F` over the Viewer before placing
points. Zoomed/panned placement lands offset (known limitation).

### Masks disappeared after reopening a script

Masks are written next to the **saved** script (`h2_samvit_masks/`). If the
script was unsaved during inference they went to `%TEMP%` and may have been
cleaned — save the script and re-run inference.

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
