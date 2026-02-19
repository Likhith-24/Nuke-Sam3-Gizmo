# H2 SamViT Gizmo

AI-powered segmentation and matting gizmo for Nuke combining **SAM3** (Segment Anything Model 3) with **ViTMatte** for high-quality alpha mattes.

## Features

### 1. Point / Bbox Mode
- **Draw Box**: Click and drag to create a bounding box around objects
- **Add FG Point**: Add green foreground points to include areas
- **Add BG Point**: Add red background points to exclude areas
- Supports up to **32 points** with animatable enabled states

### 2. Text Prompt Mode
- Describe objects in plain text (e.g., "person", "car", "red dress")
- Automatic detection using Grounding DINO
- Place a point to select which detected instance to use

### 3. Mask Refinement
- **ViTMatte refinement** for high-quality alpha edges
- **Pre-processing**: Binary threshold, hole filling
- **Trimap controls**: Refiner trimap width, erode/dilate radius
- **Temporal consistency**: Prevent mask jumping between frames
- **Post-processing**: Shrink/grow, feather, offset

### 4. Output Options
- **Overlay mode**: Colored mask preview on source
- **Matte mode**: Black/white mask view
- Alpha channel output for compositing

## Installation

### Requirements
- Nuke 13+ (tested on NukeX 14/15)
- Python 3.9+
- CUDA-capable GPU (recommended, 8GB+ VRAM)

### Quick Install

```bash
# 1. Clone or download to your Nuke plugins folder
cd /path/to/your/nuke/plugins

# 2. Install dependencies
pip install -r H2_SamViT_Gizmo/requirements.txt

# 3. Run the installer
python H2_SamViT_Gizmo/install.py
```

### Manual Install

1. Install Python dependencies:
```bash
pip install torch torchvision transformers opencv-python numpy psutil
```

2. Download SAM3 checkpoint from [Meta's repository](https://github.com/facebookresearch/segment-anything-2)
   - Save `sam2_hiera_large.pt` to `H2_SamViT_Gizmo/models/`

3. Add to your `~/.nuke/init.py`:
```python
import sys
sys.path.insert(0, "/path/to/your/plugins")
import nuke
nuke.pluginAddPath("/path/to/H2_SamViT_Gizmo/gizmos")
```

4. Add to your `~/.nuke/menu.py`:
```python
from H2_SamViT_Gizmo import menu
```

## Usage

### Basic Workflow

1. Connect an image to the H2 SamViT node input
2. Enable "Draw Box" and draw a rectangle around your object
3. Optionally add foreground (green) or background (red) points
4. Click "Run Inference" or adjust parameters to trigger computation
5. Use the alpha output for compositing

### Parameters Reference

#### Tools Panel
| Parameter | Description |
|-----------|-------------|
| Enable Edit | Master toggle for AI segmentation |
| Draw Box | Click-drag to draw bounding box |
| Add FG Point | Add foreground (include) points |
| Add BG Point | Add background (exclude) points |
| Clear All Points | Remove all points |
| Clear Box | Clear the bounding box |

#### Pre Processing
| Parameter | Default | Description |
|-----------|---------|-------------|
| Binary Threshold | 90 | Binary threshold for coarse mask |
| Fill Holes | On | Enable hole filling |
| Fill Holes Area | 16 | Max hole size to fill (pixels) |
| Crop Padding % | 20 | Extra padding around bbox |

#### Mask Post-Processing
| Parameter | Default | Description |
|-----------|---------|-------------|
| Display Mode | Overlay | Matte or Overlay view |
| Mask Shrink/Grow | 0 | Expand (+) or contract (-) mask |
| Edge Feather | 0 | Blur mask edges |
| Offset Mask X/Y | 0 | Translate mask position |
| Refiner Trimap Width | 10 | Unknown region width for ViTMatte |
| Trimap Erode Radius | 2 | Morphological erode cleanup |
| Trimap Dilate Radius | 2 | Morphological dilate cleanup |
| Final Binary Sharp | Off | Convert to hard binary mask |

#### Temporal Mask Consistency
| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable Temporal Consistency | On | Prevent mask jumping between frames |
| Temporal Weight (%) | 50 | IoU weight for mask selection |
| Suppression Threshold (%) | 30 | Threshold for jumping region removal |

#### Temporal Smoothing Motion
| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable Temporal | Off | Enable frame-to-frame smoothing |
| Smoothing | 50 | Blend strength across frames |
| Smooth edges only | Off | Only smooth edge regions |
| Edge width | 8 | Width of edge region |

#### Cache Memory
| Parameter | Default | Description |
|-----------|---------|-------------|
| Cache Memory % | 25 | RAM percentage for mask cache |
| Cache Limit | 17.9 GB | Computed cache limit display |

## Troubleshooting

### "SAM3 checkpoint not found"
Download the checkpoint from Meta's SAM2 repository and place it in the `models/` folder.

### "CUDA out of memory"
- Reduce resolution or work on a crop
- Lower Cache Memory %
- Close other GPU applications

### Mask quality issues
- Increase Refiner Trimap Width for softer edges
- Add more foreground/background points
- Try adjusting Input Threshold

## Credits

- **SAM3**: Meta AI Research - Segment Anything Model 2/3
- **ViTMatte**: HUST Vision Lab - Vision Transformer for Image Matting
- **Grounding DINO**: IDEA Research - Open-Set Object Detection

## License

This gizmo is provided for educational and production use. SAM, ViTMatte, and Grounding DINO have their own licenses - please review them for commercial use.

---

*H2 SamViT v1.0 - H2 Studios 2026*
