# YOLO Annotation Reviewer

Keyboard-driven OpenCV tool for reviewing and editing YOLO bounding-box annotations.
Designed for semi-automatic annotation pipelines where a model generates initial labels
and a human reviews them frame by frame.

## Features

- **Zoom & pan** — mouse scroll to zoom centered on cursor; right-click drag to pan; `R` to fit image
- **Bounding box editing** — drag corner handles (circles) or edge handles (squares) to resize
- **New bbox** — press `N`, then click and drag to draw a new bounding box
- **Copy / paste** — `C` copies selected boxes, `V` pastes them to the current frame
- **Multi-select** — `Ctrl+click` to select multiple boxes; `Delete` to remove selected
- **Frame rejection** — `X` moves the current image and its label to a `rejected/` subfolder
- **Progress persistence** — resumes from where you left off when reopening the same dataset
- **Direct navigation** — type a frame number and press `Enter` to jump to it
- **Fullscreen** — `F` toggles fullscreen mode
- **Auto-save** — labels are saved automatically on every frame advance

## Requirements

```
pip install opencv-python numpy
```

Python 3.8+ and `tkinter` (included in standard Python distributions).

## Dataset structure

The tool expects a standard YOLO dataset layout:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Labels must be in YOLO `.txt` format (one line per box: `class cx cy w h`, normalized).

## Usage

```bash
python review.py
```

A folder picker dialog opens. Select the root folder of your YOLO dataset.
The tool loads all images from `images/train` and `images/val` and their
corresponding labels from `labels/train` and `labels/val`.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `D` / `→` | Next frame (saves if edited) |
| `A` / `←` | Previous frame (saves if edited) |
| `X` | Reject frame (moves to `rejected/`) |
| `N` | New bbox mode (click and drag) |
| `C` | Copy selected boxes |
| `V` | Paste boxes |
| `Delete` | Delete selected boxes |
| `R` | Fit image to window |
| `+` / `-` | Zoom in / out |
| `Scroll` | Zoom centered on cursor |
| `Right-drag` | Pan |
| `0–9` + `Enter` | Jump to frame number |
| `F` | Toggle fullscreen |
| `Q` / `Esc` | Quit (saves current frame) |

## Citation

If you use this tool in your research, please cite it using the information in [CITATION.cff](CITATION.cff) or via the **"Cite this repository"** button on GitHub.

## License

MIT — see [LICENSE](LICENSE).
