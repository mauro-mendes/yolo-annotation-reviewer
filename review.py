"""
review_dataset_and_annotate_with_dataset_picker_v4.py
------------------------------------------------------
Novidades v4:
  - Handles de borda (quadrados laranjas): move só uma linha da bbox
  - Log de progresso: retoma de onde parou ao reabrir o mesmo dataset
  - Ir para imagem: digita número + Enter
  - Zoom (scroll / + / -) centrado no cursor; R enquadra imagem inteira
  - Pan com botão direito + arrastar
  - Painéis menores para maximizar área da imagem
  - Linhas mais finas
"""

import cv2
import json
import shutil
import math
import copy
import tkinter as tk
from tkinter import filedialog
import numpy as np
from pathlib import Path

# ─── Dataset state ────────────────────────────────────────────────────────────
DATASET_DIR    = None
IMAGE_DIRS     = []
LABEL_DIRS     = {}
REJECT_IMG_DIR = None
REJECT_LBL_DIR = None

CLASS_NAMES = {0: "rotomoldado"}

# ─── Layout ───────────────────────────────────────────────────────────────────
WINDOW           = "YOLO Review"
DEFAULT_CANVAS_W = 1280
DEFAULT_CANVAS_H = 960
TOP_PANEL_H      = 72
BOTTOM_PANEL_H   = 38
BG_COLOR         = (28, 28, 28)
PANEL_COLOR      = (40, 40, 40)

# ─── Colors ───────────────────────────────────────────────────────────────────
BOX_COLOR           = (0, 255, 0)
SELECTED_BOX_COLOR  = (0, 255, 255)
DRAG_BOX_COLOR      = (255, 255, 255)  # branco — enquanto arrasta
CORNER_HANDLE_COLOR = (0, 0, 255)    # vermelho — cantos
EDGE_HANDLE_COLOR   = (0, 140, 255)  # laranja  — meio das bordas
TEXT_COLOR          = (235, 235, 235)
SUBTEXT_COLOR       = (160, 160, 160)
WARN_COLOR          = (0, 80, 255)
INFO_COLOR          = (0, 220, 220)
NEW_BOX_COLOR       = (255, 0, 255)

# ─── Handle sizes ─────────────────────────────────────────────────────────────
HANDLE_RADIUS   = 7
MID_HANDLE_HALF = 5

# ─── Zoom / pan ───────────────────────────────────────────────────────────────
ZOOM_MIN  = 0.05
ZOOM_MAX  = 8.0
ZOOM_STEP = 1.15

zoom  = 1.0
pan_x = 0
pan_y = 0

is_panning       = False
pan_start_canvas = (0, 0)
pan_start_values = (0, 0)

# ─── Annotation state ─────────────────────────────────────────────────────────
is_fullscreen        = False
current_image        = None
current_boxes        = []
current_image_path   = None
current_label_path   = None
current_dirty        = False

selected_box_indices = set()
bbox_clipboard       = []

dragging       = False
drag_box_index = -1
drag_corner    = None

creating_box    = False
new_box_mode    = False
new_box_start   = None
new_box_current = None

image_offset_x = 0
image_offset_y = 0

number_buffer = ""


# ─── Progress log ─────────────────────────────────────────────────────────────

def get_progress_file(dataset_dir: Path) -> Path:
    return dataset_dir / ".review_progress.json"


def load_progress(dataset_dir: Path) -> int:
    try:
        data = json.loads(get_progress_file(dataset_dir).read_text(encoding="utf-8"))
        return max(0, int(data.get("last_index", 0)))
    except Exception:
        return 0


def save_progress(dataset_dir: Path, index: int):
    try:
        get_progress_file(dataset_dir).write_text(
            json.dumps({"last_index": index}, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[WARN] Não foi possível salvar progresso: {e}")


# ─── Dataset setup ────────────────────────────────────────────────────────────

def ask_dataset_dir():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Selecione a pasta raiz do dataset YOLO")
    root.destroy()
    return Path(selected) if selected else None


def validate_dataset_dir(dataset_dir: Path):
    has_img = (dataset_dir / "images" / "train").exists() or (dataset_dir / "images" / "val").exists()
    has_lbl = (dataset_dir / "labels" / "train").exists() or (dataset_dir / "labels" / "val").exists()
    if not has_img:
        raise FileNotFoundError("A pasta não possui images/train nem images/val.")
    if not has_lbl:
        raise FileNotFoundError("A pasta não possui labels/train nem labels/val.")


def configure_dataset_paths(dataset_dir: Path):
    global DATASET_DIR, IMAGE_DIRS, LABEL_DIRS, REJECT_IMG_DIR, REJECT_LBL_DIR
    DATASET_DIR = dataset_dir
    IMAGE_DIRS  = [dataset_dir / "images" / "train", dataset_dir / "images" / "val"]
    LABEL_DIRS  = {
        str(dataset_dir / "images" / "train"): dataset_dir / "labels" / "train",
        str(dataset_dir / "images" / "val"):   dataset_dir / "labels" / "val",
    }
    REJECT_IMG_DIR = dataset_dir / "rejected" / "images"
    REJECT_LBL_DIR = dataset_dir / "rejected" / "labels"


def create_reject_dirs():
    REJECT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    REJECT_LBL_DIR.mkdir(parents=True, exist_ok=True)


def collect_images():
    images = []
    for d in IMAGE_DIRS:
        if d.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                images.extend(sorted(d.glob(ext)))
    return sorted(set(images))


def get_label_path(image_path: Path) -> Path:
    return LABEL_DIRS[str(image_path.parent)] / f"{image_path.stem}.txt"


# ─── Label I/O ────────────────────────────────────────────────────────────────

def load_labels(label_path: Path, w: int, h: int) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts)
        x1 = max(0, min(int((xc - bw / 2) * w), w - 1))
        y1 = max(0, min(int((yc - bh / 2) * h), h - 1))
        x2 = max(0, min(int((xc + bw / 2) * w), w - 1))
        y2 = max(0, min(int((yc + bh / 2) * h), h - 1))
        if x2 > x1 and y2 > y1:
            boxes.append({"cls": int(cls), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def save_labels(label_path: Path, boxes: list, w: int, h: int):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        x1 = max(0, min(box["x1"], w - 1))
        y1 = max(0, min(box["y1"], h - 1))
        x2 = max(0, min(box["x2"], w - 1))
        y2 = max(0, min(box["y2"], h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        bw = x2 - x1
        bh = y2 - y1
        lines.append(
            f"{box['cls']} {(x1+bw/2)/w:.6f} {(y1+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ─── Box helpers ──────────────────────────────────────────────────────────────

def clamp_box(box: dict, w: int, h: int):
    box["x1"] = max(0, min(box["x1"], w - 1))
    box["y1"] = max(0, min(box["y1"], h - 1))
    box["x2"] = max(0, min(box["x2"], w - 1))
    box["y2"] = max(0, min(box["y2"], h - 1))
    if box["x1"] > box["x2"]: box["x1"], box["x2"] = box["x2"], box["x1"]
    if box["y1"] > box["y2"]: box["y1"], box["y2"] = box["y2"], box["y1"]
    if box["x2"] == box["x1"]: box["x2"] = min(w - 1, box["x1"] + 1)
    if box["y2"] == box["y1"]: box["y2"] = min(h - 1, box["y1"] + 1)


def copy_selected_boxes():
    global bbox_clipboard
    bbox_clipboard = [
        copy.deepcopy(current_boxes[i])
        for i in sorted(selected_box_indices)
        if 0 <= i < len(current_boxes)
    ]


def paste_clipboard_boxes(w: int, h: int) -> int:
    global current_boxes, current_dirty, selected_box_indices
    if not bbox_clipboard:
        return 0
    new_indices = []
    for box in bbox_clipboard:
        nb = copy.deepcopy(box)
        clamp_box(nb, w, h)
        current_boxes.append(nb)
        new_indices.append(len(current_boxes) - 1)
    if new_indices:
        selected_box_indices = set(new_indices)
        current_dirty = True
    return len(new_indices)


def delete_selected_boxes() -> int:
    global current_boxes, current_dirty, selected_box_indices
    if not selected_box_indices:
        return 0
    kept = [b for i, b in enumerate(current_boxes) if i not in selected_box_indices]
    removed = len(current_boxes) - len(kept)
    current_boxes = kept
    selected_box_indices = set()
    if removed:
        current_dirty = True
    return removed


def move_to_rejected(image_path: Path, label_path: Path):
    shutil.move(str(image_path), str(REJECT_IMG_DIR / image_path.name))
    if label_path.exists():
        shutil.move(str(label_path), str(REJECT_LBL_DIR / label_path.name))


# ─── Handle hit-test ──────────────────────────────────────────────────────────

def nearest_handle(box: dict, ix: int, iy: int):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    handles = {
        "tl": (x1, y1), "tr": (x2, y1), "bl": (x1, y2), "br": (x2, y2),
        "tm": (mx, y1), "bm": (mx, y2), "lm": (x1, my), "rm": (x2, my),
    }
    best, best_d = None, 1e9
    for name, (hx, hy) in handles.items():
        d = math.hypot(ix - hx, iy - hy)
        if d < best_d:
            best_d, best = d, name
    eff_r = max(HANDLE_RADIUS, MID_HANDLE_HALF) / zoom
    return best if best_d <= eff_r else None


def point_inside_box(box: dict, x: int, y: int) -> bool:
    return box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]


def find_box_at_point(boxes: list, x: int, y: int):
    for i in range(len(boxes) - 1, -1, -1):
        if point_inside_box(boxes[i], x, y):
            return i
    return None


# ─── Zoom helpers ─────────────────────────────────────────────────────────────

def get_canvas_size():
    try:
        _, _, w, h = cv2.getWindowImageRect(WINDOW)
        if w and h and w > 200 and h > 200:
            return w, h
    except cv2.error:
        pass
    return DEFAULT_CANVAS_W, DEFAULT_CANVAS_H


def fit_zoom_for(img_w: int, img_h: int) -> float:
    """Calcula o zoom que enquadra a imagem inteira na área de conteúdo."""
    canvas_w, canvas_h = get_canvas_size()
    content_w = canvas_w
    content_h = canvas_h - TOP_PANEL_H - BOTTOM_PANEL_H
    return min(content_w / img_w, content_h / img_h)


def reset_view(img_w: int = 0, img_h: int = 0):
    """Reseta zoom para enquadrar a imagem inteira; zera pan."""
    global zoom, pan_x, pan_y
    pan_x = 0
    pan_y = 0
    if img_w > 0 and img_h > 0:
        zoom = fit_zoom_for(img_w, img_h)
    else:
        # Recalcula com a imagem atual se disponível
        if current_image is not None:
            h, w = current_image.shape[:2]
            zoom = fit_zoom_for(w, h)
        else:
            zoom = 1.0


def apply_zoom_centered(new_zoom: float, cx: int, cy: int):
    """Aplica novo zoom mantendo o pixel de canvas (cx,cy) fixo."""
    global zoom, pan_x, pan_y
    if current_image is None or new_zoom == zoom:
        return
    canvas_w, canvas_h = get_canvas_size()
    content_h = canvas_h - TOP_PANEL_H - BOTTOM_PANEL_H
    orig_h, orig_w = current_image.shape[:2]

    new_disp_w = orig_w * new_zoom
    new_disp_h = orig_h * new_zoom
    new_base_x = (canvas_w - new_disp_w) / 2
    new_base_y = TOP_PANEL_H + (content_h - new_disp_h) / 2

    ix_f = (cx - image_offset_x) / zoom
    iy_f = (cy - image_offset_y) / zoom

    pan_x = int((cx - ix_f * new_zoom) - new_base_x)
    pan_y = int((cy - iy_f * new_zoom) - new_base_y)
    zoom  = new_zoom


# ─── Coordinate transform ─────────────────────────────────────────────────────

def canvas_to_image_coords(mx: int, my: int):
    if current_image is None:
        return None
    dx = (mx - image_offset_x) / zoom
    dy = (my - image_offset_y) / zoom
    h, w = current_image.shape[:2]
    if dx < 0 or dy < 0 or dx >= w or dy >= h:
        return None
    return int(dx), int(dy)


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_boxes(img: np.ndarray, boxes: list, z: float = 1.0) -> np.ndarray:
    out = img.copy()
    for i, box in enumerate(boxes):
        x1 = int(box["x1"] * z)
        y1 = int(box["y1"] * z)
        x2 = int(box["x2"] * z)
        y2 = int(box["y2"] * z)

        is_dragging_this = dragging and drag_box_index == i

        if is_dragging_this:
            color         = DRAG_BOX_COLOR
            thickness     = 1
            show_handles  = False
        elif i in selected_box_indices:
            color         = SELECTED_BOX_COLOR
            thickness     = 2
            show_handles  = True
        else:
            color         = BOX_COLOR
            thickness     = 1
            show_handles  = True

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = CLASS_NAMES.get(box["cls"], str(box["cls"]))
        cv2.putText(out, f"{i}:{label}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if show_handles:
            # Cantos: círculos
            for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.circle(out, (hx, hy), HANDLE_RADIUS, CORNER_HANDLE_COLOR, -1)
            # Meio das bordas: quadrados
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2
            for hx, hy in [(mx, y1), (mx, y2), (x1, my), (x2, my)]:
                cv2.rectangle(out,
                              (hx - MID_HANDLE_HALF, hy - MID_HANDLE_HALF),
                              (hx + MID_HANDLE_HALF, hy + MID_HANDLE_HALF),
                              EDGE_HANDLE_COLOR, -1)
    return out


def draw_new_box_preview(img: np.ndarray, z: float = 1.0) -> np.ndarray:
    if creating_box and new_box_start and new_box_current:
        x1 = int(min(new_box_start[0], new_box_current[0]) * z)
        y1 = int(min(new_box_start[1], new_box_current[1]) * z)
        x2 = int(max(new_box_start[0], new_box_current[0]) * z)
        y2 = int(max(new_box_start[1], new_box_current[1]) * z)
        cv2.rectangle(img, (x1, y1), (x2, y2), NEW_BOX_COLOR, 1)
        cv2.putText(img, "nova bbox", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEW_BOX_COLOR, 1, cv2.LINE_AA)
    return img


def draw_count_in_margin(canvas: np.ndarray, n: int,
                         zone_x: int, zone_w: int,
                         zone_y: int, zone_h: int,
                         alpha: float = 0.45):
    """Desenha o número de bboxes centralizado na faixa lateral (zona preta)."""
    if zone_w < 20 or zone_h < 20:
        return
    text = str(n)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Escala máxima que cabe na largura e na altura da zona
    for fs in [fs * 0.1 for fs in range(1, 500)]:
        thk = max(1, int(round(fs * 3)))
        (tw, th), bl = cv2.getTextSize(text, font, fs, thk)
        if tw > zone_w * 0.85 or th > zone_h * 0.75:
            break
    fs  = max(0.5, fs - 0.1)
    thk = max(1, int(round(fs * 3)))
    (tw, th), bl = cv2.getTextSize(text, font, fs, thk)

    ox = zone_x + (zone_w - tw) // 2
    oy = zone_y + (zone_h + th) // 2 - bl

    overlay = canvas.copy()
    cv2.putText(overlay, text, (ox, oy), font, fs, (0, 0, 0),     thk + 8, cv2.LINE_AA)
    cv2.putText(overlay, text, (ox, oy), font, fs, (255, 255, 255), thk,    cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


# ─── Canvas builder ───────────────────────────────────────────────────────────

def set_fullscreen(enabled: bool):
    global is_fullscreen
    is_fullscreen = enabled
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)


def build_canvas(index: int, total: int) -> np.ndarray:
    global image_offset_x, image_offset_y

    canvas_w, canvas_h = get_canvas_size()
    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    top_h = TOP_PANEL_H
    bot_h = BOTTOM_PANEL_H

    cv2.rectangle(canvas, (0, 0),              (canvas_w, top_h),    PANEL_COLOR, -1)
    cv2.rectangle(canvas, (0, canvas_h-bot_h), (canvas_w, canvas_h), PANEL_COLOR, -1)

    orig_h, orig_w = current_image.shape[:2]
    disp_w = max(1, int(orig_w * zoom))
    disp_h = max(1, int(orig_h * zoom))

    if abs(zoom - 1.0) > 1e-4:
        interp  = cv2.INTER_LINEAR if zoom > 1 else cv2.INTER_AREA
        img_scl = cv2.resize(current_image, (disp_w, disp_h), interpolation=interp)
    else:
        img_scl = current_image.copy()

    img_scl = draw_boxes(img_scl, current_boxes, zoom)
    img_scl = draw_new_box_preview(img_scl, zoom)

    content_h = canvas_h - top_h - bot_h
    base_x = (canvas_w - disp_w) // 2 + pan_x
    base_y = top_h + (content_h - disp_h) // 2 + pan_y

    image_offset_x = base_x
    image_offset_y = base_y

    dst_y0 = max(top_h,           base_y)
    dst_y1 = min(canvas_h-bot_h,  base_y + disp_h)
    dst_x0 = max(0,               base_x)
    dst_x1 = min(canvas_w,        base_x + disp_w)
    src_y0 = dst_y0 - base_y
    src_x0 = dst_x0 - base_x
    cp_w   = dst_x1 - dst_x0
    cp_h   = dst_y1 - dst_y0
    if cp_w > 0 and cp_h > 0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = \
            img_scl[src_y0:src_y0+cp_h, src_x0:src_x0+cp_w]

    # ── Número de bboxes nas margens laterais ────────────────────────────────
    left_w  = dst_x0                    # largura da faixa esquerda
    right_x = dst_x1                    # início da faixa direita
    right_w = canvas_w - dst_x1        # largura da faixa direita
    margin_y = top_h
    margin_h = canvas_h - top_h - bot_h
    draw_count_in_margin(canvas, len(current_boxes), 0,       left_w,  margin_y, margin_h)
    draw_count_in_margin(canvas, len(current_boxes), right_x, right_w, margin_y, margin_h)

    # ── Painel superior ──────────────────────────────────────────────────────
    split   = current_image_path.parent.name
    title   = f"{index+1}/{total}  [{split}]  {current_image_path.name}"
    status  = (f"Boxes:{len(current_boxes)}  Sel:{len(selected_box_indices)}  "
               f"Clip:{len(bbox_clipboard)}  Zoom:{zoom:.2f}x")

    cv2.putText(canvas, title,  (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, TEXT_COLOR,    1, cv2.LINE_AA)
    cv2.putText(canvas, status, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, SUBTEXT_COLOR, 1, cv2.LINE_AA)

    # Linha 3 do painel: avisos / modos ativos
    warn_y = top_h - 8
    msgs = []
    if current_dirty:  msgs.append("EDITADO")
    if new_box_mode:   msgs.append("NOVA BBOX: clique e arraste")
    if number_buffer:  msgs.append(f"Ir para: {number_buffer}|")
    if msgs:
        cv2.putText(canvas, "  |  ".join(msgs), (10, warn_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, WARN_COLOR if current_dirty else INFO_COLOR,
                    1, cv2.LINE_AA)

    # ── Painel inferior ──────────────────────────────────────────────────────
    help_txt = ("D/->:prox  A/<-:voltar  X:reject  N:nova bbox  "
                "+/-/scroll:zoom  R:fit  BotDir+drag:pan  "
                "digitos+Enter:ir para  C:copiar  V:colar  Del:apagar  F:fullscreen  Q:sair")
    cv2.putText(canvas, help_txt, (10, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, SUBTEXT_COLOR, 1, cv2.LINE_AA)

    return canvas


# ─── Mouse callback ───────────────────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    global dragging, drag_box_index, drag_corner
    global current_boxes, current_image, current_dirty, selected_box_indices
    global creating_box, new_box_mode, new_box_start, new_box_current
    global is_panning, pan_start_canvas, pan_start_values, pan_x, pan_y

    # ── Scroll: zoom centrado no cursor ─────────────────────────────────────
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            apply_zoom_centered(min(ZOOM_MAX, zoom * ZOOM_STEP), x, y)
        else:
            apply_zoom_centered(max(ZOOM_MIN, zoom / ZOOM_STEP), x, y)
        return

    # ── Botão direito: pan ───────────────────────────────────────────────────
    if event == cv2.EVENT_RBUTTONDOWN:
        is_panning       = True
        pan_start_canvas = (x, y)
        pan_start_values = (pan_x, pan_y)
        return
    if event == cv2.EVENT_MOUSEMOVE and is_panning:
        pan_x = pan_start_values[0] + (x - pan_start_canvas[0])
        pan_y = pan_start_values[1] + (y - pan_start_canvas[1])
        return
    if event == cv2.EVENT_RBUTTONUP:
        is_panning = False
        return

    if current_image is None:
        return

    img_coords = canvas_to_image_coords(x, y)

    if img_coords is None:
        if event == cv2.EVENT_LBUTTONUP:
            dragging = False; drag_box_index = -1; drag_corner = None
            if creating_box:
                creating_box = False; new_box_start = new_box_current = None
        return

    ix, iy = img_coords
    h, w   = current_image.shape[:2]
    ix = max(0, min(ix, w - 1))
    iy = max(0, min(iy, h - 1))

    # ── Modo nova bbox ───────────────────────────────────────────────────────
    if new_box_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            creating_box = True; new_box_start = new_box_current = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and creating_box:
            new_box_current = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and creating_box:
            bx1 = min(new_box_start[0], ix); by1 = min(new_box_start[1], iy)
            bx2 = max(new_box_start[0], ix); by2 = max(new_box_start[1], iy)
            if (bx2 - bx1) >= 5 and (by2 - by1) >= 5:
                current_boxes.append({"cls": 0, "x1": bx1, "y1": by1, "x2": bx2, "y2": by2})
                selected_box_indices = {len(current_boxes) - 1}
                current_dirty = True
            creating_box = False; new_box_start = new_box_current = None; new_box_mode = False
        return

    # ── Drag / select ────────────────────────────────────────────────────────
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(current_boxes):
            hdl = nearest_handle(box, ix, iy)
            if hdl is not None:
                dragging = True; drag_box_index = i; drag_corner = hdl
                selected_box_indices = {i}
                return
        clicked = find_box_at_point(current_boxes, ix, iy)
        if clicked is not None:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if clicked in selected_box_indices: selected_box_indices.remove(clicked)
                else: selected_box_indices.add(clicked)
            else:
                selected_box_indices = {clicked}
            return
        selected_box_indices = set()

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        if not (0 <= drag_box_index < len(current_boxes)): return
        box = current_boxes[drag_box_index]
        if drag_corner in ("tl", "bl", "lm"): box["x1"] = ix
        if drag_corner in ("tl", "tr", "tm"): box["y1"] = iy
        if drag_corner in ("tr", "br", "rm"): box["x2"] = ix
        if drag_corner in ("bl", "br", "bm"): box["y2"] = iy
        clamp_box(box, w, h)
        current_dirty = True

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False; drag_box_index = -1; drag_corner = None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global current_image, current_boxes, current_image_path, current_label_path, current_dirty
    global new_box_mode, creating_box, new_box_start, new_box_current, selected_box_indices
    global dragging, drag_box_index, drag_corner, number_buffer

    dataset_dir = ask_dataset_dir()
    if dataset_dir is None:
        print("Nenhuma pasta selecionada. Encerrando.")
        return

    try:
        validate_dataset_dir(dataset_dir)
    except FileNotFoundError as e:
        print(f"[ERRO] {e}"); return

    configure_dataset_paths(dataset_dir)
    create_reject_dirs()

    images = collect_images()
    if not images:
        print("[ERRO] Nenhuma imagem encontrada."); return

    start_i = min(load_progress(dataset_dir), len(images) - 1)
    if start_i > 0:
        print(f"[INFO] Retomando do índice {start_i + 1}/{len(images)}")

    print(f"Dataset: {dataset_dir}  |  {len(images)} imagens")

    i = start_i
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, DEFAULT_CANVAS_W, DEFAULT_CANVAS_H)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    while 0 <= i < len(images):
        img_path = images[i]
        if not img_path.exists():
            images.pop(i); continue

        label_path = get_label_path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Erro ao abrir: {img_path}"); i += 1; continue

        h, w = img.shape[:2]
        current_image        = img
        current_boxes        = load_labels(label_path, w, h)
        current_image_path   = img_path
        current_label_path   = label_path
        current_dirty        = False
        selected_box_indices = set()
        dragging = False; drag_box_index = -1; drag_corner = None
        new_box_mode = False; creating_box = False
        new_box_start = new_box_current = None
        number_buffer = ""
        reset_view(w, h)   # enquadra imagem inteira

        while True:
            canvas = build_canvas(i, len(images))
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKeyEx(30)
            if key == -1:
                continue

            # ── Captura de dígitos ────────────────────────────────────────
            if ord('0') <= key <= ord('9'):
                number_buffer += chr(key)
                continue

            # ── Processamento do buffer ───────────────────────────────────
            if number_buffer:
                if key == 13:   # Enter → pular para imagem
                    try:
                        target = max(0, min(int(number_buffer) - 1, len(images) - 1))
                        if current_dirty:
                            save_labels(label_path, current_boxes, w, h)
                            current_dirty = False
                        save_progress(dataset_dir, target)
                        i = target; number_buffer = ""; break
                    except ValueError:
                        number_buffer = ""
                    continue
                elif key == 8:  # Backspace → apaga dígito
                    number_buffer = number_buffer[:-1]; continue
                elif key == 27: # Esc → cancela sem sair
                    number_buffer = ""; continue
                else:
                    number_buffer = ""  # cai no processamento normal

            # ── Zoom por teclado ──────────────────────────────────────────
            if key in (ord('+'), ord('=')):
                apply_zoom_centered(min(ZOOM_MAX, zoom * ZOOM_STEP),
                                    DEFAULT_CANVAS_W // 2, DEFAULT_CANVAS_H // 2)
                continue
            if key in (ord('-'), ord('_')):
                apply_zoom_centered(max(ZOOM_MIN, zoom / ZOOM_STEP),
                                    DEFAULT_CANVAS_W // 2, DEFAULT_CANVAS_H // 2)
                continue

            # ── Navegação ────────────────────────────────────────────────
            if key in (ord('d'), ord('D'), 2555904):
                if current_dirty: save_labels(label_path, current_boxes, w, h)
                i += 1; save_progress(dataset_dir, i); break

            elif key in (ord('a'), ord('A'), 2424832):
                if current_dirty: save_labels(label_path, current_boxes, w, h)
                i = max(0, i - 1); save_progress(dataset_dir, i); break

            elif key in (ord('x'), ord('X')):
                if current_dirty: save_labels(label_path, current_boxes, w, h)
                print(f"Rejected: {img_path.name}")
                move_to_rejected(img_path, label_path)
                images.pop(i)
                if i >= len(images): i = max(0, len(images) - 1)
                save_progress(dataset_dir, i); break

            elif key in (ord('n'), ord('N')):
                new_box_mode = True; creating_box = False
                new_box_start = new_box_current = None
                print("[INFO] Modo nova bbox. Clique e arraste.")

            elif key in (ord('c'), ord('C')):
                copy_selected_boxes()
                print(f"[INFO] {len(bbox_clipboard)} bbox(es) copiada(s).")

            elif key in (ord('v'), ord('V')):
                n = paste_clipboard_boxes(w, h)
                print(f"[INFO] {n} bbox(es) colada(s)." if n else "[INFO] Clipboard vazio.")

            elif key in (8, 127, 3014656, 3014658, 3014660):
                n = delete_selected_boxes()
                print(f"[INFO] {n} bbox(es) apagada(s)." if n else "[INFO] Nada selecionado.")

            elif key in (ord('r'), ord('R')):
                reset_view(w, h)

            elif key in (ord('f'), ord('F')):
                set_fullscreen(not is_fullscreen)

            elif key in (ord('q'), ord('Q'), 27):
                if current_dirty: save_labels(label_path, current_boxes, w, h)
                save_progress(dataset_dir, i)
                cv2.destroyAllWindows(); return

    cv2.destroyAllWindows()
    if DATASET_DIR:
        save_progress(DATASET_DIR, i)


if __name__ == "__main__":
    main()
