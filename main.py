import sys
import os
import json
import multiprocessing
import time
import queue
import functools
import contextlib
import urllib.request
import numpy as np
import OpenImageIO as oiio
import cv2
import math
import shutil
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QProgressBar, QLabel, 
                               QFileDialog, QTabWidget, QMessageBox, QGraphicsView,
                               QGraphicsScene, QSlider, QComboBox, QGroupBox, QGridLayout, QSpinBox,
                               QInputDialog, QDialog)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtCore import QTimer, Qt, QPointF, QThread, Signal

# Directories
MODELS_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Models Dictionary
MODELS = {
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    "cotracker3/scaled_online.pth": "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth",
    "vggsfm/vggsfm_v2_0_0.bin": "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_0_0.bin"
}

# Color Space Constants
CS_LINEAR_SRGB = "Linear sRGB"
CS_ACESCG = "ACEScg"

def apply_color_space(rgb_array, color_space):
    if color_space == CS_ACESCG:
        acescg_to_srgb = np.array([
            [1.705051, -0.621861, -0.083190],
            [-0.130256, 1.140802, -0.010546],
            [-0.024007, -0.128967, 1.152974]
        ])
        shape = rgb_array.shape
        flat_rgb = rgb_array.reshape(-1, 3)
        flat_linear = np.dot(flat_rgb, acescg_to_srgb.T)
        rgb_array = flat_linear.reshape(shape)

    linear = np.clip(rgb_array, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 
                    linear * 12.92, 
                    1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return srgb

@functools.lru_cache(maxsize=50)
def fetch_qpixmap(frame_path, color_space, mask_path=None, mask_mtime=0):
    try:
        input_image = oiio.ImageInput.open(frame_path)
        if not input_image:
            return QPixmap()
        
        image_data = input_image.read_image()
        input_image.close()

        if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
            rgb_array = image_data[:, :, :3]
        elif len(image_data.shape) == 2:
            rgb_array = np.stack((image_data,)*3, axis=-1)
        else:
            return QPixmap()
            
        srgb_array = apply_color_space(rgb_array, color_space)
        rgb_uint8 = np.clip(srgb_array * 255.0, 0, 255).astype(np.uint8)
        height, width, channel = rgb_uint8.shape
        
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape != (height, width):
                    mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
                
                lut = np.zeros((256, 3), dtype=np.uint8)
                lut[255] = [200, 20, 40] # Ruby Red for alpha
                for obj_id in range(1, 100):
                    hue = (obj_id * 137.508) % 360
                    color = QColor.fromHsv(int(hue), 255, 255)
                    lut[obj_id] = [color.red(), color.green(), color.blue()]
                    
                colored_mask = lut[mask_img]
                mask_bool = mask_img > 0
                
                blended = cv2.addWeighted(rgb_uint8, 1 - 0.45, colored_mask, 0.45, 0)
                rgb_uint8[mask_bool] = blended[mask_bool]

        bytes_per_line = 3 * width
        qimg = QImage(rgb_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)
    except Exception as e:
        print(f"Failed to fetch qpixmap: {e}")
        return QPixmap()

@functools.lru_cache(maxsize=200)
def fetch_proxy_qpixmap(proxy_path, mask_path=None, mask_mtime=0):
    try:
        if not os.path.exists(proxy_path):
            return QPixmap()
            
        proxy_img = cv2.imread(proxy_path)
        if proxy_img is None: return QPixmap()
        proxy_img = cv2.cvtColor(proxy_img, cv2.COLOR_BGR2RGB)
        
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask_img = cv2.resize(mask_img, (proxy_img.shape[1], proxy_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                lut = np.zeros((256, 3), dtype=np.uint8)
                lut[255] = [200, 20, 40] # Ruby Red for alpha
                for obj_id in range(1, 100):
                    hue = (obj_id * 137.508) % 360
                    color = QColor.fromHsv(int(hue), 255, 255)
                    lut[obj_id] = [color.red(), color.green(), color.blue()]
                    
                colored_mask = lut[mask_img]
                mask_bool = mask_img > 0
                
                blended = cv2.addWeighted(proxy_img, 1 - 0.45, colored_mask, 0.45, 0)
                proxy_img[mask_bool] = blended[mask_bool]
                
        h, w, c = proxy_img.shape
        bytes_per_line = 3 * w
        qimg = QImage(proxy_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)
    except Exception as e:
        print(f"Failed to fetch proxy qpixmap: {e}")
        return QPixmap()

class DownloadWorker(QThread):
    progress = Signal(int)
    finished = Signal()
    error = Signal(str)

    def run(self):
        try:
            items_to_download = [k for k, v in MODELS.items() if v != "placeholder" and not os.path.exists(os.path.join(MODELS_DIR, k))]
            if not items_to_download:
                self.progress.emit(100)
                self.finished.emit()
                return

            for i, model_name in enumerate(items_to_download):
                url = MODELS[model_name]
                out_path = os.path.join(MODELS_DIR, model_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                def reporthook(count, block_size, total_size):
                    if total_size > 0:
                        percent = (count * block_size * 100) / total_size
                        base_prog = (i / len(items_to_download)) * 100
                        file_prog = percent / len(items_to_download)
                        val = int(base_prog + file_prog)
                        if val > 100: val = 100
                        self.progress.emit(val)

                urllib.request.urlretrieve(url, out_path, reporthook)
                
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

from PySide6.QtWidgets import QStyle, QStyleOptionSlider

class KeyframeSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.keyframes = set()

    def set_keyframes(self, frames):
        self.keyframes = set(frames)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.keyframes or self.maximum() <= 0:
            return
            
        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        
        groove_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        
        # Yellow ticks for visibility
        painter.setPen(QPen(QColor(255, 215, 0), 2))
        
        for frame in self.keyframes:
            ratio = frame / self.maximum()
            x = groove_rect.left() + int(ratio * groove_rect.width())
            painter.drawLine(x, groove_rect.top(), x, groove_rect.bottom())
            
        painter.end()

class ClickableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent_widget):
        super().__init__(scene)
        self.parent_widget = parent_widget
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self.parent_widget.pixmap_item:
            scene_pos = self.mapToScene(event.position().toPoint())
            item_pos = self.parent_widget.pixmap_item.mapFromScene(scene_pos)
            pixmap_rect = self.parent_widget.pixmap_item.boundingRect()
            if pixmap_rect.contains(item_pos):
                x, y = int(item_pos.x()), int(item_pos.y())
                frame_idx = self.parent_widget.slider.value()
                
                if self.parent_widget.interactive and hasattr(self.parent_widget, 'native_size') and self.parent_widget.native_size != (0,0):
                    proxy_w = self.parent_widget.pixmap_item.pixmap().width()
                    native_w, native_h = self.parent_widget.native_size
                    if proxy_w > 0 and proxy_w != native_w:
                        proxy_h = self.parent_widget.pixmap_item.pixmap().height()
                        scale_w = native_w / proxy_w
                        scale_h = native_h / proxy_h
                        x = int(x * scale_w)
                        y = int(y * scale_h)

                if event.button() == Qt.LeftButton:
                    self.parent_widget.add_click(x, y, frame_idx, 1, self.parent_widget.spin_obj.value()) # 1 = Positive
                elif event.button() == Qt.RightButton:
                    self.parent_widget.add_click(x, y, frame_idx, 0, self.parent_widget.spin_obj.value()) # 0 = Negative
                elif event.button() == Qt.MiddleButton or (event.modifiers() == Qt.AltModifier and event.button() == Qt.LeftButton):
                    self.parent_widget.remove_closest_click(x, y, frame_idx)
                    
        super().mousePressEvent(event)

class SequenceViewerWidget(QWidget):
    # Signals to communicate with MainWindow
    clicksUpdated = Signal(int) # emits frame_idx

    def __init__(self, parent=None, interactive=False, show_tracks=False):
        super().__init__(parent)
        self.exr_files = []
        self.color_space = CS_LINEAR_SRGB
        self.project_dir = None
        self.interactive = interactive
        self.show_tracks = show_tracks
        self.show_overlay = False
        self.native_size = (0, 0)
        
        # [(x, y, frame_idx, type, obj_id), ...] where type: 1=pos, 0=neg
        self.click_data = []
        
        self.track_data = None
        self.track_vis = None
        
        layout = QVBoxLayout(self)
        
        self.scene = QGraphicsScene()
        if self.interactive:
            self.view = ClickableGraphicsView(self.scene, self)
        else:
            self.view = QGraphicsView(self.scene)
            
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.view)
        
        self.loading_overlay = QLabel(self.view)
        self.loading_overlay.setText("Initializing SAM 2...\nPlease wait.")
        self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 180); color: white; font-size: 24px; font-weight: bold;")
        self.loading_overlay.setAlignment(Qt.AlignCenter)
        self.loading_overlay.hide()
        
        ctrl_layout = QHBoxLayout()
        self.slider = KeyframeSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_frame_changed)
        
        self.lbl_frame = QLabel("Frame: 0/0")
        
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.lbl_frame)
        layout.addLayout(ctrl_layout)
        
        self.pixmap_item = None
        
        if self.interactive:
            btn_prev_kf = QPushButton("|< Prev Keyframe")
            btn_prev_kf.clicked.connect(self.jump_prev_keyframe)
            btn_next_kf = QPushButton("Next Keyframe >|")
            btn_next_kf.clicked.connect(self.jump_next_keyframe)
            ctrl_layout.addWidget(btn_prev_kf)
            ctrl_layout.addWidget(btn_next_kf)
            
            self.spin_obj = QSpinBox()
            self.spin_obj.setMinimum(1)
            self.spin_obj.setMaximum(99)
            self.spin_obj.setPrefix("Object ID: ")
            self.spin_obj.setToolTip("Change this number to select distinct objects for tracking")
            ctrl_layout.addWidget(self.spin_obj)

            btn_clear = QPushButton("Clear All Points")
            btn_clear.clicked.connect(self.clear_clicks)
            ctrl_layout.addWidget(btn_clear)

    def jump_prev_keyframe(self):
        curr = self.slider.value()
        keyframes = sorted(list(set([f_idx for cx, cy, f_idx, ptype, obj_id in self.click_data])))
        for kf in reversed(keyframes):
            if kf < curr:
                self.slider.setValue(kf)
                return

    def jump_next_keyframe(self):
        curr = self.slider.value()
        keyframes = sorted(list(set([f_idx for cx, cy, f_idx, ptype, obj_id in self.click_data])))
        for kf in keyframes:
            if kf > curr:
                self.slider.setValue(kf)
                return

    def set_loading_state(self, is_loading):
        if is_loading:
            self.loading_overlay.resize(self.view.size())
            self.loading_overlay.show()
            self.view.setEnabled(False)
        else:
            self.loading_overlay.hide()
            self.view.setEnabled(True)

    def save_clicks(self):
        if self.project_dir:
            clicks_file = os.path.join(self.project_dir, 'clicks.json')
            try:
                with open(clicks_file, 'w') as f:
                    json.dump(self.click_data, f)
            except Exception as e:
                print(f"Failed to save clicks: {e}")
                
    def update_keyframes(self):
        frames = set([f_idx for cx, cy, f_idx, ptype, obj_id in self.click_data])
        if hasattr(self.slider, 'set_keyframes'):
            self.slider.set_keyframes(frames)

    def add_click(self, x, y, frame_idx, ptype, obj_id):
        self.click_data.append((x, y, frame_idx, ptype, obj_id))
        print(f"Added point ({x}, {y}, type={ptype}, obj={obj_id}) on frame {frame_idx}")
        self.save_clicks()
        self.update_keyframes()
        self.clicksUpdated.emit(frame_idx)
        self.on_frame_changed(self.slider.value())

    def remove_closest_click(self, x, y, frame_idx):
        if not self.click_data: return
        
        # Dynamic distance threshold based on zoom/proxy state
        threshold = 30
        if hasattr(self, 'native_size') and self.native_size != (0,0) and self.pixmap_item:
            proxy_w = self.pixmap_item.pixmap().width()
            native_w = self.native_size[0]
            if proxy_w > 0 and proxy_w != native_w:
                threshold = 30 * (native_w / proxy_w)
                
        closest_idx = -1
        min_dist = float('inf')
        for i, (cx, cy, f_idx, ptype, obj_id) in enumerate(self.click_data):
            if f_idx == frame_idx:
                dist = math.hypot(cx - x, cy - y)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
        
        if closest_idx != -1 and min_dist < threshold:
            removed = self.click_data.pop(closest_idx)
            print(f"Removed point {removed}")
            self.save_clicks()
            self.update_keyframes()
            self.clicksUpdated.emit(frame_idx)
            self.on_frame_changed(self.slider.value())

    def clear_clicks(self):
        self.click_data.clear()
        self.save_clicks()
        self.update_keyframes()
        self.on_frame_changed(self.slider.value())
        self.clicksUpdated.emit(-1)

    def update_sequence(self, file_list, color_space, project_dir):
        self.exr_files = file_list
        self.color_space = color_space
        self.project_dir = project_dir
        if self.exr_files:
            try:
                inp = oiio.ImageInput.open(self.exr_files[0])
                self.native_size = (inp.spec().width, inp.spec().height)
                inp.close()
            except Exception:
                pass
                
            clicks_file = os.path.join(self.project_dir, 'clicks.json')
            if os.path.exists(clicks_file):
                try:
                    with open(clicks_file, 'r') as f:
                        self.click_data = json.load(f)
                        self.click_data = [tuple(c) for c in self.click_data]
                except Exception as e:
                    print(f"Failed to load clicks: {e}")
                    self.click_data = []
            else:
                self.click_data = []
            
            # Auto-load track data from tracks.npz — only for the tracking viewer
            if self.show_tracks:
                tracks_path = os.path.join(self.project_dir, 'tracks.npz')
                if os.path.exists(tracks_path):
                    try:
                        data = np.load(tracks_path)
                        self.track_data = data['tracks']   # (T, N, 2) in native coords
                        self.track_vis  = data['visibility']  # (T, N)
                        print(f"[Viewer] Loaded {self.track_data.shape[1]} tracks from {tracks_path}")
                    except Exception as e:
                        print(f"[Viewer] Failed to load tracks.npz: {e}")
            
            self.update_keyframes()
            
            self.slider.setMaximum(len(self.exr_files) - 1)
            curr = self.slider.value()
            self.slider.setEnabled(True)
            self.on_frame_changed(curr)
        else:
            self.slider.setEnabled(False)
            self.scene.clear()
            self.pixmap_item = None
            self.lbl_frame.setText("Frame: 0/0")
            self.click_data.clear()
            self.update_keyframes()

    def set_color_space(self, color_space):
        self.color_space = color_space
        if self.exr_files:
            self.on_frame_changed(self.slider.value())

    def set_track_data(self, tracks, vis):
        self.track_data = tracks
        self.track_vis = vis
        if self.exr_files:
            self.on_frame_changed(self.slider.value())

    def set_overlay_mode(self, enabled):
        self.show_overlay = enabled
        if self.exr_files:
            self.on_frame_changed(self.slider.value())

    def on_frame_changed(self, index):
        if not self.exr_files or index < 0 or index >= len(self.exr_files):
            return
            
        frame_path = self.exr_files[index]
        
        mask_path = None
        mask_mtime = 0
        if self.show_overlay and self.project_dir:
            base_name = os.path.splitext(os.path.basename(frame_path))[0]
            mask_path = os.path.join(self.project_dir, 'masks', f"{base_name}_mask.png")
            if os.path.exists(mask_path):
                mask_mtime = os.path.getmtime(mask_path)
            
        base_pixmap = QPixmap()
        if self.project_dir:
            proxy_path = os.path.join(self.project_dir, 'proxies', f"{index:05d}.jpg")
            if os.path.exists(proxy_path):
                base_pixmap = fetch_proxy_qpixmap(proxy_path, mask_path, mask_mtime)
                
        if base_pixmap.isNull():
            base_pixmap = fetch_qpixmap(frame_path, self.color_space, mask_path, mask_mtime)
        
        if base_pixmap.isNull():
            return

        if self.interactive and self.click_data:
            display_pixmap = QPixmap(base_pixmap)
            painter = QPainter(display_pixmap)
            
            # scaling ratios if using proxies
            scale_x, scale_y = 1.0, 1.0
            if self.native_size != (0,0):
                native_w, native_h = self.native_size
                proxy_w = display_pixmap.width()
                proxy_h = display_pixmap.height()
                if proxy_w > 0 and proxy_w != native_w:
                    scale_x = proxy_w / native_w
                    scale_y = proxy_h / native_h
            
            for cx, cy, f_idx, ptype, obj_id in self.click_data:
                if f_idx == index:
                    draw_x = int(cx * scale_x)
                    draw_y = int(cy * scale_y)
                    
                    if ptype == 1:
                        hue = (obj_id * 137.508) % 360
                        color = QColor.fromHsv(int(hue), 255, 255)
                        painter.setPen(QPen(color, 4))
                        painter.setBrush(color)
                    else:
                        color = QColor("red")
                        painter.setPen(QPen(color, 4))
                        painter.setBrush(color)
                    painter.drawEllipse(QPointF(draw_x, draw_y), 6, 6)
            painter.end()
            final_pixmap = display_pixmap
        elif self.track_data is not None and self.track_vis is not None:
            display_pixmap = QPixmap(base_pixmap)
            painter = QPainter(display_pixmap)
            
            scale_x, scale_y = 1.0, 1.0
            if self.native_size != (0,0):
                native_w, native_h = self.native_size
                proxy_w = display_pixmap.width()
                if proxy_w > 0 and proxy_w != native_w:
                    scale_x = proxy_w / native_w
                    scale_y = display_pixmap.height() / native_h

            color = QColor(0, 255, 0, 200)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            
            if index < self.track_data.shape[0]:
                points = self.track_data[index]
                vis = self.track_vis[index]
                for i in range(points.shape[0]):
                    if vis[i]:
                        draw_x = int(points[i, 0] * scale_x)
                        draw_y = int(points[i, 1] * scale_y)
                        painter.drawEllipse(QPointF(draw_x, draw_y), 3, 3)
            painter.end()
            final_pixmap = display_pixmap
        else:
            final_pixmap = base_pixmap
        
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(final_pixmap)
        else:
            self.pixmap_item.setPixmap(final_pixmap)
            
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.lbl_frame.setText(f"Frame: {index + 1}/{len(self.exr_files)}")
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'loading_overlay') and self.loading_overlay.isVisible():
            self.loading_overlay.resize(self.view.size())
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# --- PERSISTENT WORKER DAEMON ---
def persistent_worker_daemon(cmd_queue, res_queue, exr_files, project_dir, native_sizes, proxy_res):
    print(f"\n[Daemon] Started Masking Daemon.", flush=True)
    masks_dir = os.path.join(project_dir, 'masks')
    proxies_dir = os.path.join(project_dir, 'proxies')
    alphas_dir = os.path.join(project_dir, 'alphas')
    user_masks_dir = os.path.join(project_dir, 'user_masks')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(proxies_dir, exist_ok=True)
    os.makedirs(alphas_dir, exist_ok=True)
    os.makedirs(user_masks_dir, exist_ok=True)
    
    try:
        import torch
        import gc
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        torch = None
        print("[Daemon] WARNING: PyTorch not found. Simulating inference.", flush=True)

    sam_model = None
    inference_state = None
    if torch and torch.cuda.is_available():
        print(f"[Daemon] Loading real SAM 2 model weights into VRAM...", flush=True)
        try:
            from hydra import initialize_config_module, core
            core.global_hydra.GlobalHydra.instance().clear()
            initialize_config_module("sam2.configs", version_base="1.2")
            from sam2.build_sam import build_sam2_video_predictor
            
            ckpt_path = os.path.join(MODELS_DIR, "sam2_hiera_large.pt")
            if os.path.exists(ckpt_path):
                sam_model = build_sam2_video_predictor("sam2/sam2_hiera_l.yaml", ckpt_path, device=device)
            else:
                raise FileNotFoundError("SAM 2 weights not found.")
        except Exception as e:
            print(f"[Daemon] Real SAM 2 failed to load: {e}. Simulating tensor.", flush=True)
            sam_model = torch.randn((1, 256, 1024, 1024), device=device)
        print(f"[Daemon] VRAM after allocation: {torch.cuda.memory_allocated() / (1024**2):.2f} MB", flush=True)

    res_queue.put({"status": "init_progress", "value": 100})
    if sam_model is not None and not isinstance(sam_model, torch.Tensor):
        print("[Daemon] Initializing SAM 2 Inference State...", flush=True)
        inference_state = sam_model.init_state(video_path=proxies_dir)

    def get_master_mask(frame_idx, proxy_mask):
        native_w, native_h = native_sizes[frame_idx]
        upscaled_mask = cv2.resize(proxy_mask, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        
        alpha_path = os.path.join(alphas_dir, f"alpha_{frame_idx:05d}.png")
        if os.path.exists(alpha_path):
            alpha_img = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
            if alpha_img is not None:
                upscaled_mask = cv2.bitwise_or(upscaled_mask, alpha_img)
                
        um_path = os.path.join(user_masks_dir, f"umask_{frame_idx:05d}.png")
        if os.path.exists(um_path):
            um_img = cv2.imread(um_path, cv2.IMREAD_GRAYSCALE)
            if um_img is not None:
                if um_img.shape != upscaled_mask.shape:
                    um_img = cv2.resize(um_img, (upscaled_mask.shape[1], upscaled_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                upscaled_mask = cv2.bitwise_or(upscaled_mask, um_img)
                
        return upscaled_mask

    def save_mask(frame_idx, master_mask):
        frame_path = exr_files[frame_idx]
        base_name = os.path.splitext(os.path.basename(frame_path))[0]
        out_mask = os.path.join(masks_dir, f"{base_name}_mask.png")
        # Ensure any masked pixel (SAM objects 1, 2, 3 or alphas) is saved as 255 (White)
        final_png = (master_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(out_mask, final_png)

    def simulate_frame(frame_idx, clicks):
        native_w, native_h = native_sizes[frame_idx]
        scale = proxy_res / max(native_w, native_h) if (native_w > proxy_res or native_h > proxy_res) else 1.0
        proxy_mask = np.zeros((int(native_h * scale), int(native_w * scale)), dtype=np.uint8)
        frame_clicks = [(cx, cy, ptype) for cx, cy, f_idx, ptype, obj_id in clicks if f_idx == frame_idx]
        for cx, cy, ptype in frame_clicks:
            val = 255 if ptype == 1 else 0
            cv2.circle(proxy_mask, (int(cx * scale), int(cy * scale)), int(50 * scale), val, -1)
        master_mask = get_master_mask(frame_idx, proxy_mask)
        save_mask(frame_idx, master_mask)

    while True:
        cmd = cmd_queue.get()
        action = cmd.get("action")
        
        if action == "interactive_mask":
            f_idx = cmd.get("frame_index")
            clicks = cmd.get("clicks")
            print(f"[Daemon] Real-time inference on frame {f_idx}", flush=True)
            
            if sam_model is not None and not isinstance(sam_model, torch.Tensor):
                native_w, native_h = native_sizes[0]
                scale = proxy_res / max(native_w, native_h) if (native_w > proxy_res or native_h > proxy_res) else 1.0
                
                sam_model.reset_state(inference_state)
                unique_frames = set([f for x, y, f, t, o in clicks])
                unique_objs = set([o for x, y, f, t, o in clicks])
                
                if f_idx not in unique_frames:
                    blank_mask = np.zeros((int(native_sizes[f_idx][1] * scale), int(native_sizes[f_idx][0] * scale)), dtype=np.uint8)
                    save_mask(f_idx, get_master_mask(f_idx, blank_mask))
                
                final_mask_logits = None
                for uf in unique_frames:
                    for uo in unique_objs:
                        pts_list = [[int(cx * scale), int(cy * scale)] for cx, cy, f, t, o in clicks if f == uf and o == uo]
                        if not pts_list:
                            continue
                        pts = np.array(pts_list, dtype=np.float32)
                        lbls = np.array([t for cx, cy, f, t, o in clicks if f == uf and o == uo], dtype=np.int32)
                        
                        is_cuda = (device.type == 'cuda')
                        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if is_cuda else contextlib.nullcontext()
                        
                        with ctx:
                            _, out_obj_ids, out_mask_logits = sam_model.add_new_points(
                                inference_state=inference_state,
                                frame_idx=uf,
                                obj_id=uo,
                                points=pts,
                                labels=lbls,
                                clear_old_points=True
                            )
                        if uf == f_idx:
                            final_mask_logits = out_mask_logits
                            final_obj_ids = out_obj_ids
                
                if final_mask_logits is not None:
                    proxy_mask = None
                    for i, obj_id in enumerate(final_obj_ids):
                        mask_bool = (final_mask_logits[i, 0].cpu().numpy() > 0.0)
                        if proxy_mask is None:
                            proxy_mask = np.zeros(mask_bool.shape, dtype=np.uint8)
                        proxy_mask[mask_bool] = obj_id
                    if proxy_mask is not None:
                        save_mask(f_idx, get_master_mask(f_idx, proxy_mask))
            else:
                simulate_frame(f_idx, clicks)
                
            res_queue.put({"status": "frame_done", "frame_index": f_idx})
            
        elif action == "generate_masks":
            print(f"[Daemon] Propagating sequence...", flush=True)
            if sam_model is not None and not isinstance(sam_model, torch.Tensor):
                is_cuda = (device.type == 'cuda')
                ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if is_cuda else contextlib.nullcontext()
                with ctx:
                    print("[Daemon] Forward pass...", flush=True)
                    for out_frame_idx, out_obj_ids, out_mask_logits in sam_model.propagate_in_video(inference_state, reverse=False):
                        proxy_mask = None
                        for i, obj_id in enumerate(out_obj_ids):
                            mask_bool = (out_mask_logits[i, 0].cpu().numpy() > 0.0)
                            if proxy_mask is None:
                                proxy_mask = np.zeros(mask_bool.shape, dtype=np.uint8)
                            proxy_mask[mask_bool] = obj_id
                        if proxy_mask is not None:
                            save_mask(out_frame_idx, get_master_mask(out_frame_idx, proxy_mask))
                        progress = int(((out_frame_idx + 1) / len(exr_files)) * 50)
                        res_queue.put({"status": "gen_progress", "value": progress})
                        
                    print("[Daemon] Backward pass...", flush=True)
                    for out_frame_idx, out_obj_ids, out_mask_logits in sam_model.propagate_in_video(inference_state, reverse=True):
                        proxy_mask = None
                        for i, obj_id in enumerate(out_obj_ids):
                            mask_bool = (out_mask_logits[i, 0].cpu().numpy() > 0.0)
                            if proxy_mask is None:
                                proxy_mask = np.zeros(mask_bool.shape, dtype=np.uint8)
                            proxy_mask[mask_bool] = obj_id
                        if proxy_mask is not None:
                            save_mask(out_frame_idx, get_master_mask(out_frame_idx, proxy_mask))
                        progress = 50 + int(((len(exr_files) - out_frame_idx) / len(exr_files)) * 50)
                        res_queue.put({"status": "gen_progress", "value": progress})
                        
                res_queue.put({"status": "gen_progress", "value": 100})
            else:
                clicks = cmd.get("clicks")
                for out_frame_idx in range(len(exr_files)):
                    simulate_frame(out_frame_idx, clicks)
                    progress = int(((out_frame_idx + 1) / len(exr_files)) * 100)
                    res_queue.put({"status": "gen_progress", "value": progress})
                    
            res_queue.put({"status": "gen_done"})
            
        elif action == "ingest_custom_masks":
            mask_files = cmd.get("mask_files")
            clicks = cmd.get("clicks")
            print("[Daemon] Ingesting custom user masks...", flush=True)
            
            def ingest_mask(i, um_path_in):
                um_path_out = os.path.join(user_masks_dir, f"umask_{i:05d}.png")
                um_img = cv2.imread(um_path_in, cv2.IMREAD_GRAYSCALE)
                if um_img is not None:
                    # Binarize to ensure 0 or 255
                    um_img = (um_img > 127).astype(np.uint8) * 255
                    cv2.imwrite(um_path_out, um_img)
                    
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(ingest_mask, i, path): i for i, path in enumerate(mask_files)}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    if completed % 5 == 0:
                        progress = int((completed / len(mask_files)) * 50)
                        res_queue.put({"status": "gen_progress", "value": progress})
                        
            print("[Daemon] Re-simulating sequence to bake masks...", flush=True)
            for out_frame_idx in range(len(exr_files)):
                simulate_frame(out_frame_idx, clicks)
                if out_frame_idx % 5 == 0:
                    progress = 50 + int(((out_frame_idx + 1) / len(exr_files)) * 50)
                    res_queue.put({"status": "gen_progress", "value": progress})
            
            res_queue.put({"status": "gen_done"})
            
        elif action == "exit_and_cleanup":
            print("\n[Daemon] Received exit command. Executing VRAM cleanup protocol...", flush=True)
            if sam_model is not None:
                del sam_model
            if torch:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"[Daemon] VRAM after cleanup (Final): {torch.cuda.memory_allocated() / (1024**2):.2f} MB", flush=True)
            res_queue.put({"status": "cleanup_done"})
            break



def run_tracking_worker(project_dir, exr_files, color_space, model_path, res_queue):
    import torch
    import gc
    import numpy as np
    import cv2
    import os
    import OpenImageIO as oiio
    import math
    import random
    
    try:
        from cotracker.predictor import CoTrackerOnlinePredictor
    except ImportError:
        res_queue.put({"status": "error", "message": "cotracker module not found in environment."})
        return

    print("[TrackingWorker] Initializing CoTracker3 Online...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        cotracker_model = CoTrackerOnlinePredictor(checkpoint=model_path).to(device)
    except Exception as e:
        res_queue.put({"status": "error", "message": f"Failed to load CoTracker3 model: {e}"})
        return
    
    # The online model processes chunks of `step` frames (window_len // 2)
    step = cotracker_model.step  # typically 8
    print(f"[TrackingWorker] Model loaded. Streaming window step: {step} frames.", flush=True)
    
    num_frames = len(exr_files)
    
    # 1. Determine Native & Proxy Scales
    proxies_dir = os.path.join(project_dir, 'proxies')
    first_proxy_path = os.path.join(proxies_dir, "00000.jpg")
    
    if not os.path.exists(first_proxy_path):
        res_queue.put({"status": "error", "message": "Proxy JPEGs not found. Please regenerate proxies."})
        return
        
    proxy_img0 = cv2.imread(first_proxy_path)
    proxy_h, proxy_w = proxy_img0.shape[:2]
    
    inp = oiio.ImageInput.open(exr_files[0])
    native_w, native_h = inp.spec().width, inp.spec().height
    inp.close()
    
    scale_x = native_w / proxy_w
    scale_y = native_h / proxy_h
    
    print(f"[TrackingWorker] Native: {native_w}x{native_h} | Proxy: {proxy_w}x{proxy_h} | Scale: {scale_x:.4f}x{scale_y:.4f}", flush=True)

    # 2. Sample Query Points from Exclusion Masks
    max_points = 4096
    sample_interval = 10
    
    sample_frames = list(range(0, num_frames, sample_interval))
    masks_dir = os.path.join(project_dir, 'masks')
    
    def sample_queries_from_mask(target_frame, max_samples, query_time=0):
        """Samples queries from the unmasked areas of a specific frame."""
        mask_path = os.path.join(masks_dir, f"{os.path.splitext(os.path.basename(exr_files[target_frame]))[0]}_mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
            
        if mask is not None:
            valid_y, valid_x = np.where(mask == 0)
        else:
            valid_y, valid_x = np.mgrid[0:native_h, 0:native_w]
            valid_y, valid_x = valid_y.flatten(), valid_x.flatten()
            
        queries_list = []
        if len(valid_x) > 0:
            num_samples = min(max_samples, len(valid_x))
            indices = np.random.choice(len(valid_x), num_samples, replace=False)
            for idx in indices:
                queries_list.append([query_time, valid_x[idx] / scale_x, valid_y[idx] / scale_y])
        return queries_list

    def stream_inference(queries_list, reverse=False, pass_name="Pass"):
        """Runs the online predictor over the sequence, optionally in reverse."""
        if not queries_list:
            print(f"[TrackingWorker] {pass_name}: No valid queries. Skipping.", flush=True)
            return None, None
            
        queries_tensor = torch.tensor(queries_list, dtype=torch.float32, device=device).unsqueeze(0)
        window_len = cotracker_model.model.window_len
        num_chunks = max(1, math.ceil((num_frames - window_len) / step) + 1)
        
        def read_chunk(start, length):
            chunk = torch.empty((1, length, 3, proxy_h, proxy_w), dtype=torch.float32, device='cpu')
            for j in range(length):
                rel_idx = min(start + j, num_frames - 1)
                # If reverse is True, map relative index to real index from the end of the sequence
                real_idx = (num_frames - 1 - rel_idx) if reverse else rel_idx
                proxy_path = os.path.join(proxies_dir, f"{real_idx:05d}.jpg")
                proxy_img = cv2.imread(proxy_path)
                proxy_img = cv2.cvtColor(proxy_img, cv2.COLOR_BGR2RGB)
                rgb_chw = np.transpose(proxy_img.astype(np.float32), (2, 0, 1))
                chunk[0, j] = torch.from_numpy(rgb_chw)
            return chunk.to(device, dtype=torch.float16)

        print(f"[TrackingWorker] {pass_name}: Starting inference ({num_frames} frames, reverse={reverse})...", flush=True)
        pred_tracks, pred_visibility = None, None
        
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for chunk_idx in range(num_chunks):
                    start_f = chunk_idx * step
                    chunk_tensor = read_chunk(start_f, window_len)
                    
                    if chunk_idx == 0:
                        cotracker_model(chunk_tensor, is_first_step=True, queries=queries_tensor)
                    
                    pred_tracks, pred_visibility = cotracker_model(chunk_tensor, is_first_step=False)
                    del chunk_tensor
                    
        if pred_tracks is None:
            return None, None
            
        # Trim padded frames
        pred_tracks = pred_tracks[:, :num_frames]
        pred_visibility = pred_visibility[:, :num_frames]
        
        # If reversed, flip the time dimension back to normal
        if reverse:
            pred_tracks = torch.flip(pred_tracks, dims=[1])
            pred_visibility = torch.flip(pred_visibility, dims=[1])
            
        return pred_tracks, pred_visibility

    # --- Multi-Pass Execution ---
    all_tracks = []
    all_vis = []
    
    # Distribute the 4096 budget across passes (up to 4 passes)
    points_per_pass = max_points // 4

    def add_pass_results(t_data, v_data):
        if t_data is not None:
            all_tracks.append(t_data)
            all_vis.append(v_data)

    # Pass 1: Forward Tracking from Frame 0
    q_fwd = sample_queries_from_mask(target_frame=0, max_samples=points_per_pass, query_time=0)
    t_fwd, v_fwd = stream_inference(q_fwd, reverse=False, pass_name="Pass 1 (Forward)")
    add_pass_results(t_fwd, v_fwd)
    res_queue.put({"status": "track_progress", "value": 30})

    # Pass 2: Backward Tracking from Last Frame
    q_bwd = sample_queries_from_mask(target_frame=num_frames - 1, max_samples=points_per_pass, query_time=0)
    t_bwd, v_bwd = stream_inference(q_bwd, reverse=True, pass_name="Pass 2 (Backward)")
    add_pass_results(t_bwd, v_bwd)
    res_queue.put({"status": "track_progress", "value": 50})

    # Check density and run Mid-Sequence passes if needed
    if all_vis:
        temp_vis = torch.cat(all_vis, dim=2).squeeze(0).cpu().numpy()
        active_counts = np.sum(temp_vis, axis=1)  # (T,)
        
        THRESHOLD = max_points * 0.35  # If density drops below 35% of max budget
        min_active = np.min(active_counts)
        
        if min_active < THRESHOLD:
            worst_frame = np.argmin(active_counts)
            print(f"[TrackingWorker] Density dropped to {min_active} points at frame {worst_frame}. Spawning Pass 3 & 4...", flush=True)
            
            # Pass 3: Forward from worst_frame
            q_mid_fwd = sample_queries_from_mask(target_frame=worst_frame, max_samples=points_per_pass, query_time=worst_frame)
            t_mid_fwd, v_mid_fwd = stream_inference(q_mid_fwd, reverse=False, pass_name=f"Pass 3 (Mid-Fwd F{worst_frame})")
            add_pass_results(t_mid_fwd, v_mid_fwd)
            res_queue.put({"status": "track_progress", "value": 65})
            
            # Pass 4: Backward from worst_frame
            rev_worst_frame = num_frames - 1 - worst_frame
            q_mid_bwd = sample_queries_from_mask(target_frame=worst_frame, max_samples=points_per_pass, query_time=rev_worst_frame)
            t_mid_bwd, v_mid_bwd = stream_inference(q_mid_bwd, reverse=True, pass_name=f"Pass 4 (Mid-Bwd F{worst_frame})")
            add_pass_results(t_mid_bwd, v_mid_bwd)
            
    res_queue.put({"status": "track_progress", "value": 80})

    if not all_tracks:
        res_queue.put({"status": "error", "message": "No valid tracks generated from any pass."})
        return
        
    # Merge passes along the point dimension (dim=2 for tensor shape (1, T, N, 2))
    merged_tracks = torch.cat(all_tracks, dim=2)
    merged_vis = torch.cat(all_vis, dim=2)
    
    tracks_np = merged_tracks.squeeze(0).cpu().numpy()
    vis_np = merged_vis.squeeze(0).cpu().numpy()
    
    res_queue.put({"status": "track_progress", "value": 85})
    
    # 4. Mathematically Upscale Tracks to Native Space
    print("[TrackingWorker] Upscaling tracks to native resolution...", flush=True)
    tracks_np[:, :, 0] *= scale_x
    tracks_np[:, :, 1] *= scale_y
    
    # 5. Collision Filtering Against Native-Resolution Exclusion Masks
    print("[TrackingWorker] Filtering collisions...", flush=True)
    T, N, _ = tracks_np.shape
    valid_mask = np.ones(N, dtype=bool)
    
    masks_cache = {}
    
    for t in range(T):
        mask_path = os.path.join(masks_dir, f"{os.path.splitext(os.path.basename(exr_files[t]))[0]}_mask.png")
        if os.path.exists(mask_path):
            if t not in masks_cache:
                masks_cache[t] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = masks_cache[t]
            
            for i in range(N):
                if not valid_mask[i]:
                    continue
                if vis_np[t, i]:
                    x, y = tracks_np[t, i]
                    x_idx, y_idx = int(round(x)), int(round(y))
                    
                    if 0 <= y_idx < mask_img.shape[0] and 0 <= x_idx < mask_img.shape[1]:
                        if mask_img[y_idx, x_idx] > 0:
                            valid_mask[i] = False
                    else:
                        valid_mask[i] = False
                        
    filtered_tracks = tracks_np[:, valid_mask, :]
    filtered_vis = vis_np[:, valid_mask]
    
    print(f"[TrackingWorker] Filtered {N - np.sum(valid_mask)} points. Remaining: {np.sum(valid_mask)}.", flush=True)
    
    # 6. Save & Cleanup
    out_path = os.path.join(project_dir, 'tracks.npz')
    np.savez(out_path, tracks=filtered_tracks, visibility=filtered_vis)
    print(f"[TrackingWorker] Saved tracks to {out_path}", flush=True)
    
    res_queue.put({"status": "track_progress", "value": 100})
    
    print("[TrackingWorker] Executing VRAM cleanup...", flush=True)
    del cotracker_model
    if pred_tracks is not None:
        del pred_tracks
    if pred_visibility is not None:
        del pred_visibility
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[TrackingWorker] VRAM after cleanup: {torch.cuda.memory_allocated() / (1024**2):.2f} MB", flush=True)
        
    res_queue.put({"status": "track_done"})

class ProxyGeneratorWorker(QThread):
    progress = Signal(int)
    finished = Signal(list) # Returns native sizes
    error = Signal(str)
    
    def __init__(self, exr_files, project_dir, color_space, proxy_res):
        super().__init__()
        self.exr_files = exr_files
        self.project_dir = project_dir
        self.color_space = color_space
        self.proxy_res = proxy_res # E.g., 1024, 1536, 1920
        
    def run(self):
        import concurrent.futures
        try:
            proxies_dir = os.path.join(self.project_dir, 'proxies')
            alphas_dir = os.path.join(self.project_dir, 'alphas')
            masks_dir = os.path.join(self.project_dir, 'masks')
            os.makedirs(proxies_dir, exist_ok=True)
            os.makedirs(alphas_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            native_sizes = [None] * len(self.exr_files)
            
            def process_frame(i, frame_path):
                proxy_path = os.path.join(proxies_dir, f"{i:05d}.jpg")
                alpha_path = os.path.join(alphas_dir, f"alpha_{i:05d}.png")
                
                inp = oiio.ImageInput.open(frame_path)
                spec = inp.spec()
                native_w, native_h = spec.width, spec.height
                has_alpha = spec.nchannels >= 4
                
                needs_proxy = not os.path.exists(proxy_path)
                needs_alpha = has_alpha and not os.path.exists(alpha_path)
                
                if needs_proxy or needs_alpha:
                    image_data = inp.read_image()
                    if needs_proxy:
                        rgb_array = image_data[:, :, :3] if image_data.shape[2] >= 3 else np.stack((image_data,)*3, axis=-1)
                        srgb_array = apply_color_space(rgb_array, self.color_space)
                        rgb_uint8 = np.clip(srgb_array * 255.0, 0, 255).astype(np.uint8)
                        scale = self.proxy_res / max(native_w, native_h) if (native_w > self.proxy_res or native_h > self.proxy_res) else 1.0
                        if scale != 1.0:
                            proxy_w, proxy_h = int(native_w * scale), int(native_h * scale)
                            proxy_img = cv2.resize(rgb_uint8, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
                        else:
                            proxy_img = rgb_uint8
                        cv2.imwrite(proxy_path, cv2.cvtColor(proxy_img, cv2.COLOR_RGB2BGR))
                        
                    if needs_alpha:
                        alpha_uint8 = np.clip(image_data[:, :, 3] * 255.0, 0, 255).astype(np.uint8)
                        # Invert alpha: user specifies black is masked out. We need White=Excluded for SAM 2 logic.
                        alpha_uint8 = 255 - alpha_uint8
                        cv2.imwrite(alpha_path, alpha_uint8)
                        
                if has_alpha:
                    base_name = os.path.splitext(os.path.basename(frame_path))[0]
                    mask_path_out = os.path.join(masks_dir, f"{base_name}_mask.png")
                    if not os.path.exists(mask_path_out) and os.path.exists(alpha_path):
                        import shutil
                        shutil.copy(alpha_path, mask_path_out)
                        
                inp.close()
                return i, (native_w, native_h)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_frame, i, path): i for i, path in enumerate(self.exr_files)}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    i, size = future.result()
                    native_sizes[i] = size
                    completed += 1
                    prog = int((completed / len(self.exr_files)) * 100)
                    self.progress.emit(prog)
            
            self.finished.emit(native_sizes)
        except Exception as e:
            self.error.emit(str(e))

class ImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Sequence Settings")
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Select Proxy Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItem("1024px (Fast)", 1024)
        self.combo_res.addItem("1536px (Balanced)", 1536)
        self.combo_res.addItem("1920px (High Quality)", 1920)
        self.combo_res.setCurrentIndex(1)
        layout.addWidget(self.combo_res)
        
        layout.addWidget(QLabel("Select Source Color Space:"))
        self.combo_cs = QComboBox()
        self.combo_cs.addItems([CS_LINEAR_SRGB, CS_ACESCG])
        layout.addWidget(self.combo_cs)
        
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Import")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def get_settings(self):
        return {
            "proxy_resolution": self.combo_res.currentData(),
            "color_space": self.combo_cs.currentText()
        }

# --- MAIN UI ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Camera Tracker")
        self.resize(1024, 768)
        
        self.daemon_process = None
        self.cmd_queue = None
        self.res_queue = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_queue)
        self.dl_worker = None
        
        self.exr_files = []
        self.project_dir = None
        self.project_root = None
        self.project_file = None
        self.undistorted_source_dir = None
        
        self.color_space = CS_LINEAR_SRGB
        self.proxy_res = 1536
        self.native_sizes = []

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.setup_tab = QWidget()
        self.tabs.addTab(self.setup_tab, "Setup")
        self.init_setup_tab()
        
        self.masking_tab = QWidget()
        self.tabs.addTab(self.masking_tab, "Masking")
        self.init_masking_tab()
        
        self.tracking_tab = QWidget()
        self.tabs.addTab(self.tracking_tab, "Tracking")
        self.init_tracking_tab()
        
        self.solve_tab = QWidget()
        self.tabs.addTab(self.solve_tab, "Solve")
        self.init_solve_tab()
        
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def init_setup_tab(self):
        layout = QVBoxLayout()
        
        top_layout = QHBoxLayout()
        self.btn_new_proj = QPushButton("New Project")
        self.btn_new_proj.clicked.connect(self.new_project)
        self.btn_load_proj = QPushButton("Load Project")
        self.btn_load_proj.clicked.connect(self.load_project)
        self.btn_save_proj = QPushButton("Save Project")
        self.btn_save_proj.clicked.connect(self.save_project)
        
        self.btn_import_exr = QPushButton("Import EXR Sequence")
        self.btn_import_exr.clicked.connect(self.import_exr)
        self.btn_import_exr.setEnabled(False)
        
        top_layout.addWidget(self.btn_new_proj)
        top_layout.addWidget(self.btn_load_proj)
        top_layout.addWidget(self.btn_save_proj)
        top_layout.addWidget(self.btn_import_exr)
        
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Second row of controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Proxy Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItem("1024px (Fast)", 1024)
        self.combo_res.addItem("1536px (Balanced)", 1536)
        self.combo_res.addItem("1920px (High Quality)", 1920)
        self.combo_res.setCurrentIndex(1)
        ctrl_layout.addWidget(self.combo_res)
        
        ctrl_layout.addWidget(QLabel("Color Space:"))
        self.combo_colorspace = QComboBox()
        self.combo_colorspace.addItems([CS_LINEAR_SRGB, CS_ACESCG])
        ctrl_layout.addWidget(self.combo_colorspace)
        
        self.btn_regen_proxies = QPushButton("Regenerate Proxies")
        self.btn_regen_proxies.clicked.connect(self.regenerate_proxies)
        self.btn_regen_proxies.setEnabled(False)
        ctrl_layout.addWidget(self.btn_regen_proxies)
        
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        self.lbl_file = QLabel("No sequence loaded.")
        layout.addWidget(self.lbl_file)
        
        self.setup_viewer = SequenceViewerWidget(interactive=False)
        layout.addWidget(self.setup_viewer, stretch=1)
        
        # Model Manager Group
        grp_models = QGroupBox("Model Manager")
        models_layout = QGridLayout()
        
        self.model_labels = {}
        for row, (model_name, url) in enumerate(MODELS.items()):
            models_layout.addWidget(QLabel(model_name), row, 0)
            status_lbl = QLabel()
            self.model_labels[model_name] = status_lbl
            models_layout.addWidget(status_lbl, row, 1)
            
            if url == "placeholder":
                status_lbl.setText("Placeholder (No URL)")
                status_lbl.setStyleSheet("color: orange;")
                
        self.update_model_status()
        
        self.btn_download_models = QPushButton("Download Missing Models")
        self.btn_download_models.clicked.connect(self.download_models)
        models_layout.addWidget(self.btn_download_models, len(MODELS), 0, 1, 2)
        
        self.dl_progress = QProgressBar()
        self.dl_progress.setValue(0)
        self.dl_progress.setVisible(False)
        models_layout.addWidget(self.dl_progress, len(MODELS)+1, 0, 1, 2)
        
        grp_models.setLayout(models_layout)
        layout.addWidget(grp_models)
        
        self.setup_tab.setLayout(layout)

    def update_model_status(self):
        for model_name, url in MODELS.items():
            if url == "placeholder":
                continue
            path = os.path.join(MODELS_DIR, model_name)
            if os.path.exists(path):
                self.model_labels[model_name].setText("Installed")
                self.model_labels[model_name].setStyleSheet("color: green;")
            else:
                self.model_labels[model_name].setText("Missing")
                self.model_labels[model_name].setStyleSheet("color: red;")

    def download_models(self):
        self.btn_download_models.setEnabled(False)
        self.dl_progress.setVisible(True)
        self.dl_progress.setValue(0)
        
        self.dl_worker = DownloadWorker()
        self.dl_worker.progress.connect(self.dl_progress.setValue)
        self.dl_worker.finished.connect(self.on_download_finished)
        self.dl_worker.error.connect(self.on_download_error)
        self.dl_worker.start()

    def on_download_finished(self):
        self.btn_download_models.setEnabled(True)
        self.dl_progress.setVisible(False)
        self.update_model_status()
        QMessageBox.information(self, "Download Complete", "Required models downloaded successfully.")

    def on_download_error(self, err_str):
        self.btn_download_models.setEnabled(True)
        self.dl_progress.setVisible(False)
        QMessageBox.critical(self, "Download Error", f"Failed to download models: {err_str}")

    def init_masking_tab(self):
        layout = QVBoxLayout()
        self.masking_viewer = SequenceViewerWidget(interactive=True)
        self.masking_viewer.clicksUpdated.connect(self.on_interactive_click)
        layout.addWidget(self.masking_viewer, stretch=1)
        
        ctrl_layout = QHBoxLayout()
        self.btn_gen_mask = QPushButton("Generate Masks (Sequence)")
        self.btn_gen_mask.clicked.connect(self.generate_sequence_masks)
        
        self.btn_load_user_mask = QPushButton("Load Custom Mask Sequence (JPG/PNG)")
        self.btn_load_user_mask.clicked.connect(self.load_custom_masks)
        
        self.lbl_user_mask_status = QLabel("No custom masks loaded")
        self.lbl_user_mask_status.setStyleSheet("color: gray;")
        
        ctrl_layout.addWidget(self.btn_load_user_mask)
        ctrl_layout.addWidget(self.lbl_user_mask_status)
        ctrl_layout.addWidget(self.btn_gen_mask)
        layout.addLayout(ctrl_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.masking_tab.setLayout(layout)

    def init_tracking_tab(self):
        layout = QVBoxLayout()
        
        top_layout = QHBoxLayout()
        self.btn_run_tracking = QPushButton("Run CoTracker3")
        self.btn_run_tracking.clicked.connect(self.run_tracking)
        self.btn_run_tracking.setEnabled(False)
        top_layout.addWidget(self.btn_run_tracking)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        self.tracking_viewer = SequenceViewerWidget(interactive=False, show_tracks=True)
        layout.addWidget(self.tracking_viewer, stretch=1)
        
        self.tracking_tab.setLayout(layout)

    def init_solve_tab(self):
        layout = QVBoxLayout()
        self.btn_start_solve = QPushButton("Start 3D Solver (Cleanup VRAM)")
        self.btn_start_solve.clicked.connect(self.shutdown_daemon)
        layout.addWidget(self.btn_start_solve)
        layout.addStretch()
        self.solve_tab.setLayout(layout)

    def on_color_space_changed(self):
        cs = self.color_space
        self.setup_viewer.set_color_space(cs)
        self.masking_viewer.set_color_space(cs)
        if hasattr(self, 'tracking_viewer'):
            self.tracking_viewer.set_color_space(cs)
        self.save_project()

    def new_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder to Create Project", "")
        if not dir_path: return
        
        proj_name, ok = QInputDialog.getText(self, "Project Name", "Enter a name for the new project:")
        if not ok or not proj_name.strip(): return
        proj_name = proj_name.strip()
        
        self.project_root = os.path.join(dir_path, proj_name)
        if os.path.exists(self.project_root):
            QMessageBox.warning(self, "Warning", "A folder with this name already exists here.")
            return
            
        os.makedirs(self.project_root, exist_ok=True)
        self.undistorted_source_dir = os.path.join(self.project_root, "undistorted_source")
        self.project_dir = os.path.join(self.project_root, "source_AICameraSolver_Data")
        os.makedirs(self.undistorted_source_dir, exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'proxies'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'exports'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'user_masks'), exist_ok=True)
        
        self.project_file = os.path.join(self.project_root, "project.json")
        self.exr_files = []
        self.save_project()
        
        self.btn_import_exr.setEnabled(True)
        self.lbl_file.setText(f"Project: {self.project_root} | No sequence imported.")
        QMessageBox.information(self, "Success", "Project created successfully. You can now import an EXR sequence.")

    def save_project(self):
        if not self.project_file: return
        data = {
            "exr_files": self.exr_files,
            "color_space": getattr(self, 'color_space', CS_LINEAR_SRGB),
            "proxy_res": getattr(self, 'proxy_res', 1536),
            "native_sizes": getattr(self, 'native_sizes', [])
        }
        with open(self.project_file, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Project File", "", "JSON (*.json)")
        if not file_path: return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.project_file = file_path
            self.project_root = os.path.dirname(file_path)
            self.undistorted_source_dir = os.path.join(self.project_root, "undistorted_source")
            self.project_dir = os.path.join(self.project_root, "source_AICameraSolver_Data")
            
            self.exr_files = data.get("exr_files", [])
            self.color_space = data.get("color_space", CS_LINEAR_SRGB)
            self.proxy_res = data.get("proxy_res", 1536)
            self.native_sizes = data.get("native_sizes", [])
                
            cs_idx = self.combo_colorspace.findText(self.color_space)
            if cs_idx >= 0:
                self.combo_colorspace.setCurrentIndex(cs_idx)
                
            res_idx = self.combo_res.findData(self.proxy_res)
            if res_idx >= 0:
                self.combo_res.setCurrentIndex(res_idx)
                
            self.btn_import_exr.setEnabled(True)
            self.btn_regen_proxies.setEnabled(True)
            self.lbl_file.setText(f"Project: {self.project_root} | Frames: {len(self.exr_files)}")
            
            if self.exr_files:
                cs = self.color_space
                self.setup_viewer.update_sequence(self.exr_files, cs, self.project_dir)
                self.masking_viewer.update_sequence(self.exr_files, cs, self.project_dir)
                if hasattr(self, 'tracking_viewer'):
                    self.tracking_viewer.update_sequence(self.exr_files, cs, self.project_dir)
                    if hasattr(self, 'btn_run_tracking'):
                        self.btn_run_tracking.setEnabled(True)
                um_path = os.path.join(self.project_dir, 'user_masks', "umask_00000.png")
                if os.path.exists(um_path):
                    self.lbl_user_mask_status.setText("Custom Masks Loaded (from project)")
                    self.lbl_user_mask_status.setStyleSheet("color: white;")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

    def import_exr(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Source EXR Sequence Folder", "")
        if not dir_path: return
        
        try:
            files = [f for f in os.listdir(dir_path) if f.lower().endswith('.exr')]
            if not files:
                QMessageBox.warning(self, "Warning", "No EXR files found in the selected folder.")
                return
            files.sort()
            
            if os.path.abspath(dir_path) == os.path.abspath(self.undistorted_source_dir):
                final_files = [os.path.join(dir_path, f) for f in files]
            else:
                msgBox = QMessageBox()
                msgBox.setWindowTitle("Import EXR Sequence")
                msgBox.setText("Do you want to Copy or Move the source files into the project's 'undistorted_source' folder?")
                btn_copy = msgBox.addButton("Copy", QMessageBox.ActionRole)
                btn_move = msgBox.addButton("Move", QMessageBox.ActionRole)
                btn_cancel = msgBox.addButton("Cancel", QMessageBox.RejectRole)
                msgBox.exec()
                
                if msgBox.clickedButton() == btn_cancel:
                    return
                
                action = "copy" if msgBox.clickedButton() == btn_copy else "move"
                final_files = []
                
                for f in files:
                    src = os.path.join(dir_path, f)
                    dst = os.path.join(self.undistorted_source_dir, f)
                    if action == "copy":
                        shutil.copy2(src, dst)
                    else:
                        shutil.move(src, dst)
                    final_files.append(dst)
                    
            self.exr_files = final_files
            
            # Show Dialog
            dialog = ImportDialog(self)
            if dialog.exec() == QDialog.Accepted:
                settings = dialog.get_settings()
                self.color_space = settings["color_space"]
                self.proxy_res = settings["proxy_resolution"]
                
                self.btn_import_exr.setEnabled(False)
                self.btn_regen_proxies.setEnabled(False)
                self.lbl_file.setText("Generating Proxies... Please wait.")
                
                self.proxy_worker = ProxyGeneratorWorker(self.exr_files, self.project_dir, self.color_space, self.proxy_res)
                self.proxy_worker.progress.connect(self.on_proxy_progress)
                self.proxy_worker.finished.connect(self.on_proxy_finished)
                self.proxy_worker.error.connect(self.on_proxy_error)
                self.proxy_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import EXR sequence: {str(e)}")

    def on_proxy_progress(self, val):
        self.lbl_file.setText(f"Generating Proxies... {val}%")

    def on_proxy_finished(self, native_sizes):
        self.native_sizes = native_sizes
        self.save_project()
        
        self.btn_import_exr.setEnabled(True)
        self.btn_regen_proxies.setEnabled(True)
        self.lbl_file.setText(f"Project: {self.project_root} | Frames: {len(self.exr_files)}")
        
        cs = self.color_space
        self.setup_viewer.update_sequence(self.exr_files, cs, self.project_dir)
        self.masking_viewer.update_sequence(self.exr_files, cs, self.project_dir)
        if hasattr(self, 'tracking_viewer'):
            self.tracking_viewer.update_sequence(self.exr_files, cs, self.project_dir)
            if hasattr(self, 'btn_run_tracking'):
                self.btn_run_tracking.setEnabled(True)
            
    def on_proxy_error(self, err_str):
        self.btn_import_exr.setEnabled(True)
        if hasattr(self, 'exr_files') and self.exr_files:
            self.btn_regen_proxies.setEnabled(True)
        self.lbl_file.setText("Proxy generation failed.")
        QMessageBox.critical(self, "Error", f"Failed to generate proxies: {err_str}")

    def regenerate_proxies(self):
        if not self.exr_files or not self.project_dir:
            return
            
        import shutil
        proxies_dir = os.path.join(self.project_dir, 'proxies')
        if os.path.exists(proxies_dir):
            shutil.rmtree(proxies_dir)
            
        self.color_space = self.combo_colorspace.currentText()
        self.proxy_res = self.combo_res.currentData()
        
        self.btn_import_exr.setEnabled(False)
        self.btn_regen_proxies.setEnabled(False)
        self.lbl_file.setText("Regenerating Proxies... Please wait.")
        
        self.proxy_worker = ProxyGeneratorWorker(self.exr_files, self.project_dir, self.color_space, self.proxy_res)
        self.proxy_worker.progress.connect(self.on_proxy_progress)
        self.proxy_worker.finished.connect(self.on_proxy_finished)
        self.proxy_worker.error.connect(self.on_proxy_error)
        self.proxy_worker.start()

    def run_tracking(self):
        import os
        from PySide6.QtWidgets import QMessageBox
        import multiprocessing
        
        MODELS_DIR = os.path.join(os.getcwd(), 'models')
        model_path = os.path.join(MODELS_DIR, "cotracker3", "scaled_online.pth")
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Model Not Found", "CoTracker3 model (scaled_online.pth) is not found in the model manager.\nPlease go to the Setup tab and download the missing models.")
            self.tabs.setCurrentIndex(0)
            return
            
        if not self.exr_files or not self.project_dir:
            return
            
        self.tracking_viewer.set_loading_state(True)
        self.tracking_viewer.loading_overlay.setText("Running CoTracker3 Inference...\nPlease wait.")
        self.btn_run_tracking.setEnabled(False)
        
        if self.daemon_process and self.daemon_process.is_alive():
            self.cmd_queue.put({"action": "exit_and_cleanup"})
            self.daemon_process.join(timeout=10)
            
        self.cmd_queue = multiprocessing.Queue()
        self.res_queue = multiprocessing.Queue()
        cs = getattr(self, 'color_space', CS_LINEAR_SRGB)
        
        self.daemon_process = multiprocessing.Process(
            target=run_tracking_worker, 
            args=(self.project_dir, self.exr_files, cs, model_path, self.res_queue)
        )
        self.daemon_process.start()
        self.timer.start(50)

    def on_tab_changed(self, index):
        if self.tabs.widget(index) == self.masking_tab:
            if not self.exr_files:
                QMessageBox.warning(self, "Warning", "Please load an EXR sequence first.")
                self.tabs.setCurrentWidget(self.setup_tab)
                return
            if self.daemon_process is None or not self.daemon_process.is_alive():
                self.masking_viewer.set_loading_state(True)
                self.start_daemon()
                self.masking_viewer.set_overlay_mode(True)
                
                if self.masking_viewer.click_data:
                    self.cmd_queue.put({
                        "action": "interactive_mask",
                        "frame_index": self.masking_viewer.slider.value(),
                        "clicks": self.masking_viewer.click_data
                    })
        else:
            if self.daemon_process and self.daemon_process.is_alive():
                print("Exiting SAM 2 Daemon.", flush=True)
                self.cmd_queue.put({"action": "exit_and_cleanup"})
                self.daemon_process = None

        if self.tabs.widget(index) == getattr(self, 'tracking_tab', None) or index == 2:
            self.tracking_viewer.update_sequence(self.exr_files, getattr(self, 'color_space', CS_LINEAR_SRGB), self.project_dir)
            if self.exr_files and hasattr(self, 'btn_run_tracking'):
                self.btn_run_tracking.setEnabled(True)

    def start_daemon(self):
        self.cmd_queue = multiprocessing.Queue()
        self.res_queue = multiprocessing.Queue()
        
        self.daemon_process = multiprocessing.Process(
            target=persistent_worker_daemon, 
            args=(self.cmd_queue, self.res_queue, self.exr_files, self.project_dir, self.native_sizes, self.proxy_res)
        )
        self.daemon_process.start()
        self.timer.start(50)

    def on_interactive_click(self, frame_idx):
        if self.daemon_process and self.daemon_process.is_alive():
            clicks = self.masking_viewer.click_data
            self.cmd_queue.put({
                "action": "interactive_mask",
                "frame_index": frame_idx,
                "clicks": clicks
            })

    def generate_sequence_masks(self):
        if self.daemon_process and self.daemon_process.is_alive():
            self.btn_gen_mask.setEnabled(False)
            self.progress_bar.setValue(0)
            clicks = self.masking_viewer.click_data
            self.cmd_queue.put({
                "action": "generate_masks",
                "clicks": clicks
            })

    def shutdown_daemon(self):
        if self.daemon_process and self.daemon_process.is_alive():
            self.cmd_queue.put({"action": "exit_and_cleanup"})
            # We wait for the cleanup_done signal in poll_queue before notifying UI

    def poll_queue(self):
        if self.res_queue is None: return
        while not self.res_queue.empty():
            try:
                res = self.res_queue.get_nowait()
                status = res.get("status")
                
                if status == "frame_done":
                    # Instant UI overlay refresh
                    f_idx = res.get("frame_index")
                    fetch_qpixmap.cache_clear()
                    if self.masking_viewer.slider.value() == f_idx:
                        self.masking_viewer.on_frame_changed(f_idx)
                
                elif status == "init_progress":
                    val = res.get("value")
                    self.progress_bar.setValue(val)
                    if val == 100:
                        self.progress_bar.setValue(0)
                        self.masking_viewer.set_loading_state(False)
                
                elif status == "gen_progress":
                    self.progress_bar.setValue(res.get("value"))
                    
                elif status == "gen_done":
                    self.progress_bar.setValue(100)
                    self.btn_gen_mask.setEnabled(True)
                    if hasattr(self, 'btn_load_user_mask'):
                        self.btn_load_user_mask.setEnabled(True)
                    fetch_qpixmap.cache_clear()
                    self.masking_viewer.on_frame_changed(self.masking_viewer.slider.value())
                    self.masking_viewer.set_loading_state(False)
                    QMessageBox.information(self, "Finished", "Sequence Generation Complete!")
                    
                elif status == "cleanup_done":
                    self.daemon_process.join()
                    self.daemon_process = None
                    self.timer.stop()
                    QMessageBox.information(self, "Cleanup Complete", "VRAM Cleared! Ready for CoTracker3.")
                    
                elif status == "error":
                    QMessageBox.critical(self, "Worker Error", res.get("message", "Unknown error."))
                    if hasattr(self, 'tracking_viewer'):
                        self.tracking_viewer.set_loading_state(False)
                    if hasattr(self, 'btn_run_tracking'):
                        self.btn_run_tracking.setEnabled(True)
                
                elif status == "track_progress":
                    if hasattr(self, 'tracking_viewer'):
                        self.tracking_viewer.loading_overlay.setText(f"Running CoTracker3 Inference...\nProgress: {res.get('value')}%")
                    
                elif status == "track_done":
                    if hasattr(self, 'tracking_viewer'):
                        self.tracking_viewer.set_loading_state(False)
                    if hasattr(self, 'btn_run_tracking'):
                        self.btn_run_tracking.setEnabled(True)
                    if hasattr(self, 'tracking_viewer'):
                        self.tracking_viewer.update_sequence(self.exr_files, getattr(self, 'color_space', CS_LINEAR_SRGB), self.project_dir)
                    QMessageBox.information(self, "Finished", "Tracking Complete!")
            except queue.Empty:
                pass
                
    def load_custom_masks(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Custom Mask Sequence Folder", "")
        if not dir_path: return
        
        valid_exts = {'.jpg', '.jpeg', '.png'}
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.splitext(f.lower())[1] in valid_exts]
        if not files:
            QMessageBox.warning(self, "Warning", "No JPG/PNG files found in the selected folder.")
            return
            
        files.sort()
        if len(files) != len(self.exr_files):
            QMessageBox.warning(self, "Count Mismatch", f"You selected a folder with {len(files)} mask files, but there are {len(self.exr_files)} EXR frames. They must match.")
            return
            
        user_masks_source_dir = os.path.join(self.project_root, "user_masks_source")
        os.makedirs(user_masks_source_dir, exist_ok=True)
        
        if os.path.abspath(dir_path) == os.path.abspath(user_masks_source_dir):
            final_files = [os.path.join(dir_path, f) for f in files]
        else:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Import User Masks")
            msgBox.setText("Do you want to Copy or Move the mask files into the project's 'user_masks_source' folder?")
            btn_copy = msgBox.addButton("Copy", QMessageBox.ActionRole)
            btn_move = msgBox.addButton("Move", QMessageBox.ActionRole)
            btn_cancel = msgBox.addButton("Cancel", QMessageBox.RejectRole)
            msgBox.exec()
            
            if msgBox.clickedButton() == btn_cancel:
                return
            
            action = "copy" if msgBox.clickedButton() == btn_copy else "move"
            final_files = []
            
            for f in files:
                src = os.path.join(dir_path, f)
                dst = os.path.join(user_masks_source_dir, f)
                if action == "copy":
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
                final_files.append(dst)
                
        if self.daemon_process and self.daemon_process.is_alive():
            self.lbl_user_mask_status.setText(f"Loaded: {os.path.basename(final_files[0])}...")
            self.lbl_user_mask_status.setStyleSheet("color: white;")
            self.btn_load_user_mask.setEnabled(False)
            self.btn_gen_mask.setEnabled(False)
            self.progress_bar.setValue(0)
            self.masking_viewer.set_loading_state(True)
            self.cmd_queue.put({
                "action": "ingest_custom_masks",
                "mask_files": final_files,
                "clicks": self.masking_viewer.click_data
            })

if __name__ == '__main__':
    expected_exe = os.path.abspath(os.path.join(os.getcwd(), 'python', 'python.exe'))
    if sys.executable.lower() != expected_exe.lower():
        print(f"WARNING: Executable mismatch!\nExpected: {expected_exe}\nGot: {sys.executable}")
        
    multiprocessing.set_start_method('spawn')
    multiprocessing.set_executable(sys.executable)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
