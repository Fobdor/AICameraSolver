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
import concurrent.futures
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QProgressBar, QLabel, 
                               QFileDialog, QTabWidget, QMessageBox, QGraphicsView,
                               QGraphicsScene, QSlider, QComboBox, QGroupBox, QGridLayout, QSpinBox,
                               QInputDialog, QDialog, QCheckBox, QListWidget, QDoubleSpinBox)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QWindow
from PySide6.QtCore import QTimer, Qt, QPointF, QThread, Signal
import win32gui
import open3d as o3d

# Directories
MODELS_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Models Dictionary
MODELS = {
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    "cotracker3/scaled_online.pth": "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth",
    "vggsfm/vggsfm_v2_0_0.bin": "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_0_0.bin",
    "depth_anything_v2_hf/config.json": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/config.json",
    "depth_anything_v2_hf/model.safetensors": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/model.safetensors",
    "depth_anything_v2_hf/preprocessor_config.json": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/preprocessor_config.json"
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

class AIDepthSequenceWorker(QThread):
    progress = Signal(int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, proxies_dir, depth_proxies_dir):
        super().__init__()
        self.proxies_dir = proxies_dir
        self.depth_proxies_dir = depth_proxies_dir

    def run(self):
        try:
            self.progress.emit(2)
            from transformers import pipeline
            from PIL import Image
            import torch
            import cv2
            import numpy as np
            
            self.progress.emit(5)
            model_path = os.path.join(MODELS_DIR, "depth_anything_v2_hf")
            if not os.path.exists(os.path.join(model_path, "model.safetensors")):
                raise Exception("AI Depth Model not downloaded. Please click 'Download Setup Models' in the Setup tab.")
                
            pipe = pipeline('depth-estimation', model=model_path, local_files_only=True, device=0 if torch.cuda.is_available() else -1)
            
            self.progress.emit(10)
            
            frame_files = sorted([f for f in os.listdir(self.proxies_dir) if f.endswith('.jpg') or f.endswith('.png')])
            if not frame_files:
                raise Exception("No proxy frames found to process.")
                
            total = len(frame_files)
            
            for i, f_name in enumerate(frame_files):
                in_path = os.path.join(self.proxies_dir, f_name)
                
                # Derive output filename (replace extension with .npy)
                base_name = os.path.splitext(f_name)[0]
                out_path = os.path.join(self.depth_proxies_dir, f"{base_name}.npy")
                
                # Skip if already exists? No, we should probably overwrite to be safe.
                
                img = Image.open(in_path).convert('RGB')
                w, h = img.size
                res = pipe(img)
                
                depth_tensor = res['predicted_depth']
                depth_np = depth_tensor.squeeze().cpu().numpy()
                depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
                
                np.save(out_path, depth_np)
                
                # Progress from 10% to 99%
                prog = 10 + int((i + 1) / total * 89)
                self.progress.emit(prog)
            
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"AI Depth Error: {str(e)}")

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

                if getattr(self.parent_widget, 'show_tracks', False):
                    if event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier:
                        frame_idx = getattr(self.parent_widget, 'current_frame_idx', 0)
                        self.parent_widget.pick_track(x, y, frame_idx)
                else:
                    obj_id = getattr(self.parent_widget, 'spin_obj', None)
                    obj_val = obj_id.value() if obj_id else 1
                    if event.button() == Qt.LeftButton:
                        self.parent_widget.add_click(x, y, frame_idx, 1, obj_val)
                    elif event.button() == Qt.RightButton:
                        self.parent_widget.add_click(x, y, frame_idx, 0, obj_val)
                    elif event.button() == Qt.MiddleButton or (event.modifiers() == Qt.AltModifier and event.button() == Qt.LeftButton):
                        self.parent_widget.remove_closest_click(x, y, frame_idx)
                    
        super().mousePressEvent(event)

class SequenceViewerWidget(QWidget):
    # Signals to communicate with MainWindow
    clicksUpdated = Signal(int) # emits frame_idx
    trackPicked = Signal(int) # emits track_id

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
        self.track_colors = None
        self.current_frame_idx = 0
        
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
        
        if self.show_tracks:
            self.combo_view_mode = QComboBox()
            self.combo_view_mode.addItems(["RGB", "Depth"])
            self.combo_view_mode.currentIndexChanged.connect(lambda: self.on_frame_changed(self.slider.value()))
            ctrl_layout.addWidget(self.combo_view_mode)
            
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

    def pick_track(self, x, y, frame_idx):
        if self.track_data is None or self.track_vis is None:
            return
        if frame_idx >= self.track_data.shape[0]:
            return
            
        points = self.track_data[frame_idx]
        vis = self.track_vis[frame_idx]
        
        valid_indices = np.where(vis)[0]
        if len(valid_indices) == 0:
            return
            
        valid_points = points[valid_indices]
        dists = np.linalg.norm(valid_points - np.array([x, y]), axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 20.0: # 20 pixels tolerance
            track_id = valid_indices[min_idx]
            self.trackPicked.emit(int(track_id))

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
            
        self.current_frame_idx = index
            
        frame_path = self.exr_files[index]
        
        mask_path = None
        mask_mtime = 0
        if self.show_overlay and self.project_dir:
            base_name = os.path.splitext(os.path.basename(frame_path))[0]
            mask_path = os.path.join(self.project_dir, 'masks', f"{base_name}_mask.png")
            if os.path.exists(mask_path):
                mask_mtime = os.path.getmtime(mask_path)
            
        view_mode = "RGB"
        if hasattr(self, 'combo_view_mode'):
            view_mode = self.combo_view_mode.currentText()
            
        base_pixmap = QPixmap()
        if self.project_dir:
            if view_mode == "Depth":
                depth_path = os.path.join(self.project_dir, 'depth_proxies', f"{index:05d}.npy")
                if os.path.exists(depth_path):
                    import numpy as np
                    import cv2
                    depth = np.load(depth_path)
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                    h, w, c = depth_color.shape
                    qImg = QImage(depth_color.data, w, h, 3 * w, QImage.Format_BGR888)
                    base_pixmap = QPixmap.fromImage(qImg)
            else:
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
                        
                        if self.track_colors is not None and i < len(self.track_colors):
                            c = self.track_colors[i]
                            painter.setPen(QPen(c, 2))
                            painter.setBrush(c)
                            
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
    if 'merged_tracks' in locals():
        del merged_tracks
    if 'merged_vis' in locals():
        del merged_vis
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[TrackingWorker] VRAM after cleanup: {torch.cuda.memory_allocated() / (1024**2):.2f} MB", flush=True)
        
    res_queue.put({"status": "track_done"})

def run_vggsfm_worker(project_dir, proxies_dir, tracks_npz, camera_data, model_path, res_queue):
    """
    Isolated worker for running 3D Bundle Adjustment and Triangulation using PyCOLMAP
    on the pre-computed tracks from Phase 3.
    """
    try:
        import os
        import tempfile
        import numpy as np
        import pycolmap
        
        res_queue.put({"status": "solve_progress", "value": 5, "message": "Loading tracks and configuring solver..."})
        print("[VGGSfMWorker] Loading tracks for PyCOLMAP...", flush=True)

        if not os.path.exists(tracks_npz):
            raise FileNotFoundError(f"Tracks file not found at {tracks_npz}")

        # Load Tracks
        data = np.load(tracks_npz)
        tracks_2d = data['tracks']  # (S, N, 2)
        visibility = data['visibility']  # (S, N)
        S, N, _ = tracks_2d.shape

        # Retrieve Camera Properties
        eff_width = camera_data.get('effective_sensor_width', 36.0)
        eff_height = camera_data.get('effective_sensor_height', 24.0)
        focal_length_in = camera_data.get('focal_length', 'auto')
        auto_estimate = camera_data.get('auto_estimate_focal', True)
        
        # Original plate resolution
        plate_w = camera_data.get('plate_width', 1920)
        plate_h = camera_data.get('plate_height', 1080)
        
        # --- TRACK RELIABILITY SCORING & SPATIAL BINNING ---
        res_queue.put({"status": "solve_progress", "value": 10, "message": "Scoring and Binning tracks..."})
        
        # 1. Lifespan Calculation
        lifespan = visibility.sum(axis=0)  # (N,)
        
        # 2. Motion Jitter (Acceleration Penalty)
        if S >= 3:
            accel = np.diff(tracks_2d, n=2, axis=0)
            accel_mag = np.linalg.norm(accel, axis=-1)  # (S-2, N)
            valid_accel = visibility[:-2] & visibility[1:-1] & visibility[2:]
            accel_mag[~valid_accel] = 0.0
            max_accel = np.max(accel_mag, axis=0)  # (N,)
        else:
            max_accel = np.zeros(N)
        
        # 3. Reliability Score
        lifespan_weight = 1.0
        jitter_weight = 5.0
        score = (lifespan_weight * lifespan) - (jitter_weight * max_accel)
        
        # 4. Grid-Based Spatial Binning
        first_visible_idx = np.argmax(visibility, axis=0) # (N,)
        first_x = tracks_2d[first_visible_idx, np.arange(N), 0]
        first_y = tracks_2d[first_visible_idx, np.arange(N), 1]
        
        grid_size_x, grid_size_y = 24, 24
        cell_w = plate_w / grid_size_x
        cell_h = plate_h / grid_size_y
        
        grid_x = np.clip((first_x / cell_w).astype(int), 0, grid_size_x - 1)
        grid_y = np.clip((first_y / cell_h).astype(int), 0, grid_size_y - 1)
        
        selected_indices = []
        points_per_cell = 3 # Top 3 tracks per cell
        
        for gy in range(grid_size_y):
            for gx in range(grid_size_x):
                mask = (grid_x == gx) & (grid_y == gy)
                cell_indices = np.where(mask)[0]
                if len(cell_indices) > 0:
                    cell_scores = score[cell_indices]
                    sorted_idx = np.argsort(cell_scores)[::-1]
                    best_tracks = cell_indices[sorted_idx[:points_per_cell]]
                    selected_indices.extend(best_tracks)
                    
        selected_indices = np.array(selected_indices, dtype=int)
        
        # 5. Final Decimation
        if len(selected_indices) > 0:
            tracks_2d = tracks_2d[:, selected_indices, :]
            visibility = visibility[:, selected_indices]
            N = tracks_2d.shape[1]
            print(f"[VGGSfMWorker] Decimated tracks to {N} using spatial binning.", flush=True)
        # ---------------------------------------------------
        
        # Calculate focal length in pixels
        if focal_length_in == 'auto' or auto_estimate:
            focal_length = 50.0  # Safe default to start BA
        else:
            focal_length = float(focal_length_in)
            
        focal_px_x = (focal_length / eff_width) * plate_w
        focal_px_y = (focal_length / eff_height) * plate_h
        focal_px = (focal_px_x + focal_px_y) / 2.0

        res_queue.put({"status": "solve_progress", "value": 15, "message": "Building PyCOLMAP Database from tracks..."})

        # 2. Camera Initialization & Database Setup
        temp_dir = tempfile.mkdtemp(prefix="colmap_")
        db_path = os.path.join(temp_dir, "database.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            
        db = pycolmap.Database.open(db_path)

        camera = pycolmap.Camera(
            model="PINHOLE",
            width=int(plate_w),
            height=int(plate_h),
            params=[focal_px, focal_px, plate_w / 2.0, plate_h / 2.0]
        )
        camera.has_prior_focal_length = not auto_estimate
        cam_id = db.write_camera(camera)

        keypoint_maps = [] # Maps (img_id) -> {track_idx: kp_idx}
        
        for i in range(S):
            img_name = f"frame_{i:04d}.jpg"
            img_id = db.write_image(pycolmap.Image(name=img_name, camera_id=cam_id))
            
            vis_i = visibility[i] > 0
            valid_track_indices = np.where(vis_i)[0]
            pts = tracks_2d[i, vis_i]
            
            db.write_keypoints(img_id, pts.astype(np.float64))
            
            kp_map = {track_idx: kp_idx for kp_idx, track_idx in enumerate(valid_track_indices)}
            keypoint_maps.append((img_id, kp_map))

        res_queue.put({"status": "solve_progress", "value": 30, "message": "Generating exhaustive two-view matches..."})

        for i in range(S):
            img_id_i, map_i = keypoint_maps[i]
            for j in range(i + 1, S):
                img_id_j, map_j = keypoint_maps[j]
                
                common_tracks = set(map_i.keys()).intersection(set(map_j.keys()))
                if len(common_tracks) < 15:
                    continue
                    
                matches = []
                for t in common_tracks:
                    matches.append([map_i[t], map_j[t]])
                
                matches = np.array(matches, dtype=np.uint32)
                db.write_matches(img_id_i, img_id_j, matches)

        db.close()

        res_queue.put({"status": "solve_progress", "value": 45, "message": "Verifying two-view geometry (RANSAC)..."})

        pycolmap.geometric_verification(db_path)

        res_queue.put({"status": "solve_progress", "value": 60, "message": "Executing Incremental Bundle Adjustment... (This may take several minutes depending on sequence length)"})

        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_model_size = 3
        mapper_options.ba_refine_focal_length = auto_estimate
        mapper_options.ba_refine_extra_params = auto_estimate
        mapper_options.ba_local_max_num_iterations = 25
        mapper_options.ba_global_max_num_iterations = 50
        mapper_options.ba_use_gpu = True
        
        # Relax initialization constraints for AI tracks with potentially low parallax
        mapper_options.mapper.init_min_tri_angle = 0.1
        mapper_options.mapper.init_min_num_inliers = 15
        mapper_options.mapper.init_max_forward_motion = 0.99
        mapper_options.mapper.abs_pose_min_num_inliers = 15
        mapper_options.mapper.abs_pose_min_inlier_ratio = 0.1
        mapper_options.mapper.filter_min_tri_angle = 0.1
        mapper_options.triangulation.min_angle = 0.1

        recs = pycolmap.incremental_mapping(
            database_path=db_path, 
            image_path=proxies_dir,
            output_path=temp_dir, 
            options=mapper_options
        )

        if not recs or len(recs) == 0:
            raise RuntimeError("PyCOLMAP failed to reconstruct the scene. Not enough overlapping tracks.")

        best_rec = None
        max_imgs = 0
        for r_idx, r in recs.items():
            if r.num_images() > max_imgs:
                max_imgs = r.num_images()
                best_rec = r
                
        rec = best_rec
        res_queue.put({"status": "solve_progress", "value": 85, "message": "Reconstruction complete. Serializing data..."})

        out_pts = np.zeros((N, 3), dtype=np.float32)
        out_errs = np.zeros(N, dtype=np.float32)
        out_mask = np.zeros(N, dtype=bool)

        for img_id, image in rec.images.items():
            _, map_i = keypoint_maps[img_id - 1]
            rev_map_i = {v: k for k, v in map_i.items()}
            
            for kp_idx, p2d in enumerate(image.points2D):
                if p2d.has_point3D():
                    p3d_id = p2d.point3D_id
                    track_idx = rev_map_i.get(kp_idx)
                    if track_idx is not None and not out_mask[track_idx]:
                        p3d = rec.points3D[p3d_id]
                        out_pts[track_idx] = p3d.xyz
                        out_errs[track_idx] = p3d.error
                        out_mask[track_idx] = True

        out_cameras_rot = np.zeros((S, 3, 3), dtype=np.float32)
        out_cameras_trans = np.zeros((S, 3), dtype=np.float32)

        for i in range(S):
            img_id = i + 1 
            if img_id in rec.images:
                img = rec.images[img_id]
                out_cameras_rot[i] = img.cam_from_world().rotation.matrix()
                out_cameras_trans[i] = img.cam_from_world().translation
            else:
                out_cameras_rot[i] = np.eye(3)
                out_cameras_trans[i] = np.zeros(3)

        out_data_path = os.path.join(project_dir, 'solve_data.npz')
        np.savez(
            out_data_path,
            points_3d=out_pts,
            points_error=out_errs,
            points_mask=out_mask,
            cameras_rot=out_cameras_rot,
            cameras_trans=out_cameras_trans,
            focal_px=rec.cameras[cam_id].params[0],
            tracks_2d=tracks_2d,
            visibility=visibility
        )

        res_queue.put({"status": "solve_progress", "value": 100, "message": "Solver Complete!"})
        res_queue.put({"status": "solve_done"})

    except Exception as e:
        import traceback
        err = str(e) + "\\n" + traceback.format_exc()
        res_queue.put({"status": "error", "message": err})

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
                
                # Use display window (full_width/full_height) to include overscan padding
                native_w = spec.full_width if spec.full_width > 0 else spec.width
                native_h = spec.full_height if spec.full_height > 0 else spec.height
                
                # Check for alpha channel explicitly
                alpha_idx = -1
                for idx, name in enumerate(spec.channelnames):
                    if name.lower() in ['a', 'alpha', 'rgba.a']:
                        alpha_idx = idx
                        break
                
                has_alpha = alpha_idx != -1
                
                needs_proxy = not os.path.exists(proxy_path)
                # Force rebuild alpha if we are rebuilding the proxy to keep them perfectly in sync
                needs_alpha = has_alpha and (not os.path.exists(alpha_path) or needs_proxy)
                
                if needs_proxy or needs_alpha:
                    data_window_img = inp.read_image()
                    
                    offset_x = spec.x - spec.full_x
                    offset_y = spec.y - spec.full_y
                    
                    if native_w != spec.width or native_h != spec.height or offset_x != 0 or offset_y != 0:
                        image_data = np.zeros((native_h, native_w, data_window_img.shape[2]), dtype=data_window_img.dtype)
                        y_start = max(0, offset_y)
                        y_end = min(native_h, offset_y + spec.height)
                        x_start = max(0, offset_x)
                        x_end = min(native_w, offset_x + spec.width)
                        
                        data_y_start = max(0, -offset_y)
                        data_y_end = data_y_start + (y_end - y_start)
                        data_x_start = max(0, -offset_x)
                        data_x_end = data_x_start + (x_end - x_start)
                        
                        if y_start < y_end and x_start < x_end:
                            image_data[y_start:y_end, x_start:x_end] = data_window_img[data_y_start:data_y_end, data_x_start:data_x_end]
                    else:
                        image_data = data_window_img
                        
                    if needs_proxy:
                        # Extract RGB channels correctly
                        r_idx, g_idx, b_idx = 0, 1, 2
                        for idx, name in enumerate(spec.channelnames):
                            if name.lower() in ['r', 'rgba.r']: r_idx = idx
                            elif name.lower() in ['g', 'rgba.g']: g_idx = idx
                            elif name.lower() in ['b', 'rgba.b']: b_idx = idx
                            
                        if image_data.shape[2] >= 3:
                            rgb_array = image_data[:, :, [r_idx, g_idx, b_idx]]
                        else:
                            rgb_array = np.stack((image_data[:,:,0],)*3, axis=-1)
                            
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
                        alpha_uint8 = np.clip(image_data[:, :, alpha_idx] * 255.0, 0, 255).astype(np.uint8)
                        # Invert alpha: user specifies black is masked out. We need White=Excluded for SAM 2 logic.
                        alpha_uint8 = 255 - alpha_uint8
                        cv2.imwrite(alpha_path, alpha_uint8)
                        
                if has_alpha:
                    base_name = os.path.splitext(os.path.basename(frame_path))[0]
                    mask_path_out = os.path.join(masks_dir, f"{base_name}_mask.png")
                    # Force overwrite the mask_path_out if we just rebuilt the alpha
                    if needs_alpha or not os.path.exists(mask_path_out):
                        if os.path.exists(alpha_path):
                            import shutil
                            shutil.copy(alpha_path, mask_path_out)
                        
                inp.close()
                return i, (native_w, native_h)

            # Intelligently scale workers based on System RAM and CPU threads
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            
            total_ram_gb = stat.ullTotalPhys / (1024**3)
            
            # Budget ~1.5 GB of system RAM per concurrent worker to prevent OOM crashes on 4K EXRs
            max_workers_by_ram = max(1, int(total_ram_gb / 1.5))
            cpu_threads = os.cpu_count() or 4
            
            safe_workers = max(1, min(cpu_threads, max_workers_by_ram, 64))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
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

class SolveViewport(QWidget):
    activeCameraChanged = Signal(int)
    reqClearSelection = Signal()
    reqAddConstraint = Signal(str, float)
    reqDelConstraint = Signal(int)
    reqApplyOrientation = Signal()
    reqSelectConstraintIdx = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("O3D_Embedded", width=800, height=600, visible=False)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 2.0
        
        # Poll briefly to ensure window is created by OS
        for _ in range(10):
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)
            
        hwnd = win32gui.FindWindow(None, "O3D_Embedded")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        if hwnd:
            window = QWindow.fromWinId(hwnd)
            self.widget3d = QWidget.createWindowContainer(window)
            self.layout.addWidget(self.widget3d, stretch=1)
        else:
            self.layout.addWidget(QLabel("Failed to embed Open3D viewport."))
            
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        
        self.lbl_error = QLabel("Max Reprojection Error: 2.0px")
        self.slider_error = QSlider(Qt.Horizontal)
        self.slider_error.setMinimum(1)
        self.slider_error.setMaximum(100) # 0.1 to 10.0
        self.slider_error.setValue(20)
        self.slider_error.valueChanged.connect(self.update_error_threshold)
        
        self.lbl_frame = QLabel("Current Camera: 0")
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.valueChanged.connect(self.update_active_camera)
        
        self.control_layout.addWidget(self.lbl_error)
        self.control_layout.addWidget(self.slider_error)
        self.control_layout.addWidget(QLabel("")) # Spacer
        self.control_layout.addWidget(self.lbl_frame)
        self.control_layout.addWidget(self.slider_frame)
        
        # --- NEW: Orientation & Scale UI ---
        self.control_layout.addWidget(QLabel("")) # Spacer
        self.control_layout.addWidget(QLabel("Selected Points:"))
        self.list_selection = QListWidget()
        self.list_selection.setMaximumHeight(80)
        self.control_layout.addWidget(self.list_selection)
        
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_clear_sel.clicked.connect(self.reqClearSelection.emit)
        self.control_layout.addWidget(self.btn_clear_sel)
        
        self.control_layout.addWidget(QLabel("")) # Spacer
        self.control_layout.addWidget(QLabel("Orientation Tools:"))
        
        self.list_orientation = QListWidget()
        self.list_orientation.setMaximumHeight(150)
        self.list_orientation.currentRowChanged.connect(self.reqSelectConstraintIdx.emit)
        self.control_layout.addWidget(self.list_orientation)
        
        row1 = QHBoxLayout()
        self.combo_orient_type = QComboBox()
        self.combo_orient_type.addItems([
            "Origin (1 pt)", 
            "Scale (2 pts)", 
            "Ground Plane (3+ pts)", 
            "XY Plane (2+ pts)", 
            "YZ Plane (2+ pts)", 
            "X Line (2 pts)", 
            "Y Line (2 pts)", 
            "Z Line (2 pts)"
        ])
        row1.addWidget(self.combo_orient_type)
        
        self.spin_orient_scale = QDoubleSpinBox()
        self.spin_orient_scale.setRange(0.001, 9999.0)
        self.spin_orient_scale.setValue(1.0)
        self.spin_orient_scale.setToolTip("Target Distance for Scale constraint (meters)")
        row1.addWidget(self.spin_orient_scale)
        
        self.btn_add_orient = QPushButton("Add")
        self.btn_add_orient.clicked.connect(lambda: self.reqAddConstraint.emit(self.combo_orient_type.currentText(), self.spin_orient_scale.value()))
        row1.addWidget(self.btn_add_orient)
        self.control_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        self.btn_del_orient = QPushButton("Delete Selected")
        self.btn_del_orient.clicked.connect(lambda: self.reqDelConstraint.emit(self.list_orientation.currentRow()))
        row2.addWidget(self.btn_del_orient)
        
        self.btn_apply_orient = QPushButton("Apply Orientation")
        self.btn_apply_orient.setStyleSheet("font-weight: bold; color: #55ff55;")
        self.btn_apply_orient.clicked.connect(self.reqApplyOrientation.emit)
        row2.addWidget(self.btn_apply_orient)
        self.control_layout.addLayout(row2)
        
        persp_layout = QHBoxLayout()
        persp_layout.addWidget(QLabel("Perspective:"))
        self.combo_perspective = QComboBox()
        self.combo_perspective.addItems(["Free Camera", "Scene Camera"])
        self.combo_perspective.currentIndexChanged.connect(self.update_perspective)
        persp_layout.addWidget(self.combo_perspective)
        self.control_layout.addLayout(persp_layout)
        
        origin_scale_layout = QHBoxLayout()
        origin_scale_layout.addWidget(QLabel("Origin Axis Size:"))
        self.slider_origin_scale = QSlider(Qt.Horizontal)
        self.slider_origin_scale.setRange(1, 200)
        self.slider_origin_scale.setValue(50)
        self.slider_origin_scale.valueChanged.connect(self.update_origin_scale)
        origin_scale_layout.addWidget(self.slider_origin_scale)
        self.control_layout.addLayout(origin_scale_layout)
        
        cam_scale_layout = QHBoxLayout()
        cam_scale_layout.addWidget(QLabel("Cam Icon Size:"))
        self.slider_cam_scale = QSlider(Qt.Horizontal)
        self.slider_cam_scale.setRange(1, 100)
        self.slider_cam_scale.setValue(90)
        self.slider_cam_scale.valueChanged.connect(self.update_camera_scale)
        cam_scale_layout.addWidget(self.slider_cam_scale)
        self.control_layout.addLayout(cam_scale_layout)
        
        point_scale_layout = QHBoxLayout()
        point_scale_layout.addWidget(QLabel("Highlight Size:"))
        self.slider_point_scale = QSlider(Qt.Horizontal)
        self.slider_point_scale.setRange(1, 200)
        self.slider_point_scale.setValue(90) # Match original 9.0 scale
        self.slider_point_scale.valueChanged.connect(self.update_error_threshold)
        point_scale_layout.addWidget(self.slider_point_scale)
        self.control_layout.addLayout(point_scale_layout)
        
        self.control_layout.addStretch()
        
        self.layout.addWidget(self.control_panel)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick_o3d)
        self.timer.start(16)
        
        self.pcd = None
        self.full_points = None
        self.full_errors = None
        self.full_colors = None
        self.camera_linesets = []
        
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        self.origin_scale = 5.0
        self.grid = self.create_grid(size=100, divisions=100)
        self.vis.add_geometry(self.origin_frame)
        self.vis.add_geometry(self.grid)
        
        self.highlight_mesh = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.highlight_mesh)
        
        def reset_to_origin(vis):
            ctr = vis.get_view_control()
            ctr.set_lookat([0.0, 0.0, 0.0])
            return False
        self.vis.register_key_callback(ord('F'), reset_to_origin)
        
    def create_grid(self, size=100, divisions=100):
        lines = []
        points = []
        step = size / divisions
        half = size / 2.0
        
        idx = 0
        for i in range(divisions + 1):
            val = -half + i * step
            points.extend([[val, 0, -half], [val, 0, half]])
            lines.append([idx, idx + 1])
            idx += 2
            
            points.extend([[-half, 0, val], [half, 0, val]])
            lines.append([idx, idx + 1])
            idx += 2
            
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3] for _ in range(len(lines))])
        return grid
        
    def _tick_o3d(self):
        if hasattr(self, 'combo_perspective') and self.combo_perspective.currentText() == "Scene Camera":
            self.sync_perspective_to_camera()
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def load_solve_data(self, npz_path, camera_setup_data, reset_view=True):
        data = np.load(npz_path)
        self.full_points = data['points_3d']
        self.full_errors = data['points_error']
        points_mask = data['points_mask']
        self.cameras_rot = data['cameras_rot']
        self.cameras_trans = data['cameras_trans']
        self.focal_px = data['focal_px']
        self.camera_setup_data = camera_setup_data
        
        # Build mapping: original track index -> compacted index
        valid_original_indices = np.where(points_mask)[0]
        self.track_to_compact = {int(orig): compact for compact, orig in enumerate(valid_original_indices)}
        self.compact_to_track = {compact: int(orig) for compact, orig in enumerate(valid_original_indices)}
        
        self.full_points = self.full_points[points_mask]
        self.full_errors = self.full_errors[points_mask]
        
        t = np.clip((self.full_errors - 0.5) / 1.5, 0.0, 1.0)
        self.full_colors = np.zeros_like(self.full_points)
        self.full_colors[:, 0] = t # R
        self.full_colors[:, 1] = 1.0 - t # G
        
        view_ctl = None
        cam_params = None
        if not reset_view and self.pcd is not None:
            view_ctl = self.vis.get_view_control()
            cam_params = view_ctl.convert_to_pinhole_camera_parameters()
            
        if self.pcd is not None:
            self.vis.remove_geometry(self.pcd, reset_bounding_box=reset_view)
        for ls in self.camera_linesets:
            self.vis.remove_geometry(ls, reset_bounding_box=reset_view)
        self.camera_linesets.clear()
        
        if hasattr(self, 'path_ls') and self.path_ls is not None:
            self.vis.remove_geometry(self.path_ls, reset_bounding_box=reset_view)
            self.path_ls = None
        
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd, reset_bounding_box=reset_view)
        self.update_error_threshold()
        
        plate_w = camera_setup_data.get('plate_width', 1920)
        plate_h = camera_setup_data.get('plate_height', 1080)
        cx, cy = plate_w / 2.0, plate_h / 2.0
        
        if len(self.full_points) > 0:
            min_bound = np.min(self.full_points, axis=0)
            max_bound = np.max(self.full_points, axis=0)
            max_extent = np.max(max_bound - min_bound)
        else:
            max_extent = 10.0
            
        z = max_extent * 0.01 * (self.slider_cam_scale.value() / 10.0)
        x_max = (plate_w - cx) / self.focal_px * z
        x_min = (0 - cx) / self.focal_px * z
        y_max = (plate_h - cy) / self.focal_px * z
        y_min = (0 - cy) / self.focal_px * z
        
        frustum_pts = np.array([[0, 0, 0], [x_min, y_min, z], [x_max, y_min, z], [x_max, y_max, z], [x_min, y_max, z]])
        frustum_lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        colors = [[0.2, 0.5, 1.0] for _ in range(len(frustum_lines))]
        
        path_pts = []
        for i in range(len(self.cameras_rot)):
            R = self.cameras_rot[i]
            T = self.cameras_trans[i]
            if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)):
                continue
                
            R_inv = R.T
            cam_center = -R_inv @ T
            path_pts.append(cam_center)
            
            world_pts = (R_inv @ frustum_pts.T).T + cam_center
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(world_pts)
            ls.lines = o3d.utility.Vector2iVector(frustum_lines)
            ls.colors = o3d.utility.Vector3dVector(colors)
            self.camera_linesets.append(ls)
            self.vis.add_geometry(ls, reset_bounding_box=reset_view)
            
        self.slider_frame.setMaximum(max(0, len(self.camera_linesets) - 1))
        self.slider_frame.setValue(0)
        self.update_active_camera()
            
        if len(path_pts) > 1:
            path_lines = [[j, j+1] for j in range(len(path_pts)-1)]
            self.path_ls = o3d.geometry.LineSet()
            self.path_ls.points = o3d.utility.Vector3dVector(np.array(path_pts))
            self.path_ls.lines = o3d.utility.Vector2iVector(path_lines)
            self.path_ls.colors = o3d.utility.Vector3dVector([[1,1,0] for _ in range(len(path_lines))])
            self.vis.add_geometry(self.path_ls, reset_bounding_box=reset_view)
            
        if reset_view:
            self.vis.reset_view_point(True)
        elif view_ctl is not None and cam_params is not None:
            view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            
        self.update_perspective()
        
    def update_active_camera(self):
        if not self.camera_linesets: return
        idx = self.slider_frame.value()
        self.lbl_frame.setText(f"Current Camera: {idx}")
        for i, ls in enumerate(self.camera_linesets):
            if i == idx:
                ls.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in range(8)])
            else:
                ls.colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2] for _ in range(8)])
            self.vis.update_geometry(ls)
            
        if hasattr(self, 'combo_perspective') and self.combo_perspective.currentText() == "Scene Camera":
            self.sync_perspective_to_camera(idx)
            
        self.activeCameraChanged.emit(idx)
        
    def update_perspective(self):
        is_scene_cam = self.combo_perspective.currentText() == "Scene Camera"
        if is_scene_cam:
            self.sync_perspective_to_camera()
            
        for ls in self.camera_linesets:
            if is_scene_cam:
                self.vis.remove_geometry(ls, reset_bounding_box=False)
            else:
                self.vis.add_geometry(ls, reset_bounding_box=False)
                
        if hasattr(self, 'path_ls') and self.path_ls is not None:
            if is_scene_cam:
                self.vis.remove_geometry(self.path_ls, reset_bounding_box=False)
            else:
                self.vis.add_geometry(self.path_ls, reset_bounding_box=False)
                
        if not is_scene_cam:
            self.update_active_camera()
            
    def sync_perspective_to_camera(self, idx=None):
        if idx is None:
            idx = self.slider_frame.value()
        if not hasattr(self, 'cameras_rot') or idx >= len(self.cameras_rot): return
        if not hasattr(self, 'camera_setup_data'): return
        
        R = self.cameras_rot[idx]
        T = self.cameras_trans[idx]
        
        if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)):
            return
            
        plate_w = self.camera_setup_data.get('plate_width', 1920)
        plate_h = self.camera_setup_data.get('plate_height', 1080)
        cx, cy = plate_w / 2.0, plate_h / 2.0
        fx = fy = self.focal_px
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(plate_w), height=int(plate_h), fx=fx, fy=fy, cx=cx, cy=cy
        )
        
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = intrinsic
        
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T
        cam_params.extrinsic = extrinsic
        
        view_ctl = self.vis.get_view_control()
        view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    def update_origin_scale(self):
        self.origin_scale = self.slider_origin_scale.value() / 10.0
        self.vis.remove_geometry(self.origin_frame, reset_bounding_box=False)
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.origin_scale)
        self.vis.add_geometry(self.origin_frame, reset_bounding_box=False)
        
    def update_error_threshold(self):
        if self.pcd is None or self.full_points is None: return
        threshold = self.slider_error.value() / 10.0
        
        valid_idx = np.where(self.full_errors <= threshold)[0]
        pts = self.full_points[valid_idx]
        cols = self.full_colors[valid_idx].copy()
        
        if hasattr(self, 'selected_tracks') and self.selected_tracks:
            for t_id in self.selected_tracks:
                compact_id = self.track_to_compact.get(t_id)
                if compact_id is not None:
                    idx_in_valid = np.where(valid_idx == compact_id)[0]
                    if len(idx_in_valid) > 0:
                        cols[idx_in_valid[0]] = [1.0, 1.0, 0.0] # Yellow
                    
        self.vis.remove_geometry(self.pcd, reset_bounding_box=False)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(cols)
        self.vis.add_geometry(self.pcd, reset_bounding_box=False)
        self.lbl_error.setText(f"Max Reprojection Error: {threshold:.1f}px")
        
        # Build giant physical spheres for selected points so they are obvious
        self.vis.remove_geometry(self.highlight_mesh, reset_bounding_box=False)
        self.highlight_mesh = o3d.geometry.TriangleMesh()
        
        if hasattr(self, 'selected_tracks') and self.selected_tracks:
            # Calculate dynamic bounding box extent so the sizes scale with any scene size
            if len(self.full_points) > 0:
                min_bound = np.min(self.full_points, axis=0)
                max_bound = np.max(self.full_points, axis=0)
                max_extent = np.max(max_bound - min_bound)
            else:
                max_extent = 10.0
                
            point_scale = self.slider_point_scale.value() / 10.0 if hasattr(self, 'slider_point_scale') else 9.0
            radius = max_extent * 0.005 * point_scale
            
            for t_id in self.selected_tracks:
                compact_id = self.track_to_compact.get(t_id)
                if compact_id is not None and compact_id < len(self.full_points):
                    pt = self.full_points[compact_id]
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                    sphere.translate(pt)
                    self.highlight_mesh += sphere
                    
            self.highlight_mesh.compute_vertex_normals()
            self.highlight_mesh.paint_uniform_color([1.0, 0.0, 0.0]) # Red for spheres so they stand out against planes
            
            # Draw lines or planes based on constraint
            if hasattr(self, 'active_constraint') and self.active_constraint:
                ctype = self.active_constraint['type']
                pts = []
                for t_id in self.selected_tracks:
                    compact_id = self.track_to_compact.get(t_id)
                    if compact_id is not None and compact_id < len(self.full_points):
                        pts.append(self.full_points[compact_id])
                pts = np.array(pts)
                
                if "Line" in ctype and len(pts) == 2:
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius*0.5, height=np.linalg.norm(pts[0]-pts[1]))
                    cylinder.compute_vertex_normals()
                    cylinder.paint_uniform_color([1.0, 1.0, 0.0])
                    # Rotate cylinder to align with line
                    z = np.array([0.0, 0.0, 1.0])
                    target = pts[1] - pts[0]
                    target /= np.linalg.norm(target)
                    v = np.cross(z, target)
                    c = np.dot(z, target)
                    s = np.linalg.norm(v)
                    if s < 1e-6:
                        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
                    else:
                        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
                    cylinder.rotate(R, center=[0,0,0])
                    cylinder.translate((pts[0] + pts[1]) / 2.0)
                    self.highlight_mesh += cylinder
                    
                elif "Plane" in ctype and len(pts) >= 3:
                    centroid = np.mean(pts, axis=0)
                    centered = pts - centroid
                    u, s_svd, vh = np.linalg.svd(centered)
                    normal = vh[-1, :]
                    normal = normal / np.linalg.norm(normal)
                    
                    cam_scale = self.slider_cam_scale.value() / 10.0 if hasattr(self, 'slider_cam_scale') else 1.0
                    plane_size = max_extent * 0.5 * cam_scale
                    box = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=radius*0.2)
                    box.translate([-plane_size/2, -plane_size/2, 0])
                    
                    z = np.array([0.0, 0.0, 1.0])
                    target = normal
                    v = np.cross(z, target)
                    c = np.dot(z, target)
                    s_len = np.linalg.norm(v)
                    if s_len < 1e-6:
                        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
                    else:
                        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s_len**2))
                        
                    box.rotate(R, center=[0,0,0])
                    box.translate(centroid)
                    box.compute_vertex_normals()
                    box.paint_uniform_color([1.0, 1.0, 0.0]) # Yellow plane
                    self.highlight_mesh += box
            
        # Re-add geometry because topology (vertex count) changes dynamically
        self.vis.add_geometry(self.highlight_mesh, reset_bounding_box=False)

    def update_camera_scale(self):
        if not hasattr(self, 'cameras_rot') or not hasattr(self, 'focal_px'): return
        
        plate_w = self.camera_setup_data.get('plate_width', 1920)
        plate_h = self.camera_setup_data.get('plate_height', 1080)
        cx, cy = plate_w / 2.0, plate_h / 2.0
        
        if hasattr(self, 'full_points') and len(self.full_points) > 0:
            min_bound = np.min(self.full_points, axis=0)
            max_bound = np.max(self.full_points, axis=0)
            max_extent = np.max(max_bound - min_bound)
        else:
            max_extent = 10.0
            
        z = max_extent * 0.01 * (self.slider_cam_scale.value() / 10.0)
        
        x_max = (plate_w - cx) / self.focal_px * z
        x_min = (0 - cx) / self.focal_px * z
        y_max = (plate_h - cy) / self.focal_px * z
        y_min = (0 - cy) / self.focal_px * z
        
        frustum_pts = np.array([[0, 0, 0], [x_min, y_min, z], [x_max, y_min, z], [x_max, y_max, z], [x_min, y_max, z]])
        
        idx = 0
        for i in range(len(self.cameras_rot)):
            R = self.cameras_rot[i]
            T = self.cameras_trans[i]
            if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)): continue
            
            R_inv = R.T
            cam_center = -R_inv @ T
            world_pts = (R_inv @ frustum_pts.T).T + cam_center
            
            ls = self.camera_linesets[idx]
            ls.points = o3d.utility.Vector3dVector(world_pts)
            self.vis.update_geometry(ls)
            idx += 1
            
        self.update_error_threshold()
            
class ProxyGeoViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("O3D_ProxyGeo", width=800, height=600, visible=False)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 2.0
        
        for _ in range(10):
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)
            
        hwnd = win32gui.FindWindow(None, "O3D_ProxyGeo")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        if hwnd:
            window = QWindow.fromWinId(hwnd)
            self.widget3d = QWidget.createWindowContainer(window)
            self.layout.addWidget(self.widget3d)
        else:
            self.layout.addWidget(QLabel("Failed to embed Open3D ProxyGeo viewport."))
            
        self.point_cloud = None
        self.camera_linesets = []
        self.mesh = None
        
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        self.origin_scale = 5.0
        self.grid = self.create_grid(size=100, divisions=100)
        self.vis.add_geometry(self.origin_frame)
        self.vis.add_geometry(self.grid)
        
        self.control_panel = QWidget()
        self.control_layout = QHBoxLayout(self.control_panel)
        
        self.control_layout.addWidget(QLabel("Perspective:"))
        self.combo_perspective = QComboBox()
        self.combo_perspective.addItems(["Free Camera", "Scene Camera"])
        self.combo_perspective.setCurrentIndex(1) # Default to Scene Camera
        self.combo_perspective.currentIndexChanged.connect(self.update_perspective)
        self.control_layout.addWidget(self.combo_perspective)
        
        self.control_layout.addWidget(QLabel("Origin Axis Size:"))
        self.slider_origin_scale = QSlider(Qt.Horizontal)
        self.slider_origin_scale.setRange(1, 200)
        self.slider_origin_scale.setValue(50)
        self.slider_origin_scale.valueChanged.connect(self.update_origin_scale)
        self.control_layout.addWidget(self.slider_origin_scale)
        
        self.control_layout.addStretch()
        self.layout.addWidget(self.control_panel)
        
        self.current_idx = 0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick_o3d)
        self.timer.start(16)

    def create_grid(self, size=100, divisions=100):
        lines = []
        points = []
        step = size / divisions
        half = size / 2.0
        
        idx = 0
        for i in range(divisions + 1):
            val = -half + i * step
            points.extend([[val, 0, -half], [val, 0, half]])
            lines.append([idx, idx + 1])
            idx += 2
            
            points.extend([[-half, 0, val], [half, 0, val]])
            lines.append([idx, idx + 1])
            idx += 2
            
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3] for _ in range(len(lines))])
        return grid

    def _tick_o3d(self):
        if hasattr(self, 'combo_perspective') and self.combo_perspective.currentText() == "Scene Camera":
            self.sync_perspective_to_camera(self.current_idx)
        self.vis.poll_events()
        self.vis.update_renderer()

    def load_solve_data(self, data_path, proxy_res, camera_setup_data):
        if self.point_cloud:
            self.vis.remove_geometry(self.point_cloud)
        for ls in self.camera_linesets:
            self.vis.remove_geometry(ls)
        if self.mesh:
            self.vis.remove_geometry(self.mesh)
            
        self.point_cloud = None
        self.camera_linesets = []
        self.mesh = None
        if not os.path.exists(data_path): return
        
        data = np.load(data_path)
        points_3d = data['points_3d']
        points_mask = data['points_mask']
        self.cameras_rot = data['cameras_rot']
        self.cameras_trans = data['cameras_trans']
        self.focal_px = float(data['focal_px'])
        self.plate_w = camera_setup_data.get('plate_width', 1920)
        self.plate_h = camera_setup_data.get('plate_height', 1080)
        
        valid_pts = points_3d[points_mask]
        if len(valid_pts) > 0:
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(valid_pts)
            self.point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            self.vis.add_geometry(self.point_cloud)
            
        self.camera_linesets = []
        for i in range(len(self.cameras_rot)):
            R = self.cameras_rot[i]
            T = self.cameras_trans[i]
            if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)): continue
            
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.zeros((5, 3)))
            lines = [[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1]]
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(np.tile([1, 0.5, 0], (8, 1)))
            self.camera_linesets.append(ls)
            self.vis.add_geometry(ls)
            
        focal_px = float(data['focal_px'])
        plate_w, plate_h = self.plate_w, self.plate_h
        cx, cy = plate_w / 2, plate_h / 2
        
        if len(valid_pts) > 0:
            min_bound = np.min(valid_pts, axis=0)
            max_bound = np.max(valid_pts, axis=0)
            max_extent = np.max(max_bound - min_bound)
        else:
            max_extent = 10.0
            
        z = max_extent * 0.05
        x_max = (plate_w - cx) / focal_px * z
        x_min = (0 - cx) / focal_px * z
        y_max = (plate_h - cy) / focal_px * z
        y_min = (0 - cy) / focal_px * z
        frustum_pts = np.array([[0,0,0], [x_min,y_min,z], [x_max,y_min,z], [x_max,y_max,z], [x_min,y_max,z]])
        
        idx = 0
        for i in range(len(self.cameras_rot)):
            R = self.cameras_rot[i]
            T = self.cameras_trans[i]
            if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)): continue
            R_inv = R.T
            cam_center = -R_inv @ T
            world_pts = (R_inv @ frustum_pts.T).T + cam_center
            self.camera_linesets[idx].points = o3d.utility.Vector3dVector(world_pts)
            self.vis.update_geometry(self.camera_linesets[idx])
            idx += 1
            
        vc = self.vis.get_view_control()
        vc.set_up([0, -1, 0])
        vc.set_front([0, 0, -1])
        vc.set_lookat([0, 0, 5])
        vc.set_zoom(0.8)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        self.update_perspective()

    def update_active_camera(self, idx):
        self.current_idx = idx
        if not self.camera_linesets: return
        
        camera_idx = 0
        for i in range(len(self.cameras_rot)):
            R = self.cameras_rot[i]
            T = self.cameras_trans[i]
            if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)): continue
            
            ls = self.camera_linesets[camera_idx]
            if i == idx:
                ls.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (8, 1))) # Green for active
            else:
                ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.5, 0.0], (8, 1))) # Orange for inactive
            self.vis.update_geometry(ls)
            camera_idx += 1
            
        if hasattr(self, 'combo_perspective') and self.combo_perspective.currentText() == "Scene Camera":
            self.sync_perspective_to_camera(idx)
            
    def update_perspective(self):
        is_scene_cam = self.combo_perspective.currentText() == "Scene Camera"
        if is_scene_cam:
            self.sync_perspective_to_camera(self.current_idx)
            
        for ls in self.camera_linesets:
            if is_scene_cam:
                self.vis.remove_geometry(ls, reset_bounding_box=False)
            else:
                self.vis.add_geometry(ls, reset_bounding_box=False)
                
        if not is_scene_cam:
            self.update_active_camera(self.current_idx)
            
    def sync_perspective_to_camera(self, idx):
        if not hasattr(self, 'cameras_rot') or idx >= len(self.cameras_rot): return
        
        R = self.cameras_rot[idx]
        T = self.cameras_trans[idx]
        
        if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)):
            return
            
        plate_w, plate_h = self.plate_w, self.plate_h
        cx, cy = plate_w / 2.0, plate_h / 2.0
        fx = fy = self.focal_px
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(plate_w), height=int(plate_h), fx=fx, fy=fy, cx=cx, cy=cy
        )
        
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = intrinsic
        
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T
        cam_params.extrinsic = extrinsic
        
        view_ctl = self.vis.get_view_control()
        view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    def update_origin_scale(self):
        self.origin_scale = self.slider_origin_scale.value() / 10.0
        self.vis.remove_geometry(self.origin_frame, reset_bounding_box=False)
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.origin_scale)
        self.vis.add_geometry(self.origin_frame, reset_bounding_box=False)

    def load_proxy_mesh(self, mesh_path):
        if self.mesh:
            self.vis.remove_geometry(self.mesh)
        if os.path.exists(mesh_path):
            self.mesh = o3d.io.read_triangle_mesh(mesh_path)
            self.mesh.compute_vertex_normals()
            self.mesh.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.add_geometry(self.mesh)
            self.vis.poll_events()
            self.vis.update_renderer()
            return len(self.mesh.triangles)
        return 0

class TSDFWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, project_dir, subsample_rate, plate_w, plate_h):
        super().__init__()
        self.project_dir = project_dir
        self.subsample_rate = subsample_rate
        self.plate_w = plate_w
        self.plate_h = plate_h

    def run(self):
        try:
            from scipy.optimize import least_squares
            
            self.status.emit("Loading solve data...")
            solve_path = os.path.join(self.project_dir, 'solve_data.npz')
            if not os.path.exists(solve_path):
                raise Exception("solve_data.npz not found.")
                
            data = np.load(solve_path)
            points_3d = data['points_3d']
            points_mask = data['points_mask']
            cameras_rot = data['cameras_rot']
            cameras_trans = data['cameras_trans']
            focal_px = float(data['focal_px'])
            visibility = data['visibility']
            track_2d = data['tracks_2d']
            
            depth_dir = os.path.join(self.project_dir, 'depth_proxies')
            if not os.path.exists(depth_dir):
                raise Exception("AI Depth proxies not found. Please generate them in Setup/Solve tab first.")
                
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
            if not depth_files:
                raise Exception("No .npy depth files found in depth_proxies directory.")
                
            first_depth = np.load(os.path.join(depth_dir, depth_files[0]))
            H, W = first_depth.shape
            
            # CRITICAL: tracks_2d and focal_px are in NATIVE resolution (e.g. 3840x2160),
            # but depth maps are at PROXY resolution (e.g. 1920x1080).
            # Use the actual native plate dimensions passed from the UI.
            native_w = self.plate_w
            native_h = self.plate_h
            
            # Scale factor: native -> proxy (depth map) resolution
            scale_x = W / native_w  
            scale_y = H / native_h
            
            # Focal length scaled to proxy (depth map) resolution
            focal_proxy = focal_px * scale_x
            cx, cy = W / 2.0, H / 2.0
            
            num_frames = len(cameras_rot)

            # --- Screen-Space Depth Meshing ---
            # Instead of fusing noisy point clouds and hoping Poisson creates
            # something clean, we build a structured triangle grid directly from
            # each depth map's pixel grid. This guarantees clean, connected
            # topology by construction.
            
            # Downsample factor for depth grid (every Nth pixel becomes a vertex)
            grid_step = 4
            grid_h = H // grid_step
            grid_w = W // grid_step
            
            self.status.emit("Building screen-space depth meshes...")
            
            frames_to_process = list(range(0, num_frames, self.subsample_rate))
            total_work = len(frames_to_process)
            
            all_vertices = []
            all_triangles = []
            vertex_offset = 0
            
            for idx, i in enumerate(frames_to_process):
                if i >= len(depth_files): break
                R = cameras_rot[i]
                T = cameras_trans[i]
                if np.allclose(R, np.eye(3)) and np.allclose(T, np.zeros(3)): continue
                
                d_ai = np.load(os.path.join(depth_dir, depth_files[i]))
                
                # --- Per-frame affine depth alignment ---
                visible_mask = visibility[i, :] > 0
                frame_pts_mask = points_mask & visible_mask
                frame_pts_3d = points_3d[frame_pts_mask]
                
                if len(frame_pts_3d) < 10:
                    continue
                    
                pts_cam = (R @ frame_pts_3d.T).T + T
                z_sfm = pts_cam[:, 2]
                
                valid_z = z_sfm > 0
                if np.sum(valid_z) < 10: continue
                
                z_sfm = z_sfm[valid_z]
                
                pts_indices = np.where(frame_pts_mask)[0]
                pts_indices = pts_indices[valid_z]
                
                # tracks_2d is in NATIVE resolution — scale to PROXY resolution for depth map sampling
                x_px_native = track_2d[i, pts_indices, 0]
                y_px_native = track_2d[i, pts_indices, 1]
                x_px = x_px_native * scale_x
                y_px = y_px_native * scale_y
                
                valid_px = (x_px >= 0) & (x_px < W-1) & (y_px >= 0) & (y_px < H-1)
                if np.sum(valid_px) < 10: continue
                
                xi = np.clip(x_px[valid_px].astype(int), 0, W-1)
                yi = np.clip(y_px[valid_px].astype(int), 0, H-1)
                z_sfm_valid = z_sfm[valid_px]
                
                disp_ai = d_ai[yi, xi]
                
                # Fit in DISPARITY space: 1/z_sfm = a * disp + b
                # This avoids catastrophic 1/0 and 1/negative explosions from raw disparity
                inv_z_sfm = 1.0 / z_sfm_valid
                valid_disp = disp_ai > 0.01
                if np.sum(valid_disp) < 10: continue
                
                disp_fit = disp_ai[valid_disp]
                inv_z_fit = inv_z_sfm[valid_disp]
                
                def loss_func_disp(params, disp, inv_z):
                    a, b = params
                    return (a * disp + b) - inv_z
                
                a_g = np.median(inv_z_fit) / (np.median(disp_fit) + 1e-6)
                b_g = np.median(inv_z_fit) - a_g * np.median(disp_fit)
                
                res = least_squares(
                    loss_func_disp, [a_g, b_g], loss='soft_l1', args=(disp_fit, inv_z_fit)
                )
                a, b = res.x
                
                # Convert full depth map to metric depth: z = 1 / (a*disp + b)
                inv_metric = a * d_ai + b
                metric_depth = np.where(inv_metric > 1e-6, 1.0 / inv_metric, 200.0)
                metric_depth = np.clip(metric_depth, 0.1, 200.0)
                
                # Bilateral filter: smooths noise while preserving sharp depth edges
                import cv2
                metric_f32 = metric_depth.astype(np.float32)
                metric_depth = cv2.bilateralFilter(metric_f32, d=9, sigmaColor=5.0, sigmaSpace=9.0).astype(np.float64)
                
                # --- Build screen-space grid mesh for this frame ---
                # Sample at grid_step intervals
                gy, gx = np.mgrid[0:grid_h, 0:grid_w]
                py = gy * grid_step  # pixel y coordinates in proxy space
                px = gx * grid_step  # pixel x coordinates in proxy space
                py = np.clip(py, 0, H-1)
                px = np.clip(px, 0, W-1)
                
                depths = metric_depth[py, px]  # (grid_h, grid_w)
                
                # Back-project proxy pixels to camera space using PROXY-scaled focal
                cam_x = (px - cx) / focal_proxy * depths
                cam_y = (py - cy) / focal_proxy * depths
                cam_z = depths
                
                # Stack into (grid_h, grid_w, 3)
                pts_cam_grid = np.stack([cam_x, cam_y, cam_z], axis=-1)
                
                # Transform camera → world:  P_world = R^T @ (P_cam - T)
                R_inv = R.T
                cam_center = -R_inv @ T
                pts_flat = pts_cam_grid.reshape(-1, 3)
                pts_world = (R_inv @ (pts_flat - T).T).T  # (N, 3)
                
                # Build triangle indices for the grid — fully vectorised
                # Grid of top-left vertex indices for every quad cell
                rows = np.arange(grid_h - 1)
                cols = np.arange(grid_w - 1)
                row_idx, col_idx = np.meshgrid(rows, cols, indexing='ij')  # (R-1, C-1)
                
                v00 = (row_idx * grid_w + col_idx).ravel()
                v01 = (row_idx * grid_w + col_idx + 1).ravel()
                v10 = ((row_idx + 1) * grid_w + col_idx).ravel()
                v11 = ((row_idx + 1) * grid_w + col_idx + 1).ravel()
                
                # Depth at each quad corner
                d00 = depths[row_idx, col_idx].ravel()
                d01 = depths[row_idx, col_idx + 1].ravel()
                d10 = depths[row_idx + 1, col_idx].ravel()
                d11 = depths[row_idx + 1, col_idx + 1].ravel()
                
                # Depth discontinuity filter — all in one vectorised expression
                quad_max = np.maximum(np.maximum(d00, d01), np.maximum(d10, d11))
                quad_min = np.minimum(np.minimum(d00, d01), np.minimum(d10, d11))
                keep = (quad_max > 0.1) & ((quad_max - quad_min) / (quad_min + 1e-6) < 0.3)
                
                v00k = v00[keep] + vertex_offset
                v01k = v01[keep] + vertex_offset
                v10k = v10[keep] + vertex_offset
                v11k = v11[keep] + vertex_offset
                
                # Two triangles per quad, stacked as (N, 3)
                # Note: Winding order reversed (v10k before v01k) so normals point toward camera
                tri_a = np.stack([v00k, v10k, v01k], axis=1)
                tri_b = np.stack([v01k, v10k, v11k], axis=1)
                frame_tris_arr = np.vstack([tri_a, tri_b]).astype(np.int32)
                
                all_vertices.append(pts_world)
                if len(frame_tris_arr) > 0:
                    all_triangles.append(frame_tris_arr)
                vertex_offset += pts_world.shape[0]
                
                self.progress.emit(int((idx / total_work) * 60))
            
            if not all_vertices:
                raise Exception("No valid frames could be processed.")
                
            self.status.emit("Merging mesh fragments...")
            combined_verts = np.vstack(all_vertices).astype(np.float64)
            if all_triangles:
                combined_tris = np.vstack(all_triangles).astype(np.int32)
            else:
                raise Exception("No triangles were generated.")
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(combined_verts)
            mesh.triangles = o3d.utility.Vector3iVector(combined_tris)
            
            self.progress.emit(65)
            self.status.emit("Cleaning up mesh...")
            
            # Remove degenerate and duplicated triangles
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            
            # Keep only the largest connected component
            try:
                triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                
                if len(cluster_n_triangles) > 1:
                    # Keep the top N clusters that contain >1% of total triangles
                    total_tris = len(mesh.triangles)
                    min_cluster_size = max(total_tris * 0.01, 100)
                    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_cluster_size
                    mesh.remove_triangles_by_mask(triangles_to_remove)
                    mesh.remove_unreferenced_vertices()
            except Exception as e:
                print("Mesh clustering failed:", e)
            
            self.progress.emit(75)
            self.status.emit("Snapping to SfM point cloud...")
            
            # Snap nearest mesh vertices to exact SfM 3D positions
            # Use scipy cKDTree for a single vectorised batch query over all SfM points
            valid_sfm_points = points_3d[points_mask]
            if len(valid_sfm_points) > 0 and len(mesh.vertices) > 0:
                from scipy.spatial import cKDTree
                mesh_verts = np.asarray(mesh.vertices).copy()
                tree = cKDTree(mesh_verts)
                dists, vert_indices = tree.query(valid_sfm_points, workers=-1)
                snap_mask = dists < 5.0
                mesh_verts[vert_indices[snap_mask]] = valid_sfm_points[snap_mask]
                mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
            
            self.progress.emit(90)
            self.status.emit("Computing normals...")
            mesh.compute_vertex_normals()
            
            self.status.emit("Saving raw geometry...")
            raw_path = os.path.join(self.project_dir, 'proxy_geo_raw.obj')
            out_path = os.path.join(self.project_dir, 'proxy_geo.obj')
            o3d.io.write_triangle_mesh(raw_path, mesh)
            o3d.io.write_triangle_mesh(out_path, mesh) # Also write as active
                
            self.progress.emit(100)
            self.finished.emit()
            
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())

class DecimateWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, project_dir, target_tris):
        super().__init__()
        self.project_dir = project_dir
        self.target_tris = target_tris

    def run(self):
        try:
            import open3d as o3d
            
            self.status.emit("Loading current mesh...")
            self.progress.emit(10)
            
            current_path = os.path.join(self.project_dir, 'proxy_geo.obj')
            if not os.path.exists(current_path):
                raise Exception("No proxy geometry found to decimate. Generate it first.")
            mesh = o3d.io.read_triangle_mesh(current_path)
            
            if len(mesh.triangles) <= self.target_tris:
                self.status.emit("Mesh already at or below target triangles.")
                self.progress.emit(100)
                self.finished.emit()
                return
                
            self.status.emit("Decimating mesh...")
            self.progress.emit(30)
            
            mesh = mesh.simplify_quadric_decimation(self.target_tris)
            
            self.progress.emit(80)
            self.status.emit("Computing normals...")
            mesh.compute_vertex_normals()
            
            self.status.emit("Saving mesh...")
            out_path = os.path.join(self.project_dir, 'proxy_geo.obj')
            o3d.io.write_triangle_mesh(out_path, mesh)
            
            self.progress.emit(100)
            self.finished.emit()
            
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())

class SmoothWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, project_dir, iters):
        super().__init__()
        self.project_dir = project_dir
        self.iters = iters

    def run(self):
        try:
            import open3d as o3d
            
            self.status.emit("Loading current mesh...")
            self.progress.emit(10)
            
            current_path = os.path.join(self.project_dir, 'proxy_geo.obj')
            if not os.path.exists(current_path):
                raise Exception("No proxy geometry found to smooth. Generate it first.")
                    
            mesh = o3d.io.read_triangle_mesh(current_path)
                
            self.status.emit("Smoothing mesh...")
            self.progress.emit(30)
            
            mesh = mesh.filter_smooth_taubin(number_of_iterations=self.iters)
            
            self.progress.emit(80)
            self.status.emit("Computing normals...")
            mesh.compute_vertex_normals()
            
            self.status.emit("Saving mesh...")
            o3d.io.write_triangle_mesh(current_path, mesh)
            
            self.progress.emit(100)
            self.finished.emit()
            
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())

class MainWindow(QMainWindow):
    def closeEvent(self, event):
        # Kill daemon process if running
        if hasattr(self, 'daemon_process') and self.daemon_process and self.daemon_process.is_alive():
            self.daemon_process.terminate()
            self.daemon_process.join()
            
        # Kill workers
        workers = ['dl_worker', 'ai_depth_worker', 'proxy_worker', 'tsdf_worker']
        for w in workers:
            if hasattr(self, w):
                worker = getattr(self, w)
                if worker and worker.isRunning():
                    worker.terminate()
                    worker.wait()
                    
        event.accept()
        import sys
        sys.exit(0)

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
        
        self.camera_setup_tab = QWidget()
        self.tabs.addTab(self.camera_setup_tab, "Camera Setup")
        self.init_camera_setup_tab()
        
        self.solve_tab = QWidget()
        self.tabs.addTab(self.solve_tab, "Solve")
        self.init_solve_tab()
        
        self.proxy_geo_tab = QWidget()
        self.tabs.addTab(self.proxy_geo_tab, "Proxy Geo")
        self.init_proxy_geo_tab()
        
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

    def init_camera_setup_tab(self):
        layout = QVBoxLayout()
        
        # Camera preset selection
        group_camera = QGroupBox("1. Camera Profile")
        camera_layout = QHBoxLayout()
        self.combo_cameras = QComboBox()
        self.lbl_sensor_info = QLabel("Physical Sensor: N/A")
        camera_layout.addWidget(QLabel("Select Camera:"))
        camera_layout.addWidget(self.combo_cameras)
        camera_layout.addWidget(self.lbl_sensor_info)
        camera_layout.addStretch()
        group_camera.setLayout(camera_layout)
        layout.addWidget(group_camera)
        
        # Original Plate Resolution
        group_res = QGroupBox("2. Original Plate Resolution (Pre-Undistortion)")
        res_layout = QGridLayout()
        self.spin_orig_w = QSpinBox()
        self.spin_orig_w.setRange(1, 16384)
        self.spin_orig_w.setValue(3840)
        self.spin_orig_h = QSpinBox()
        self.spin_orig_h.setRange(1, 16384)
        self.spin_orig_h.setValue(2160)
        
        self.chk_assume_no_crop = QCheckBox("Unknown / Assume No Crop")
        
        res_layout.addWidget(QLabel("Width:"), 0, 0)
        res_layout.addWidget(self.spin_orig_w, 0, 1)
        res_layout.addWidget(QLabel("Height:"), 0, 2)
        res_layout.addWidget(self.spin_orig_h, 0, 3)
        res_layout.addWidget(self.chk_assume_no_crop, 0, 4)
        res_layout.setColumnStretch(5, 1)
        group_res.setLayout(res_layout)
        layout.addWidget(group_res)
        
        # Focal Length
        group_focal = QGroupBox("3. Lens Focal Length")
        focal_layout = QHBoxLayout()
        self.spin_focal = QSpinBox()
        self.spin_focal.setRange(1, 1000)
        self.spin_focal.setValue(35)
        self.spin_focal.setSuffix(" mm")
        self.chk_auto_focal = QCheckBox("Auto-Estimate")
        
        focal_layout.addWidget(QLabel("Focal Length:"))
        focal_layout.addWidget(self.spin_focal)
        focal_layout.addWidget(self.chk_auto_focal)
        focal_layout.addStretch()
        group_focal.setLayout(focal_layout)
        layout.addWidget(group_focal)
        
        # Effective Sensor Math Output
        group_eff = QGroupBox("4. Effective Sensor Size (Calculated)")
        eff_layout = QVBoxLayout()
        self.lbl_eff_sensor = QLabel("Effective Sensor: N/A")
        self.lbl_eff_sensor.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        
        self.lbl_warning = QLabel("<b>Warning:</b> Utilizing 'Unknown / Assume No Crop' or 'Auto-Estimate' fallbacks will produce a mathematically accurate projection, but physical lens properties will be fabricated and may affect downstream renders (like Z-Defocus).")
        self.lbl_warning.setStyleSheet("color: #FF9800; font-style: italic;")
        self.lbl_warning.setWordWrap(True)
        
        eff_layout.addWidget(self.lbl_eff_sensor)
        eff_layout.addWidget(self.lbl_warning)
        group_eff.setLayout(eff_layout)
        layout.addWidget(group_eff)
        
        btn_proceed = QPushButton("Save && Proceed to Solve")
        btn_proceed.clicked.connect(self.save_camera_setup)
        btn_proceed.setMinimumHeight(40)
        layout.addWidget(btn_proceed)
        
        layout.addStretch()
        self.camera_setup_tab.setLayout(layout)
        
        # Connect signals for dynamic math updates
        self.combo_cameras.currentIndexChanged.connect(self.update_camera_math)
        self.spin_orig_w.valueChanged.connect(self.update_camera_math)
        self.spin_orig_h.valueChanged.connect(self.update_camera_math)
        self.chk_assume_no_crop.toggled.connect(self.on_assume_no_crop_toggled)
        self.chk_auto_focal.toggled.connect(self.on_auto_focal_toggled)
        
        self.load_cameras_json()

    def load_cameras_json(self):
        cameras_path = os.path.join(os.path.dirname(__file__), "cameras.json")
        if not os.path.exists(cameras_path):
            cameras_path = os.path.join(os.getcwd(), "cameras.json")
            
        if os.path.exists(cameras_path):
            try:
                with open(cameras_path, 'r') as f:
                    data = json.load(f)
                    
                self.camera_profiles = data.get("cameras", [])
                for cam in self.camera_profiles:
                    name = f"{cam.get('make', '')} {cam.get('model', '')} ({cam.get('format', '')})"
                    self.combo_cameras.addItem(name, userData=cam)
            except Exception as e:
                print(f"Failed to load cameras.json: {e}")
        else:
            print("cameras.json not found")
            self.camera_profiles = []

    def on_assume_no_crop_toggled(self, checked):
        self.spin_orig_w.setEnabled(not checked)
        self.spin_orig_h.setEnabled(not checked)
        self.update_camera_math()
        
    def on_auto_focal_toggled(self, checked):
        self.spin_focal.setEnabled(not checked)

    def update_camera_math(self):
        cam = self.combo_cameras.currentData()
        if not cam:
            return
            
        phys_w = float(cam.get("sensor_width_mm", 36.0))
        phys_h = float(cam.get("sensor_height_mm", 24.0))
        
        self.lbl_sensor_info.setText(f"Physical Sensor: {phys_w:.2f}mm x {phys_h:.2f}mm")
        
        if not hasattr(self, 'native_sizes') or not self.native_sizes:
            undistorted_w, undistorted_h = 3840, 2160 # default fallback
        else:
            undistorted_w, undistorted_h = self.native_sizes[0]
            
        if self.chk_assume_no_crop.isChecked():
            orig_w = undistorted_w
        else:
            orig_w = self.spin_orig_w.value()
            
        if orig_w == 0:
            orig_w = 1 # prevent div by zero
            
        # Formula: Effective Sensor Width = Physical Sensor Width * (Undistorted EXR Width / Original Plate Width)
        eff_w = phys_w * (undistorted_w / orig_w)
        
        self.lbl_eff_sensor.setText(f"Effective Sensor Width: {eff_w:.4f} mm")
        self.current_eff_sensor_width = eff_w

    def save_camera_setup(self):
        if not hasattr(self, 'project_file') or not self.project_file:
            QMessageBox.warning(self, "Warning", "Please create or load a project first.")
            return
            
        undistorted_w, undistorted_h = self.native_sizes[0] if hasattr(self, 'native_sizes') and self.native_sizes else (1920, 1080)
        
        self.camera_setup_data = {
            "effective_sensor_width": getattr(self, 'current_eff_sensor_width', 36.0),
            "focal_length": self.spin_focal.value() if not self.chk_auto_focal.isChecked() else "auto",
            "auto_estimate_focal": self.chk_auto_focal.isChecked(),
            "plate_width": undistorted_w,
            "plate_height": undistorted_h,
            "ui_camera_name": self.combo_cameras.currentText(),
            "ui_orig_w": self.spin_orig_w.value(),
            "ui_orig_h": self.spin_orig_h.value(),
            "ui_assume_no_crop": self.chk_assume_no_crop.isChecked(),
            "ui_focal_val": self.spin_focal.value()
        }
        self.save_project()
        self.tabs.setCurrentWidget(self.solve_tab)
        QMessageBox.information(self, "Success", "Camera settings saved to project. Ready for 3D Solve.")

    def init_solve_tab(self):
        self.solve_layout = QVBoxLayout()
        top_ctrl_layout = QHBoxLayout()
        
        # 1. Base Structure Group
        group_base = QGroupBox("1. Base Structure")
        layout_base = QVBoxLayout()
        self.btn_start_solve = QPushButton("Start 3D Solver (VGGSfM)")
        self.btn_start_solve.setMinimumHeight(40)
        self.btn_start_solve.clicked.connect(self.run_solve)
        layout_base.addWidget(self.btn_start_solve)
        layout_base.addStretch()
        group_base.setLayout(layout_base)
        
        # 2. AI Spatial Refinement Group
        group_ai = QGroupBox("2. AI Spatial Refinement")
        layout_ai = QGridLayout()
        
        self.btn_gen_depth = QPushButton("Generate Depth Sequence")
        self.btn_gen_depth.clicked.connect(self.generate_depth_sequence)
        self.btn_gen_depth.setEnabled(False)
        
        self.combo_ai_mode = QComboBox()
        self.combo_ai_mode.addItems([
            "Best View (Fast)",
            "N-Keyframe Average (5 frames)",
            "Exhaustive Median (All frames - SLOW)"
        ])
        self.combo_ai_mode.setToolTip(
            "Best View: Selects the single best frame for each point based on visibility.\n"
            "N-Keyframe Average: Samples 5 evenly spaced frames and averages the depth.\n"
            "Exhaustive Median: Runs AI on every tracked frame and takes the robust median (Very Slow)."
        )
        
        self.lbl_ai_blend = QLabel("Strength: 50%")
        self.ai_blend_slider = QSlider(Qt.Horizontal)
        self.ai_blend_slider.setRange(0, 100)
        self.ai_blend_slider.setValue(50)
        self.ai_blend_slider.setFixedWidth(100)
        self.ai_blend_slider.valueChanged.connect(lambda v: self.lbl_ai_blend.setText(f"Strength: {v}%"))
        
        self.btn_ai_verify = QPushButton("AI Spatial Smoothing")
        self.btn_ai_verify.clicked.connect(self.run_ai_depth_verification)
        self.btn_ai_verify.setMinimumHeight(35)
        
        layout_ai.addWidget(self.btn_gen_depth, 0, 0, 1, 2)
        layout_ai.addWidget(QLabel("Mode:"), 1, 0)
        layout_ai.addWidget(self.combo_ai_mode, 1, 1)
        layout_ai.addWidget(self.lbl_ai_blend, 2, 0)
        layout_ai.addWidget(self.ai_blend_slider, 2, 1)
        layout_ai.addWidget(self.btn_ai_verify, 3, 0, 1, 2)
        group_ai.setLayout(layout_ai)
        
        # 3. Data Management Group
        group_data = QGroupBox("3. Data Management")
        layout_data = QVBoxLayout()
        
        self.btn_toggle_view = QPushButton("Preview Raw Data")
        self.btn_toggle_view.clicked.connect(self.toggle_solve_view)
        
        self.btn_revert_solve = QPushButton("Revert to Raw Solve")
        self.btn_revert_solve.clicked.connect(self.revert_solve)
        
        layout_data.addWidget(self.btn_toggle_view)
        layout_data.addWidget(self.btn_revert_solve)
        layout_data.addStretch()
        group_data.setLayout(layout_data)
        
        top_ctrl_layout.addWidget(group_base)
        top_ctrl_layout.addWidget(group_ai)
        top_ctrl_layout.addWidget(group_data)
        
        self.solve_layout.addLayout(top_ctrl_layout)
        
        self.lbl_solve_status = QLabel("")
        self.solve_layout.addWidget(self.lbl_solve_status)
        
        self.solve_progress = QProgressBar()
        self.solve_progress.setValue(0)
        self.solve_progress.hide()
        self.solve_layout.addWidget(self.solve_progress)
        
        from PySide6.QtWidgets import QSplitter
        self.solve_splitter = QSplitter(Qt.Horizontal)
        
        self.solve_2d_viewport = SequenceViewerWidget(self, interactive=True, show_tracks=True)
        self.solve_2d_viewport.hide()
        self.solve_2d_viewport.trackPicked.connect(self.on_track_picked)
        
        self.solve_viewport = SolveViewport()
        self.solve_viewport.hide()
        
        self.solve_splitter.addWidget(self.solve_2d_viewport)
        self.solve_splitter.addWidget(self.solve_viewport)
        self.solve_splitter.setSizes([500, 500])
        
        self.solve_layout.addWidget(self.solve_splitter, stretch=1)
        
        self.solve_viewport.activeCameraChanged.connect(self.solve_2d_viewport.on_frame_changed)
        self.solve_viewport.slider_error.valueChanged.connect(self.update_2d_solve_colors)
        
        self.solve_viewport.reqClearSelection.connect(self.clear_selection)
        self.solve_viewport.reqAddConstraint.connect(self.add_orientation_constraint)
        self.solve_viewport.reqDelConstraint.connect(self.del_orientation_constraint)
        self.solve_viewport.reqApplyOrientation.connect(self.apply_orientation_constraints)
        self.solve_viewport.reqSelectConstraintIdx.connect(self.select_orientation_constraint)
        
        self.selected_tracks = []
        self.orientation_constraints = []
        
        self.solve_tab.setLayout(self.solve_layout)

    def init_proxy_geo_tab(self):
        layout = QVBoxLayout()
        
        ctrl_layout = QHBoxLayout()
        
        self.btn_gen_proxy_geo = QPushButton("Generate Raw Geometry")
        self.btn_gen_proxy_geo.setMinimumHeight(40)
        self.btn_gen_proxy_geo.clicked.connect(self.run_tsdf_worker)
        
        self.btn_decimate_proxy_geo = QPushButton("Decimate Mesh")
        self.btn_decimate_proxy_geo.setMinimumHeight(40)
        self.btn_decimate_proxy_geo.clicked.connect(self.run_decimate_worker)
        
        self.btn_revert_proxy_geo = QPushButton("Revert to Raw")
        self.btn_revert_proxy_geo.setMinimumHeight(40)
        self.btn_revert_proxy_geo.clicked.connect(self.revert_proxy_geo)
        
        self.btn_smooth_proxy_geo = QPushButton("Smooth Mesh")
        self.btn_smooth_proxy_geo.setMinimumHeight(40)
        self.btn_smooth_proxy_geo.clicked.connect(self.run_smooth_worker)
        
        self.spin_target_tris = QSpinBox()
        self.spin_target_tris.setRange(1000, 1000000)
        self.spin_target_tris.setValue(50000)
        self.spin_target_tris.setSingleStep(5000)
        
        self.spin_smooth_iters = QSpinBox()
        self.spin_smooth_iters.setRange(1, 100)
        self.spin_smooth_iters.setValue(10)
        
        self.spin_subsample = QSpinBox()
        self.spin_subsample.setRange(1, 100)
        self.spin_subsample.setValue(5)
        self.spin_subsample.setToolTip("Temporal subsampling: process every Nth frame")
        
        ctrl_layout.addWidget(QLabel("Target Triangles:"))
        ctrl_layout.addWidget(self.spin_target_tris)
        ctrl_layout.addWidget(self.btn_decimate_proxy_geo)
        
        ctrl_layout.addWidget(QLabel("Smooth Iters:"))
        ctrl_layout.addWidget(self.spin_smooth_iters)
        ctrl_layout.addWidget(self.btn_smooth_proxy_geo)
        
        ctrl_layout.addWidget(QLabel("Subsample Step:"))
        ctrl_layout.addWidget(self.spin_subsample)
        ctrl_layout.addWidget(self.btn_gen_proxy_geo)
        ctrl_layout.addWidget(self.btn_revert_proxy_geo)
        
        layout.addLayout(ctrl_layout)
        
        self.lbl_proxy_geo_status = QLabel("")
        layout.addWidget(self.lbl_proxy_geo_status)
        
        self.lbl_tri_count = QLabel("Triangles: 0")
        self.lbl_tri_count.setStyleSheet("font-weight: bold; color: #a0a0a0;")
        layout.addWidget(self.lbl_tri_count)
        
        self.proxy_geo_progress = QProgressBar()
        self.proxy_geo_progress.setValue(0)
        self.proxy_geo_progress.hide()
        layout.addWidget(self.proxy_geo_progress)
        
        from PySide6.QtWidgets import QSplitter
        self.proxy_geo_splitter = QSplitter(Qt.Horizontal)
        
        self.proxy_geo_2d_viewport = SequenceViewerWidget(self, interactive=False, show_tracks=False)
        self.proxy_geo_3d_viewport = ProxyGeoViewport()
        self.proxy_geo_2d_viewport.slider.valueChanged.connect(self.proxy_geo_3d_viewport.update_active_camera)
        
        self.proxy_geo_splitter.addWidget(self.proxy_geo_2d_viewport)
        self.proxy_geo_splitter.addWidget(self.proxy_geo_3d_viewport)
        self.proxy_geo_splitter.setSizes([500, 500])
        
        layout.addWidget(self.proxy_geo_splitter, stretch=1)
        self.proxy_geo_tab.setLayout(layout)
        
    def run_tsdf_worker(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return
            
        self.btn_gen_proxy_geo.setEnabled(False)
        self.proxy_geo_progress.setValue(0)
        self.proxy_geo_progress.show()
        
        plate_w = self.camera_setup_data.get('plate_width', 3840)
        plate_h = self.camera_setup_data.get('plate_height', 2160)
        self.tsdf_worker = TSDFWorker(
            self.project_dir, 
            self.spin_subsample.value(),
            plate_w,
            plate_h
        )
        self.tsdf_worker.progress.connect(self.proxy_geo_progress.setValue)
        self.tsdf_worker.status.connect(self.lbl_proxy_geo_status.setText)
        self.tsdf_worker.finished.connect(self.on_tsdf_finished)
        self.tsdf_worker.error.connect(self.on_tsdf_error)
        self.tsdf_worker.start()
        
    def update_tri_count(self, count):
        self.lbl_tri_count.setText(f"Triangles: {count:,}")

    def on_tsdf_finished(self):
        self.lbl_proxy_geo_status.setText("Raw Proxy Geometry Generated!")
        self.btn_gen_proxy_geo.setEnabled(True)
        mesh_path = os.path.join(self.project_dir, 'proxy_geo_raw.obj')
        count = self.proxy_geo_3d_viewport.load_proxy_mesh(mesh_path)
        self.update_tri_count(count)
        
    def on_tsdf_error(self, err):
        self.lbl_proxy_geo_status.setText("Error generating geometry.")
        self.btn_gen_proxy_geo.setEnabled(True)
        if hasattr(self, 'btn_decimate_proxy_geo'):
            self.btn_decimate_proxy_geo.setEnabled(True)
            self.btn_smooth_proxy_geo.setEnabled(True)
            self.btn_revert_proxy_geo.setEnabled(True)
        QMessageBox.critical(self, "Geometry Error", err)
        
    def run_decimate_worker(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return
            
        self.btn_decimate_proxy_geo.setEnabled(False)
        self.btn_smooth_proxy_geo.setEnabled(False)
        self.btn_gen_proxy_geo.setEnabled(False)
        self.btn_revert_proxy_geo.setEnabled(False)
        self.proxy_geo_progress.setValue(0)
        self.proxy_geo_progress.show()
        
        self.decimate_worker = DecimateWorker(
            self.project_dir, 
            self.spin_target_tris.value()
        )
        self.decimate_worker.progress.connect(self.proxy_geo_progress.setValue)
        self.decimate_worker.status.connect(self.lbl_proxy_geo_status.setText)
        self.decimate_worker.finished.connect(self.on_decimate_finished)
        self.decimate_worker.error.connect(self.on_tsdf_error)
        self.decimate_worker.start()

    def on_decimate_finished(self):
        self.lbl_proxy_geo_status.setText("Mesh Decimated!")
        self.btn_decimate_proxy_geo.setEnabled(True)
        self.btn_smooth_proxy_geo.setEnabled(True)
        self.btn_gen_proxy_geo.setEnabled(True)
        self.btn_revert_proxy_geo.setEnabled(True)
        mesh_path = os.path.join(self.project_dir, 'proxy_geo.obj')
        count = self.proxy_geo_3d_viewport.load_proxy_mesh(mesh_path)
        self.update_tri_count(count)
        
    def run_smooth_worker(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return
            
        self.btn_decimate_proxy_geo.setEnabled(False)
        self.btn_smooth_proxy_geo.setEnabled(False)
        self.btn_gen_proxy_geo.setEnabled(False)
        self.btn_revert_proxy_geo.setEnabled(False)
        self.proxy_geo_progress.setValue(0)
        self.proxy_geo_progress.show()
        
        self.smooth_worker = SmoothWorker(
            self.project_dir, 
            self.spin_smooth_iters.value()
        )
        self.smooth_worker.progress.connect(self.proxy_geo_progress.setValue)
        self.smooth_worker.status.connect(self.lbl_proxy_geo_status.setText)
        self.smooth_worker.finished.connect(self.on_smooth_finished)
        self.smooth_worker.error.connect(self.on_tsdf_error)
        self.smooth_worker.start()

    def on_smooth_finished(self):
        self.lbl_proxy_geo_status.setText("Mesh Smoothed!")
        self.btn_decimate_proxy_geo.setEnabled(True)
        self.btn_smooth_proxy_geo.setEnabled(True)
        self.btn_gen_proxy_geo.setEnabled(True)
        self.btn_revert_proxy_geo.setEnabled(True)
        mesh_path = os.path.join(self.project_dir, 'proxy_geo.obj')
        count = self.proxy_geo_3d_viewport.load_proxy_mesh(mesh_path)
        self.update_tri_count(count)
        
    def revert_proxy_geo(self):
        if not self.project_dir: return
        raw_path = os.path.join(self.project_dir, 'proxy_geo_raw.obj')
        if not os.path.exists(raw_path):
            QMessageBox.warning(self, "Error", "No raw mesh found to revert to.")
            return
        import shutil
        out_path = os.path.join(self.project_dir, 'proxy_geo.obj')
        shutil.copy(raw_path, out_path)
        count = self.proxy_geo_3d_viewport.load_proxy_mesh(out_path)
        self.update_tri_count(count)
        self.lbl_proxy_geo_status.setText("Reverted to raw mesh.")

    def toggle_solve_view(self):
        if not hasattr(self, 'project_dir') or not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return
            
        data_path = os.path.join(self.project_dir, 'solve_data.npz')
        raw_data_path = os.path.join(self.project_dir, 'solve_data_raw.npz')
        
        if not os.path.exists(data_path) or not os.path.exists(raw_data_path):
            QMessageBox.warning(self, "Error", "Solve data is missing. Please run VGGSfM first.")
            return
            
        if not hasattr(self, 'viewing_raw'):
            self.viewing_raw = False
            
        self.viewing_raw = not self.viewing_raw
        
        if self.viewing_raw:
            oriented_raw_data = self.apply_orientation_constraints(return_data=True)
            if oriented_raw_data:
                preview_path = os.path.join(self.project_dir, 'solve_data_preview.npz')
                np.savez(preview_path, **oriented_raw_data)
                self.solve_viewport.load_solve_data(preview_path, self.camera_setup_data, reset_view=False)
            self.btn_toggle_view.setStyleSheet("background-color: #5a3b22; color: white;")
            self.btn_toggle_view.setText("Viewing: RAW (Un-smoothed)")
        else:
            if os.path.exists(data_path):
                self.solve_viewport.load_solve_data(data_path, self.camera_setup_data, reset_view=False)
            self.btn_toggle_view.setStyleSheet("")
            self.btn_toggle_view.setText("Preview Raw Data")

    def revert_solve(self):
        if not hasattr(self, 'project_dir') or not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return
            
        raw_data_path = os.path.join(self.project_dir, 'solve_data_raw.npz')
        if not os.path.exists(raw_data_path):
            QMessageBox.warning(self, "Error", "Raw solve backup not found. Please run VGGSfM first.")
            return
            
        self.viewing_raw = False
        if hasattr(self, 'btn_toggle_view'):
            self.btn_toggle_view.setStyleSheet("")
            self.btn_toggle_view.setText("Preview Raw Data")
        self.apply_orientation_constraints(reset_view=False)
        QMessageBox.information(self, "Reverted", "Restored the raw 3D point cloud from the solver.")

    def run_ai_depth_verification(self):
        if not hasattr(self, 'project_dir') or not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return
            
        data_path = os.path.join(self.project_dir, 'solve_data.npz')
        if not os.path.exists(data_path):
            QMessageBox.warning(self, "Error", "No solve data found. Run VGGSfM first.")
            return
            

            
        depth_proxies_dir = os.path.join(self.project_dir, 'depth_proxies')
        if not os.path.exists(depth_proxies_dir) or not os.listdir(depth_proxies_dir):
            QMessageBox.warning(self, "Missing Depth Sequence", "Please go to the Setup tab and click 'Generate Depth Sequence' first.")
            return
            
        data = dict(np.load(data_path))
        
        frames_dir = os.path.join(self.project_dir, "proxies")
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
        if not frame_files:
            QMessageBox.warning(self, "Error", "No proxies found.")
            return
            
        mode_idx = self.combo_ai_mode.currentIndex()
        num_frames = len(frame_files)
        target_indices = set()
        
        if mode_idx == 1: # N-Keyframe Average
            target_indices = set(np.linspace(0, num_frames - 1, min(5, num_frames), dtype=int))
        elif mode_idx == 2: # Exhaustive Median
            target_indices = set(range(num_frames))
        else: # Best View
            vis = data['visibility']
            pts_mask = data['points_mask']
            vis_tri = vis[:, pts_mask] # (S, N_tri)
            t2d_tri = data['tracks_2d'][:, pts_mask, :]
            
            plate_w = self.camera_setup_data.get('plate_width', 1920)
            plate_h = self.camera_setup_data.get('plate_height', 1080)
            cx, cy = plate_w / 2.0, plate_h / 2.0
            
            for p_idx in range(vis_tri.shape[1]):
                valid_f = np.where(vis_tri[:, p_idx])[0]
                if len(valid_f) == 0: continue
                dists = [((t2d_tri[f, p_idx, 0] - cx)**2 + (t2d_tri[f, p_idx, 1] - cy)**2) for f in valid_f]
                best_f = valid_f[np.argmin(dists)]
                target_indices.add(best_f)
                
        if not target_indices: target_indices.add(0)
        target_indices = sorted(list(target_indices))
        
        # Load the requested depth maps directly from disk
        depth_dict = {}
        for f_idx in target_indices:
            base_name = os.path.splitext(frame_files[f_idx])[0]
            npy_path = os.path.join(depth_proxies_dir, f"{base_name}.npy")
            if os.path.exists(npy_path):
                depth_dict[f_idx] = np.load(npy_path)
                
        if not depth_dict:
            QMessageBox.warning(self, "Error", "Could not load depth sequence. Please regenerate it.")
            return

        try:
            pts_3d = data['points_3d']
            track_2d = data['tracks_2d']
            pts_mask = data['points_mask']
            vis = data['visibility']
            
            from scipy.optimize import least_squares
            def depth_residuals(params, sfm_z, ai_z):
                m, c = params
                return sfm_z - (m * ai_z + c)
            
            frame_data = {}
            for f_idx in target_indices:
                if f_idx not in depth_dict: continue
                depth_map = depth_dict[f_idx]
                h, w = depth_map.shape
                
                R_cam = data['cameras_rot'][f_idx]
                T_cam = data['cameras_trans'][f_idx]
                
                ai_depths = []
                sfm_depths = []
                
                for i in range(len(pts_3d)):
                    if not pts_mask[i] or not vis[f_idx, i]: continue
                    x, y = track_2d[f_idx, i]
                    if x < 0 or y < 0: continue
                    
                    py, px = int(round(y)), int(round(x))
                    if 0 <= px < w and 0 <= py < h:
                        raw_ai_val = depth_map[py, px]
                        ai_z = 1.0 / (raw_ai_val + 1e-6)
                        
                        P_cam = R_cam @ pts_3d[i] + T_cam
                        sfm_z = P_cam[2]
                        
                        ai_depths.append(ai_z)
                        sfm_depths.append(sfm_z)
                        
                if len(ai_depths) > 5:
                    ai_arr = np.array(ai_depths)
                    sfm_arr = np.array(sfm_depths)
                    m_g = np.median(sfm_arr) / (np.median(ai_arr) + 1e-6)
                    c_g = np.median(sfm_arr) - m_g * np.median(ai_arr)
                    
                    res = least_squares(
                        depth_residuals, [m_g, c_g], args=(sfm_arr, ai_arr),
                        loss='soft_l1', f_scale=1.0, bounds=([1e-6, -np.inf], [np.inf, np.inf])
                    )
                    if not res.success or np.isnan(res.x).any():
                        continue
                    m, c = res.x
                    frame_data[f_idx] = {'m': m, 'c': c, 'R': R_cam, 'T': T_cam, 'depth_map': depth_map}
                    
            if not frame_data:
                raise Exception("No valid tracks found to align depth in any target frame.")
                
            blend_factor = self.ai_blend_slider.value() / 100.0
            plate_w = self.camera_setup_data.get('plate_width', 1920)
            plate_h = self.camera_setup_data.get('plate_height', 1080)
            cx, cy = plate_w / 2.0, plate_h / 2.0
            
            for i in range(len(pts_3d)):
                if not pts_mask[i]: continue
                
                valid_f = [f for f in target_indices if vis[f, i] and f in frame_data]
                if not valid_f: continue
                
                world_candidates = []
                dists_to_center = []
                
                for f_idx in valid_f:
                    fdata = frame_data[f_idx]
                    x, y = track_2d[f_idx, i]
                    py, px = int(round(y)), int(round(x))
                    
                    h, w = fdata['depth_map'].shape
                    if 0 <= px < w and 0 <= py < h:
                        raw_ai_val = fdata['depth_map'][py, px]
                        ai_z = 1.0 / (raw_ai_val + 1e-6)
                        target_z = fdata['m'] * ai_z + fdata['c']
                        
                        P_cam = fdata['R'] @ pts_3d[i] + fdata['T']
                        current_z = P_cam[2]
                        new_z = (current_z * (1.0 - blend_factor)) + (target_z * blend_factor)
                        
                        if current_z > 1e-6:
                            P_cam_new = P_cam * (new_z / current_z)
                        else:
                            P_cam_new = P_cam.copy()
                            P_cam_new[2] = new_z
                            
                        P_world_new = fdata['R'].T @ (P_cam_new - fdata['T'])
                        
                        if np.isnan(P_world_new).any(): continue
                        
                        world_candidates.append(P_world_new)
                        dists_to_center.append(math.hypot(x - cx, y - cy))
                        
                if not world_candidates: continue
                
                cand_arr = np.array(world_candidates)
                if mode_idx == 0: # Best View
                    best_idx = np.argmin(dists_to_center)
                    pts_3d[i] = cand_arr[best_idx]
                elif mode_idx == 1: # N-Keyframe Average
                    pts_3d[i] = np.mean(cand_arr, axis=0)
                else: # Exhaustive Median
                    pts_3d[i] = np.median(cand_arr, axis=0)
                    
            data['points_3d'] = pts_3d
            np.savez(data_path, **data)
            
            self.solve_viewport.load_solve_data(data_path, self.camera_setup_data, reset_view=False)
            self.load_2d_solve_data(data_path)
            
            self.viewing_raw = False
            if hasattr(self, 'btn_toggle_view'):
                self.btn_toggle_view.setStyleSheet("")
                self.btn_toggle_view.setText("Preview Raw Data")
                
            QMessageBox.information(self, "AI Smoothing", "Point cloud spatially smoothed using AI depth priors!")
            
            
        except Exception as e:
            QMessageBox.critical(self, "Alignment Error", str(e))


    def load_2d_solve_data(self, data_path):
        try:
            data = np.load(data_path)
            if 'tracks_2d' in data and 'visibility' in data:
                # Update the 2D viewer with the sequence FIRST so it doesn't overwrite our decimated data
                cs = getattr(self, 'color_space', CS_LINEAR_SRGB)
                self.solve_2d_viewport.update_sequence(self.exr_files, cs, self.project_dir)
                
                self.solve_2d_viewport.track_data = data['tracks_2d']
                self.solve_2d_viewport.track_vis = data['visibility']
                
                if 'points_mask' in data and 'points_error' in data:
                    self.solve_2d_points_mask = data['points_mask']
                    self.solve_2d_points_error = data['points_error']
                    self.update_2d_solve_colors()
                    
                self.solve_2d_viewport.on_frame_changed(0)
        except Exception as e:
            print(f"Failed to load 2D solve data: {e}")

    def on_track_picked(self, track_id):
        if track_id in self.selected_tracks:
            self.selected_tracks.remove(track_id)
        else:
            self.selected_tracks.append(track_id)
        self.update_selection_ui()
        
    def clear_selection(self):
        self.selected_tracks = []
        self.active_constraint = None
        self.update_selection_ui()
        
    def update_selection_ui(self):
        self.solve_viewport.list_selection.clear()
        for t_id in self.selected_tracks:
            self.solve_viewport.list_selection.addItem(f"Track ID: {t_id}")
            
        self.solve_viewport.selected_tracks = self.selected_tracks
        self.solve_viewport.active_constraint = getattr(self, 'active_constraint', None)
        self.solve_viewport.update_error_threshold()
        self.update_2d_solve_colors()

    def update_2d_solve_colors(self, slider_val=None):
        if not hasattr(self, 'solve_2d_points_mask') or not hasattr(self, 'solve_2d_points_error'):
            return
            
        if slider_val is None:
            slider_val = self.solve_viewport.slider_error.value()
            
        threshold = slider_val / 10.0
        colors = []
        for i in range(len(self.solve_2d_points_mask)):
            is_valid = self.solve_2d_points_mask[i]
            err = self.solve_2d_points_error[i]
            
            if i in self.selected_tracks:
                colors.append(QColor(255, 255, 0, 255)) # Yellow (Selected)
            elif is_valid and err <= threshold:
                colors.append(QColor(0, 255, 0, 200)) # Green (Valid)
            else:
                colors.append(QColor(255, 0, 0, 100)) # Red (Filtered/Rejected)
                
        self.solve_2d_viewport.track_colors = colors
        
        # Redraw current frame
        current_frame = self.solve_viewport.slider_frame.value()
        self.solve_2d_viewport.on_frame_changed(current_frame)

    def load_orientation_constraints(self):
        if not hasattr(self, 'project_dir') or not self.project_dir: return
        path = os.path.join(self.project_dir, 'orientation.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.orientation_constraints = json.load(f)
            except Exception:
                self.orientation_constraints = []
        else:
            self.orientation_constraints = []
        self.update_orientation_ui()

    def save_orientation_constraints(self):
        if not hasattr(self, 'project_dir') or not self.project_dir: return
        path = os.path.join(self.project_dir, 'orientation.json')
        with open(path, 'w') as f:
            json.dump(self.orientation_constraints, f)

    def update_orientation_ui(self):
        self.solve_viewport.list_orientation.clear()
        for c in self.orientation_constraints:
            desc = f"{c['type']} - Tracks: {c['tracks']}"
            if c['type'] == 'Scale (2 pts)':
                desc += f" ({c['scale_dist']}m)"
            self.solve_viewport.list_orientation.addItem(desc)

    def add_orientation_constraint(self, ctype, scale_dist):
        if not self.selected_tracks:
            QMessageBox.warning(self, "Warning", "Please select tracks first.")
            return
            
        tracks = list(self.selected_tracks)
        
        # Validation
        if ctype == "Origin (1 pt)":
            if len(tracks) != 1:
                QMessageBox.warning(self, "Warning", "Origin requires exactly 1 point.")
                return
            if any(c['type'] == ctype for c in self.orientation_constraints):
                QMessageBox.warning(self, "Warning", "Only one Origin constraint is allowed.")
                return
        elif ctype == "Scale (2 pts)":
            if len(tracks) != 2:
                QMessageBox.warning(self, "Warning", "Scale requires exactly 2 points.")
                return
            if any(c['type'] == ctype for c in self.orientation_constraints):
                QMessageBox.warning(self, "Warning", "Only one Scale constraint is allowed.")
                return
        elif ctype == "Ground Plane (3+ pts)":
            if len(tracks) < 3:
                QMessageBox.warning(self, "Warning", "Ground Plane requires 3 or more points.")
                return
            if any(c['type'] == ctype for c in self.orientation_constraints):
                QMessageBox.warning(self, "Warning", "Only one Ground Plane constraint is allowed.")
                return
        elif "Plane" in ctype:
            if len(tracks) < 2:
                QMessageBox.warning(self, "Warning", f"{ctype} requires 2 or more points.")
                return
        elif "Line" in ctype:
            if len(tracks) != 2:
                QMessageBox.warning(self, "Warning", f"{ctype} requires exactly 2 points.")
                return
                
        self.orientation_constraints.append({
            'type': ctype,
            'tracks': tracks,
            'scale_dist': scale_dist
        })
        self.save_orientation_constraints()
        self.update_orientation_ui()
        self.clear_selection()
        
    def del_orientation_constraint(self, idx):
        if idx < 0 or idx >= len(self.orientation_constraints): return
        self.orientation_constraints.pop(idx)
        self.save_orientation_constraints()
        self.update_orientation_ui()
        
    def select_orientation_constraint(self, idx):
        if idx < 0 or idx >= len(self.orientation_constraints): return
        c = self.orientation_constraints[idx]
        self.selected_tracks = list(c['tracks'])
        self.active_constraint = c
        self.update_selection_ui()

    def apply_orientation_constraints(self, reset_view=True, return_data=False):
        if not hasattr(self, 'project_dir') or not self.project_dir: return
        
        data_path = os.path.join(self.project_dir, 'solve_data.npz')
        raw_data_path = os.path.join(self.project_dir, 'solve_data_raw.npz')
        
        # Ensure raw data exists
        if not os.path.exists(raw_data_path):
            if not os.path.exists(data_path): return
            import shutil
            shutil.copy2(data_path, raw_data_path)
            
        data = dict(np.load(raw_data_path))
        pts_3d = data['points_3d'].copy()
        
        if not self.orientation_constraints:
            if return_data: return data
            # If no constraints, just restore raw
            np.savez(data_path, **data)
            self.solve_viewport.load_solve_data(data_path, self.camera_setup_data, reset_view=reset_view)
            self.load_2d_solve_data(data_path)
            self.solve_viewport.update_error_threshold()
            return
            
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        
        # Parameters: [s, tx, ty, tz, rx, ry, rz]
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        def residuals(x):
            s = x[0]
            t = x[1:4]
            r = x[4:7]
            
            has_scale = any(c['type'] == "Scale (2 pts)" for c in self.orientation_constraints)
            if not has_scale:
                s = 1.0
                
            has_rot = any(c['type'] not in ["Origin (1 pt)", "Scale (2 pts)"] for c in self.orientation_constraints)
            if not has_rot:
                r = np.zeros(3)
                
            R = Rotation.from_rotvec(r).as_matrix()
            
            # Calculate camera centers to enforce "up" direction
            cam_centers = []
            for i in range(len(data['cameras_rot'])):
                R_cam = data['cameras_rot'][i]
                T_cam = data['cameras_trans'][i]
                if not (np.allclose(R_cam, np.eye(3)) and np.allclose(T_cam, np.zeros(3))):
                    cam_centers.append(-R_cam.T @ T_cam)
            cam_centers = np.array(cam_centers)
            if len(cam_centers) > 0:
                cam_centers_t = s * (cam_centers @ R.T) + t
            else:
                cam_centers_t = np.array([[0.0, 1.0, 0.0]])
                
            has_ground = any(c['type'] in ["Ground Plane (3+ pts)", "XZ Plane (2+ pts)"] for c in self.orientation_constraints)
            
            res = []
            for c in self.orientation_constraints:
                tracks = c['tracks']
                ctype = c['type']
                
                # Filter tracks to ensure they exist in raw data
                valid_tracks = [tr for tr in tracks if tr < len(pts_3d)]
                if not valid_tracks: continue
                
                P = pts_3d[valid_tracks]
                P_t = s * (P @ R.T) + t
                
                if ctype == "Origin (1 pt)":
                    res.extend(P_t[0].tolist())
                elif ctype == "Scale (2 pts)" and len(P_t) == 2:
                    dist = np.linalg.norm(P_t[0] - P_t[1])
                    res.append(dist - c['scale_dist'])
                elif ctype == "Ground Plane (3+ pts)" or ctype == "XZ Plane (2+ pts)":
                    res.extend(P_t[:, 1].tolist())
                elif ctype == "XY Plane (2+ pts)":
                    res.extend(P_t[:, 2].tolist())
                elif ctype == "YZ Plane (2+ pts)":
                    res.extend(P_t[:, 0].tolist())
                elif ctype == "X Line (2 pts)" and len(P_t) == 2:
                    res.append(P_t[0, 1] - P_t[1, 1])
                    res.append(P_t[0, 2] - P_t[1, 2])
                elif ctype == "Y Line (2 pts)" and len(P_t) == 2:
                    res.append(P_t[0, 0] - P_t[1, 0])
                    res.append(P_t[0, 2] - P_t[1, 2])
                elif ctype == "Z Line (2 pts)" and len(P_t) == 2:
                    res.append(P_t[0, 0] - P_t[1, 0])
                    res.append(P_t[0, 1] - P_t[1, 1])
            
            if has_ground:
                avg_cam_y = np.mean(cam_centers_t[:, 1])
                penalty = min(0.0, avg_cam_y * 1000.0)
                res.append(penalty)
                    
            # Regularization to prevent under-determined Jacobian failure
            res.append((x[0] - 1.0) * 1e-4)
            res.extend((x[1:4] * 1e-4).tolist())
            res.extend((x[4:7] * 1e-4).tolist())
            
            return np.array(res)
            
        x0_flip = np.array([1.0, 0.0, 0.0, 0.0, np.pi, 0.0, 0.0])
        res_0 = np.sum(np.square(residuals(x0)))
        res_flip = np.sum(np.square(residuals(x0_flip)))
        
        if res_flip < res_0:
            x0 = x0_flip
            
        res_opt = least_squares(residuals, x0, method='lm')
        x_opt = res_opt.x
        
        # Manually enforce locks on the final output just in case
        has_scale = any(c['type'] == "Scale (2 pts)" for c in self.orientation_constraints)
        if not has_scale:
            x_opt[0] = 1.0
            
        has_rot = any(c['type'] not in ["Origin (1 pt)", "Scale (2 pts)"] for c in self.orientation_constraints)
        if not has_rot:
            x_opt[4:7] = 0.0
            
        s = x_opt[0]
        t = x_opt[1:4]
        R = Rotation.from_rotvec(x_opt[4:7]).as_matrix()
        
        # Apply transformation
        data['points_3d'] = s * (data['points_3d'] @ R.T) + t
        
        for i in range(len(data['cameras_rot'])):
            R_cam = data['cameras_rot'][i]
            T_cam = data['cameras_trans'][i]
            
            if np.allclose(R_cam, np.eye(3)) and np.allclose(T_cam, np.zeros(3)):
                continue
                
            R_cam_new = R_cam @ R.T
            T_cam_new = s * T_cam - R_cam_new @ t
            
            data['cameras_rot'][i] = R_cam_new
            data['cameras_trans'][i] = T_cam_new
            
        if return_data:
            return data
            
        np.savez(data_path, **data)
        self.solve_viewport.load_solve_data(data_path, self.camera_setup_data, reset_view=reset_view)
        self.load_2d_solve_data(data_path)
        self.solve_viewport.update_error_threshold()

    def run_solve(self):
        if not hasattr(self, 'project_dir') or not self.project_dir:
            QMessageBox.warning(self, "Warning", "Please load a project first.")
            return
            
        if self.daemon_process and self.daemon_process.is_alive():
            self.shutdown_daemon()
            self.daemon_process.join(timeout=10)
            
        if not hasattr(self, 'camera_setup_data') or not self.camera_setup_data:
            QMessageBox.warning(self, "Warning", "Please complete Camera Setup first.")
            self.tabs.setCurrentWidget(self.camera_setup_tab)
            return
            
        self.btn_start_solve.setEnabled(False)
        self.solve_progress.setValue(0)
        self.solve_progress.show()
        self.lbl_solve_status.setText("Initializing Solver Worker...")
        
        self.cmd_queue = multiprocessing.Queue()
        self.res_queue = multiprocessing.Queue()
        
        tracks_path = os.path.join(self.project_dir, 'tracks.npz')
        proxies_dir = os.path.join(self.project_dir, 'proxies')
        model_path = os.path.join(MODELS_DIR, "vggsfm", "vggsfm_v2_0_0.bin")
        
        self.daemon_process = multiprocessing.Process(
            target=run_vggsfm_worker, 
            args=(self.project_dir, proxies_dir, tracks_path, self.camera_setup_data, model_path, self.res_queue)
        )
        self.daemon_process.start()
        self.timer.start(50)

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
        os.makedirs(os.path.join(self.project_dir, 'depth_proxies'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'exports'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'user_masks'), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, 'user_masks_source'), exist_ok=True)
        
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
            "native_sizes": getattr(self, 'native_sizes', []),
            "camera_setup_data": getattr(self, 'camera_setup_data', {})
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
            os.makedirs(os.path.join(self.project_dir, 'depth_proxies'), exist_ok=True)
            
            self.exr_files = data.get("exr_files", [])
            self.color_space = data.get("color_space", CS_LINEAR_SRGB)
            self.proxy_res = data.get("proxy_res", 1536)
            self.native_sizes = data.get("native_sizes", [])
            self.camera_setup_data = data.get("camera_setup_data", {})
            
            if self.camera_setup_data:
                cam_name = self.camera_setup_data.get("ui_camera_name", "")
                if cam_name:
                    idx = self.combo_cameras.findText(cam_name)
                    if idx >= 0:
                        self.combo_cameras.setCurrentIndex(idx)
                
                self.chk_assume_no_crop.setChecked(self.camera_setup_data.get("ui_assume_no_crop", True))
                self.spin_orig_w.setValue(self.camera_setup_data.get("ui_orig_w", 1920))
                self.spin_orig_h.setValue(self.camera_setup_data.get("ui_orig_h", 1080))
                self.chk_auto_focal.setChecked(self.camera_setup_data.get("auto_estimate_focal", True))
                self.spin_focal.setValue(self.camera_setup_data.get("ui_focal_val", 50.0))
                
            cs_idx = self.combo_colorspace.findText(self.color_space)
            if cs_idx >= 0:
                self.combo_colorspace.setCurrentIndex(cs_idx)
                
            res_idx = self.combo_res.findData(self.proxy_res)
            if res_idx >= 0:
                self.combo_res.setCurrentIndex(res_idx)
                
            self.btn_import_exr.setEnabled(True)
            self.btn_regen_proxies.setEnabled(True)
            self.btn_gen_depth.setEnabled(True)
            self.lbl_file.setText(f"Project: {self.project_root} | Frames: {len(self.exr_files)}")
            
            if hasattr(self, 'solve_viewport'):
                out_data_path = os.path.join(self.project_dir, 'solve_data.npz')
                if os.path.exists(out_data_path) and self.camera_setup_data:
                    self.solve_2d_viewport.show()
                    self.solve_viewport.show()
                    self.solve_viewport.load_solve_data(out_data_path, self.camera_setup_data)
                    self.load_2d_solve_data(out_data_path)
                    self.load_orientation_constraints()
                    self.solve_progress.hide()
                    self.btn_start_solve.setText("Re-Start 3D Solver (Overwrite)")
                    self.lbl_solve_status.hide()
            
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
                self.btn_gen_depth.setEnabled(False)
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
        self.btn_gen_depth.setEnabled(True)
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
            self.btn_gen_depth.setEnabled(True)
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
        self.btn_gen_depth.setEnabled(False)
        self.lbl_file.setText("Regenerating Proxies... Please wait.")
        
        self.proxy_worker = ProxyGeneratorWorker(self.exr_files, self.project_dir, self.color_space, self.proxy_res)
        self.proxy_worker.progress.connect(self.on_proxy_progress)
        self.proxy_worker.finished.connect(self.on_proxy_finished)
        self.proxy_worker.error.connect(self.on_proxy_error)
        self.proxy_worker.start()

    def generate_depth_sequence(self):
        if not self.exr_files or not self.project_dir:
            QMessageBox.warning(self, "Warning", "Please load a sequence and generate proxies in the Setup tab first.")
            return
            
        proxies_dir = os.path.join(self.project_dir, 'proxies')
        depth_proxies_dir = os.path.join(self.project_dir, 'depth_proxies')
        
        if not os.path.exists(proxies_dir) or not os.listdir(proxies_dir):
            QMessageBox.warning(self, "Warning", "Please generate image proxies first.")
            return
            
        os.makedirs(depth_proxies_dir, exist_ok=True)
        
        self.btn_import_exr.setEnabled(False)
        self.btn_regen_proxies.setEnabled(False)
        self.btn_gen_depth.setEnabled(False)
        self.lbl_file.setText("Generating Depth Sequence... Please wait. This may take a while.")
        
        self.setup_viewer.loading_overlay.setText("Generating AI Depth Sequence...\nPlease wait.")
        self.setup_viewer.loading_overlay.resize(self.setup_viewer.view.size())
        self.setup_viewer.loading_overlay.show()
        
        self.ai_depth_seq_worker = AIDepthSequenceWorker(proxies_dir, depth_proxies_dir)
        self.ai_depth_seq_worker.progress.connect(self.on_proxy_progress)
        self.ai_depth_seq_worker.finished.connect(self.on_depth_seq_finished)
        self.ai_depth_seq_worker.error.connect(self.on_depth_seq_error)
        self.ai_depth_seq_worker.start()

    def on_depth_seq_finished(self):
        self.setup_viewer.loading_overlay.hide()
        self.btn_import_exr.setEnabled(True)
        self.btn_regen_proxies.setEnabled(True)
        self.btn_gen_depth.setEnabled(True)
        self.lbl_file.setText(f"Project: {self.project_root} | Depth Sequence Generated.")
        QMessageBox.information(self, "Success", "Depth sequence generated successfully!")

    def on_depth_seq_error(self, err_str):
        self.setup_viewer.loading_overlay.hide()
        self.btn_import_exr.setEnabled(True)
        self.btn_regen_proxies.setEnabled(True)
        self.btn_gen_depth.setEnabled(True)
        self.lbl_file.setText("Depth generation failed.")
        QMessageBox.critical(self, "Error", f"Failed to generate depth sequence: {err_str}")

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

        elif self.tabs.widget(index) == getattr(self, 'proxy_geo_tab', None):
            self.proxy_geo_2d_viewport.update_sequence(self.exr_files, getattr(self, 'color_space', CS_LINEAR_SRGB), self.project_dir)
            if self.project_dir:
                solve_path = os.path.join(self.project_dir, 'solve_data.npz')
                if os.path.exists(solve_path):
                    self.proxy_geo_3d_viewport.load_solve_data(solve_path, self.proxy_res, self.camera_setup_data)
                mesh_path = os.path.join(self.project_dir, 'proxy_geo.obj')
                if os.path.exists(mesh_path):
                    self.proxy_geo_3d_viewport.load_proxy_mesh(mesh_path)

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
                    
                elif status == "solve_progress":
                    self.solve_progress.show()
                    self.solve_progress.setValue(res.get("value"))
                    self.lbl_solve_status.setText(res.get("message", "Running Bundle Adjustment..."))
                    
                elif status == "solve_done":
                    self.solve_progress.setValue(100)
                    if hasattr(self, 'btn_start_solve'):
                        self.btn_start_solve.setEnabled(True)
                    self.lbl_solve_status.setText("3D Solve Complete!")
                    
                    out_data_path = os.path.join(self.project_dir, 'solve_data.npz')
                    raw_data_path = os.path.join(self.project_dir, 'solve_data_raw.npz')
                    if os.path.exists(raw_data_path):
                        os.remove(raw_data_path)
                        
                    if os.path.exists(out_data_path):
                        self.solve_2d_viewport.show()
                        self.solve_viewport.show()
                        self.solve_viewport.load_solve_data(out_data_path, self.camera_setup_data)
                        self.load_2d_solve_data(out_data_path)
                        self.load_orientation_constraints()
                        self.solve_progress.hide()
                        self.btn_start_solve.setText("Re-Start 3D Solver (Overwrite)")
                        self.lbl_solve_status.hide()
                        
                    QMessageBox.information(self, "Finished", "3D Solve Complete!")
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
