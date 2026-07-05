import sys
import os
import multiprocessing
import time
import queue
import functools
import numpy as np
import OpenImageIO as oiio
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QProgressBar, QLabel, 
                               QFileDialog, QTabWidget, QMessageBox, QGraphicsView,
                               QGraphicsScene, QSlider, QComboBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt

# Color Space Constants
CS_LINEAR_SRGB = "Linear sRGB"
CS_ACESCG = "ACEScg"

def apply_color_space(rgb_array, color_space):
    """Applies color transformation mathematically to the RGB float array."""
    if color_space == CS_ACESCG:
        # ACEScg (AP1) to Linear sRGB (D65) matrix
        acescg_to_srgb = np.array([
            [1.705051, -0.621861, -0.083190],
            [-0.130256, 1.140802, -0.010546],
            [-0.024007, -0.128967, 1.152974]
        ])
        # Reshape for dot product: (H*W, 3) dot (3, 3).T
        shape = rgb_array.shape
        flat_rgb = rgb_array.reshape(-1, 3)
        flat_linear = np.dot(flat_rgb, acescg_to_srgb.T)
        rgb_array = flat_linear.reshape(shape)

    # Assume Linear sRGB state at this point. Apply standard sRGB gamma curve.
    linear = np.clip(rgb_array, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 
                    linear * 12.92, 
                    1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return srgb

@functools.lru_cache(maxsize=50)
def fetch_qpixmap(frame_path, color_space):
    """Reads EXR, applies color transform, normalizes to uint8, returns QPixmap."""
    try:
        input_image = oiio.ImageInput.open(frame_path)
        if not input_image:
            print(f"Error opening {frame_path}")
            return QPixmap()
        
        image_data = input_image.read_image()
        input_image.close()

        # Extract RGB channels if >= 3
        if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
            rgb_array = image_data[:, :, :3]
        elif len(image_data.shape) == 2: # Grayscale
            rgb_array = np.stack((image_data,)*3, axis=-1)
        else:
            return QPixmap()
            
        # Apply color transformation
        srgb_array = apply_color_space(rgb_array, color_space)
            
        # Convert to 0-255 uint8
        rgb_uint8 = np.clip(srgb_array * 255.0, 0, 255).astype(np.uint8)
        
        height, width, channel = rgb_uint8.shape
        bytes_per_line = 3 * width
        
        # Format_RGB888 expects data memory to be kept alive, so we copy inside QImage
        qimg = QImage(rgb_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)
    except Exception as e:
        print(f"Failed to fetch qpixmap: {e}")
        return QPixmap()


class SequenceViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.exr_files = []
        self.color_space = CS_LINEAR_SRGB
        
        layout = QVBoxLayout(self)
        
        # Viewer
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.view)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_frame_changed)
        
        self.lbl_frame = QLabel("Frame: 0/0")
        
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.lbl_frame)
        layout.addLayout(ctrl_layout)
        
        self.pixmap_item = None

    def update_sequence(self, file_list, color_space):
        self.exr_files = file_list
        self.color_space = color_space
        if self.exr_files:
            self.slider.setMaximum(len(self.exr_files) - 1)
            # If already at a frame, force update, else go to 0
            curr = self.slider.value()
            self.slider.setEnabled(True)
            self.on_frame_changed(curr)
        else:
            self.slider.setEnabled(False)
            self.scene.clear()
            self.pixmap_item = None
            self.lbl_frame.setText("Frame: 0/0")

    def set_color_space(self, color_space):
        self.color_space = color_space
        if self.exr_files:
            self.on_frame_changed(self.slider.value())

    def on_frame_changed(self, index):
        if not self.exr_files or index < 0 or index >= len(self.exr_files):
            return
            
        frame_path = self.exr_files[index]
        pixmap = fetch_qpixmap(frame_path, self.color_space)
        
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pixmap)
        else:
            self.pixmap_item.setPixmap(pixmap)
            
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.lbl_frame.setText(f"Frame: {index + 1}/{len(self.exr_files)}")
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# Dummy worker function
def worker_process(progress_queue, exr_files, color_space):
    print(f"[Worker] Started with executable: {sys.executable}")
    print(f"[Worker] Target sequence length: {len(exr_files)} frames")
    print(f"[Worker] Target color space for inference: {color_space}")
    
    # Simulate a task where it iterates over the file list
    num_frames = len(exr_files)
    if num_frames == 0:
        progress_queue.put("DONE")
        return
        
    for i in range(num_frames):
        time.sleep(1.0) # simulate processing a frame
        progress = int(((i + 1) / num_frames) * 100)
        progress_queue.put(progress)
        
    progress_queue.put("DONE")
    print("[Worker] Task finished.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Camera Tracker")
        self.resize(1024, 768)
        
        self.worker = None
        self.progress_queue = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_queue)
        
        self.exr_files = []

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Setup Tab
        self.setup_tab = QWidget()
        self.tabs.addTab(self.setup_tab, "Setup")
        self.init_setup_tab()
        
        # Masking Tab
        self.masking_tab = QWidget()
        self.tabs.addTab(self.masking_tab, "Masking")
        self.init_masking_tab()

    def init_setup_tab(self):
        layout = QVBoxLayout()
        
        # Top bar: Load button + Color Space combo
        top_layout = QHBoxLayout()
        
        btn_load = QPushButton("Load EXR Sequence Folder")
        btn_load.clicked.connect(self.load_exr)
        top_layout.addWidget(btn_load)
        
        top_layout.addWidget(QLabel("Color Space:"))
        self.combo_colorspace = QComboBox()
        self.combo_colorspace.addItems([CS_LINEAR_SRGB, CS_ACESCG])
        self.combo_colorspace.currentIndexChanged.connect(self.on_color_space_changed)
        top_layout.addWidget(self.combo_colorspace)
        
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        self.lbl_file = QLabel("No sequence loaded.")
        layout.addWidget(self.lbl_file)
        
        # Setup Viewer
        self.setup_viewer = SequenceViewerWidget()
        layout.addWidget(self.setup_viewer, stretch=1)
        
        # Progress and Controls
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Solver")
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_worker)
        self.btn_cancel.setEnabled(False)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_cancel)
        layout.addLayout(ctrl_layout)
        
        self.setup_tab.setLayout(layout)

    def init_masking_tab(self):
        layout = QVBoxLayout()
        # Masking Viewer (scrubbable)
        self.masking_viewer = SequenceViewerWidget()
        layout.addWidget(self.masking_viewer)
        self.masking_tab.setLayout(layout)

    def get_color_space(self):
        return self.combo_colorspace.currentText()

    def on_color_space_changed(self):
        cs = self.get_color_space()
        
        self.setup_viewer.set_color_space(cs)
        
        # Update masking viewer
        self.masking_viewer.set_color_space(cs)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def load_exr(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select EXR Sequence Folder", "")
        if dir_path:
            try:
                files = [f for f in os.listdir(dir_path) if f.lower().endswith('.exr')]
                if not files:
                    QMessageBox.warning(self, "Warning", "No EXR files found in the selected folder.")
                    return
                
                files.sort()
                self.exr_files = [os.path.join(dir_path, f) for f in files]
                
                self.lbl_file.setText(f"Loaded {len(self.exr_files)} frames from: {os.path.basename(dir_path)}")
                
                cs = self.get_color_space()
                
                # Update viewers
                self.setup_viewer.update_sequence(self.exr_files, cs)
                self.masking_viewer.update_sequence(self.exr_files, cs)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load EXR sequence: {str(e)}")

    def start_worker(self):
        if self.worker is not None and self.worker.is_alive():
            return
            
        if not self.exr_files:
            QMessageBox.warning(self, "Warning", "Please load an EXR sequence first.")
            return
            
        self.progress_bar.setValue(0)
        self.progress_queue = multiprocessing.Queue()
        
        cs = self.get_color_space()
        
        self.worker = multiprocessing.Process(target=worker_process, args=(self.progress_queue, self.exr_files, cs))
        self.worker.start()
        
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.timer.start(100) # Poll every 100ms
        
    def cancel_worker(self):
        if self.worker and self.worker.is_alive():
            print("Terminating worker process...")
            self.worker.terminate()
            self.worker.join() # Wait for it to die
            print("Worker terminated.")
            self.progress_bar.setValue(0)
            self.cleanup_worker()
            
    def poll_queue(self):
        if self.progress_queue is None:
            return
            
        while not self.progress_queue.empty():
            try:
                msg = self.progress_queue.get_nowait()
                if msg == "DONE":
                    self.progress_bar.setValue(100)
                    self.cleanup_worker()
                    QMessageBox.information(self, "Finished", "Task Completed Successfully!")
                else:
                    self.progress_bar.setValue(int(msg))
            except queue.Empty:
                pass
                
        # If worker died unexpectedly
        if self.worker and not self.worker.is_alive():
            if self.progress_bar.value() < 100 and self.progress_bar.value() > 0:
                # We check > 0 to not fire right after cancel
                if not self.btn_cancel.isEnabled():
                    return
                print("Worker died unexpectedly.")
                self.cleanup_worker()

    def cleanup_worker(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None

if __name__ == '__main__':
    # Print system diagnostics to ensure we're in the portable env
    expected_exe = os.path.abspath(os.path.join(os.getcwd(), 'python', 'python.exe'))
    if sys.executable.lower() != expected_exe.lower():
        print(f"WARNING: Executable mismatch!\nExpected: {expected_exe}\nGot: {sys.executable}")
        
    # Multiprocessing configuration
    multiprocessing.set_start_method('spawn')
    multiprocessing.set_executable(sys.executable)

    # Launch PySide6 UI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
