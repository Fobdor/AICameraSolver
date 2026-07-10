import os
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from PIL import Image

class AIDepthWorker(QThread):
    progress = Signal(int)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, frame_path):
        super().__init__()
        self.frame_path = frame_path

    def run(self):
        try:
            self.progress.emit(10)
            from transformers import pipeline
            
            self.progress.emit(20)
            # This will automatically download to ~/.cache/huggingface on first run
            pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=0 if torch.cuda.is_available() else -1)
            
            self.progress.emit(50)
            img = Image.open(self.frame_path).convert('RGB')
            w, h = img.size
            result = pipe(img)
            
            self.progress.emit(80)
            depth_tensor = result['predicted_depth']
            depth_np = depth_tensor.squeeze().cpu().numpy()
            
            import cv2
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
            
            self.progress.emit(100)
            self.finished.emit(depth_np)
        except Exception as e:
            self.error.emit(f"AI Depth Error: {str(e)}")
