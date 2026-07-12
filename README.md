# AI Camera Solver

An automated, AI-assisted 3D Camera Tracking and Scene Reconstruction tool designed for VFX and compositing workflows.

AI Camera Solver utilizes cutting edge AI models (such as VGGSfM for camera tracking, SAM-2 for intelligent masking, and monocular depth estimators) to automatically track camera movement and build low-resolution proxy geometry of an environment directly from 2D footage.

It runs locally with a fully embedded, isolated Python runtime to avoid dependency conflicts on your system.

## Features
- **Auto-Masking:** Uses Segment Anything (SAM-2) to track and mask dynamic elements (actors, vehicles, etc.) out of the solve.
- **AI Camera Tracking:** Employs VGGSfM to solve complex, featureless, or extreme-motion camera tracks where traditional photogrammetry fails.
- **Proxy Geometry Generation:** Generates a 3D topological mesh (Proxy Geo) using AI monocular depth estimation perfectly aligned to the solved 3D camera.
- **Intelligent Decimation:** Refine and smooth generated mesh data using Quadric Error Metric (QEM) decimation and Taubin smoothing directly in the app.
- **VFX Exporting:** Natively exports tracking and scene data to automated Nuke (`.nk`) scripts and Blender macros.

---

## Installation 

This repository ships with a fully functional, lightweight Embedded Python runtime (`python/python.exe`). You **do not** need to install Python on your system, and this software will not interfere with any existing Python environments you have.

To set up the software, you simply need to clone the repository and tell the embedded Python interpreter to download the heavy AI libraries.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Fobdor/AICameraSolver.git
cd AICameraSolver
```

### 2. Install Dependencies (Into Embedded Python)
To install the required AI models and packages (like PyTorch, Open3D, and OpenCV), you must use the embedded Python executable included in the repo. 

Run the following command from the root of the project:
```bat
.\python\python.exe -m pip install -r requirements.txt
```
*(Note: This download will be several gigabytes in size.)*

### 3. Run the Application
Once the installation finishes, you can launch the application by simply double-clicking the `launch.bat` file, or running:
```bat
.\launch.bat
```
