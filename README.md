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

This repository provides automated installation scripts for Windows, Mac, and Linux. These scripts create isolated Python environments so that this software will not interfere with any existing Python environments on your system.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Fobdor/AICameraSolver.git
cd AICameraSolver
```

### 2. Install Dependencies
To install the required AI models and packages (like PyTorch, Open3D, and OpenCV), simply run the automated setup script for your OS. This will download everything into a local, isolated environment.

**On Windows:**
Double-click `install.bat` or run:
```bat
.\install.bat
```
*(This automatically downloads an isolated Embedded Python runtime and installs the dependencies).*

**On Mac / Linux:**
Run the install script from your terminal:
```bash
chmod +x install.sh
./install.sh
```
*(This creates an isolated virtual environment (`venv`) using your system's Python 3 and installs the dependencies).*

### 3. Run the Application
Once the installation finishes, you can launch the application:

**On Windows:**
Double-click `launch.bat` or run:
```bat
.\launch.bat
```

**On Mac / Linux:**
```bash
chmod +x launch.sh
./launch.sh
```
