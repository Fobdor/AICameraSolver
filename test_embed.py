import sys
import time
import win32gui
import open3d as o3d
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtGui import QWindow
from PySide6.QtCore import QTimer

app = QApplication(sys.argv)
main_w = QWidget()
main_w.setWindowTitle('Test Embedding')
main_w.resize(800, 600)
layout = QVBoxLayout(main_w)

vis = o3d.visualization.Visualizer()
vis.create_window('O3D_Embedded', width=800, height=600, visible=False)

# Let it spin briefly to ensure window is created
for _ in range(10):
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.01)

hwnd = win32gui.FindWindow(None, 'O3D_Embedded')
print('HWND:', hwnd)

if hwnd:
    window = QWindow.fromWinId(hwnd)
    widget3d = QWidget.createWindowContainer(window)
    layout.addWidget(widget3d)
else:
    print('Failed to find HWND')

main_w.show()

timer = QTimer()
timer.timeout.connect(lambda: (vis.poll_events(), vis.update_renderer()))
timer.start(16)

QTimer.singleShot(2000, app.quit) # close after 2 sec
sys.exit(app.exec())
