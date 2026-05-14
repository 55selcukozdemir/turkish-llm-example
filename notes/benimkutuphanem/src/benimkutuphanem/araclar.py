import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np

import qdarkstyle
import torch

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial import Delaunay


def selam():
    return "Merhaba"

def visualize_tensor_scatter_dict(view_widget, tensor_dict, dot_size=10, threshold_low=0.0, threshold_high=0.0, x_spacing=50):
    cmap = sns.color_palette("magma_r", as_cmap=True)

    all_scatter = []
    a = 0
    for i, tensor in enumerate(tensor_dict.values()):
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        
        if len(tensor.shape) == 1:
            tensor = np.expand_dims(tensor, axis=-1)
            tensor = np.expand_dims(tensor, axis=-1)
        
        if len(tensor.shape) == 2:
            tensor = np.expand_dims(tensor, axis=-1)

        mask = (tensor >= threshold_low) & (tensor <= threshold_high)
        indices = np.argwhere(mask)
       
        values = tensor[mask]
        if values.size == 0:
            continue
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        colors = cmap(norm(values))

        # 🔥 X ekseninde offset
        indices = indices.copy()
        a += 1
        indices[:, 0] += a * x_spacing

        scatter = gl.GLScatterPlotItem(
            pos=indices,
            color=colors,
            size=dot_size,
            pxMode=True
        )
        

        view_widget.addItem(scatter)
        all_scatter.append(scatter)

    if len(all_scatter) == 0:
        print("Görselleştirilecek veri bulunamadı.")
        return None

    return all_scatter

class ViewPresetButtonGroup(pg.QtWidgets.QWidget):

    def __init__(self, opengl_view : gl.GLViewWidget):
        super().__init__()
        self.opengl_view = opengl_view

        layout = pg.QtWidgets.QHBoxLayout(self)
        
        self.button_side = pg.QtWidgets.QPushButton(text="yan")
        self.button_front = pg.QtWidgets.QPushButton(text="ön")
        self.button_top = pg.QtWidgets.QPushButton(text="üst")

        self.button_side.clicked.connect(lambda :self.set_camera_angle("side"))
        self.button_front.clicked.connect(lambda: self.set_camera_angle("front"))
        self.button_top.clicked.connect(lambda: self.set_camera_angle("top"))

        layout.addWidget(self.button_side)
        layout.addWidget(self.button_front)
        layout.addWidget(self.button_top)


    def set_camera_angle(self, position):

        if position == "top":
            self.opengl_view.setCameraPosition(azimuth=0, elevation=90)

        elif position == "front":
            self.opengl_view.setCameraPosition(azimuth=-90, elevation=0)

        elif position == "side": # right
            self.opengl_view.setCameraPosition(azimuth=0, elevation=0)
        pass

    
class DragValue(pg.QtWidgets.QDoubleSpinBox):

    def __init__(self, default = 0):
        super().__init__()
        self.setValue(default)
        self.setRange(-1e9, 1e9)
        self.setDecimals(4)
        self.setSingleStep(0.1)
        
        locale = QtCore.QLocale(QtCore.QLocale.C)
        locale.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)
        self.setLocale(locale)

        self.dragging = False
        self.last_x = 0

    def mousePressEvent(self, ev):

        if ev.button() == QtCore.Qt.MiddleButton:
            self.dragging = True
            self.last_x = ev.globalX()
            self.setCursor(QtCore.Qt.SizeHorCursor)

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):

        if self.dragging:

            dx = ev.globalX() - self.last_x
            self.last_x = ev.globalX()

            sensitivity = self.singleStep()

            modifiers = pg.QtWidgets.QApplication.keyboardModifiers()

            # Shift = hassas
            if modifiers & QtCore.Qt.ShiftModifier:
                sensitivity *= 0.1

            # Ctrl = snapping
            if modifiers & QtCore.Qt.ControlModifier:
                sensitivity *= 10

            self.setValue(
                self.value() + dx * sensitivity
            )

            # Sonsuz drag hissi
            pg.QtGui.QCursor.setPos(
                self.mapToGlobal(self.rect().center())
            )

            self.last_x = pg.QtGui.QCursor.pos().x()

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):

        if ev.button() == QtCore.Qt.MiddleButton:
            self.dragging = False
            self.unsetCursor()

        super().mouseReleaseEvent(ev)

class CustomDragValue(pg.QtWidgets.QWidget):
    def __init__(self, label, default = 0):
        super().__init__()
        layout = pg.QtWidgets.QHBoxLayout(self)
        self.drag_value = DragValue(default)
        self.textChanged = self.drag_value.textChanged.connect

        self.label = pg.QtWidgets.QLabel(label)
        self.value = self.drag_value.value

        layout.addWidget(self.label)
        layout.addWidget(self.drag_value)

class ControlLayout(pg.QtWidgets.QVBoxLayout):
    def __init__(self, opengl_view):
        super().__init__()
        self.opengl_view = opengl_view
        # Center ==
        self.slide_x = CustomDragValue("x")
        self.slide_y = CustomDragValue("y")
        self.slide_z = CustomDragValue("z")

        self.addWidget(self.slide_x)
        self.addWidget(self.slide_y)
        self.addWidget(self.slide_z)

        for a in [self.slide_x, self.slide_y, self.slide_z]:
            a.textChanged(self.update_center)
        # Center --

        self.view_preset_button_group = ViewPresetButtonGroup(opengl_view)
        self.addWidget(self.view_preset_button_group)

        self.update_distance = CustomDragValue("distance", default=200)
        self.update_distance.textChanged(self.set_distance)
        self.addWidget(self.update_distance)
        self.addStretch()

        self.threshold_slider_low = CustomDragValue("threshold low", -10)
        self.threshold_slider_high = CustomDragValue("threshold high", 10)

        self.addWidget(self.threshold_slider_low)
        self.addWidget(self.threshold_slider_high)

    
    def update_center(self):
        x = int(self.slide_x.value())
        y = int(self.slide_y.value())
        z = int(self.slide_z.value())

        self.opengl_view.opts['center'] = pg.Vector(x, y, z)

        self.opengl_view.update()

    def set_distance(self, distance):
        self.opengl_view.opts["distance"] = float(distance)
        self.opengl_view.update()



class TensorMonitor(QtCore.QObject):
    def __init__(self):
        super().__init__()

        # Variable
        self.current_data = None
        self.app = pg.mkQApp("Monitor")
        self.app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # =========================
        # WINDOW
        # =========================
        self.window = pg.QtWidgets.QWidget()

        self.main_layout = pg.QtWidgets.QHBoxLayout()

        self.opengl_view = gl.GLViewWidget()

        self.controls_layout = ControlLayout(self.opengl_view)
        self.controls_layout.threshold_slider_low.textChanged(self.update_data)
        self.controls_layout.threshold_slider_high.textChanged(self.update_data)

        # =========================
        # MAIN LAYOUT
        # =========================
        self.main_layout.addLayout(self.controls_layout, stretch=1)
        self.main_layout.addWidget(self.opengl_view, stretch=5)

        self.window.setLayout(self.main_layout)

        self.window.resize(1200, 800)
        self.window.show()

        # =========================
        # STATE
        # =========================
        self.scatter = None

    @QtCore.pyqtSlot(object)
    def guncelle(self, data_dic):

        if getattr(self, "_updating", False):
            return

        self._updating = True
        try:
            self.current_data_dic = data_dic
            self.update_data()

            pg.Qt.QtTest.QTest.qWait(1000)
        except Exception as e:
            print(f"Görselleştirme hatası: {e.with_traceback()}")
        finally:
            self._updating = False

    def update_data(self):
        if self.scatter is not None:
                self.opengl_view.clear()

        if self.current_data_dic is not None:
            self.scatter = visualize_tensor_scatter_dict(self.opengl_view, self.current_data_dic, threshold_low=self.get_threshold_value()[0], threshold_high=self.get_threshold_value()[1])
        
    def get_threshold_value(self):
        return (self.controls_layout.threshold_slider_low.value() / 100, self.controls_layout.threshold_slider_high.value() / 100)
    
    def run(self):
        self.app.exec_()