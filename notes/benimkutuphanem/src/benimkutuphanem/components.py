from madcad import *
from pyqtgraph.opengl import GLMeshItem, GLScatterPlotItem, GLViewWidget
from pyglm import *
import pyqtgraph.opengl as gl
import numpy as np
import torch
import pyqtgraph as pg
import pyqtgraph.Qt.QtCore as QtCore
import matplotlib.colors as mcolors
import seaborn as sns

class Text3D(GLMeshItem):
    _mesh_cache = {}

    def __init__(self, text_str: str, xyz: tuple=(0,0,0)):
        try:
            if text_str in self._mesh_cache:
                self.vertices, self.faces = self._mesh_cache[text_str]
            else:
                label = text.text(text_str, fill=True, size=5)
                rot = mat3(rotate(radians(135), vec3(0,0,1)))

                self.vertices = np.array([
                    rot * p + xyz
                    for p in label.points
                ], dtype=np.float32)

                self.faces = np.array([
                    list(face)
                    for face in label.faces
                ], dtype=np.uint32)
                self._mesh_cache[text_str] = (self.vertices, self.faces)
            super().__init__(
                vertexes=self.vertices,
                faces=self.faces,
                smooth=False,
                drawEdges=False,
                color=(1, 1, 1, 1)
            )
        except Exception as e:
            print(f"text hatası: {e.with_traceback()}")
    def get_shape(self):
        if self.vertexes is None:
            return (0,0,0)
        xmin, ymin, zmin = self.vertexes.min(axis=0)
        xmax, ymax, zmax = self.vertexes.max(axis=0)

        width  = xmax - xmin
        height = ymax - ymin
        depth  = zmax - zmin
        return (width, height, depth)

class TensorPlotItem(GLScatterPlotItem):

    def __init__(self, tensor_name,tensor_data, xyz:tuple = (0,0,0) ,dot_size=10, threshold_low=0.0, threshold_high=0.0, x_spacing=50):
        # super().__init__(parentItem, **kwds)
        if torch.is_tensor(tensor_data):
            tensor_data = tensor_data.detach().cpu().numpy()
        
        if len(tensor_data.shape) == 1:
            tensor_data = np.expand_dims(tensor_data, axis=-1)
            tensor_data = np.expand_dims(tensor_data, axis=-1)
        elif len(tensor_data.shape) == 2:
            tensor_data = np.expand_dims(tensor_data, axis=-1)

        self.tensor_data = tensor_data
        self.tensor_name = tensor_name
        self.tensor_name_obj = Text3D(tensor_name, (xyz[0], xyz[1] + (tensor_data.shape[1]/2),0))

        mask = (tensor_data >= threshold_low) & (tensor_data <= threshold_high)
        indices = np.argwhere(mask)
        
        values = tensor_data[mask]

        if values is None or len(values) == 0 or np.all(np.isnan(values)):
            norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            norm = mcolors.Normalize(
                vmin=np.nanmin(values),
                vmax=np.nanmax(values)
            )

        cmap = sns.color_palette("magma_r", as_cmap=True)
        colors = cmap(norm(values))

        indices = indices.copy()

        indices[:, 0] += xyz[0]
        indices[:, 1] += xyz[1]
        indices[:, 2] += xyz[2]

        super().__init__(    
            pos=indices,
            color=colors,
            size=dot_size,
            pxMode=True)
        


    def get_max_lenght(self):
        tensor = self.tensor_data.shape
        name = self.tensor_name_obj.get_shape()
        
        return (tensor[0] + name[0], tensor[1] + name[1], tensor[2] + name[2])

    def _setView(self, v):
        super()._setView(v)
        if v is not None:
            v.addItem(self.tensor_name_obj)

class TensorPlotItemGrid():
    def __init__(self,opengl_view: gl.GLViewWidget, tensor_data_dict: dict[str, torch.Tensor | np.ndarray], dot_size=10, threshold_low=0.0, threshold_high=0.0, x_spacing=50):    
        self.opengl_items: list[TensorPlotItem] = []

        toplam_y = 0 
        for key, value in tensor_data_dict.items():
            plot_view = TensorPlotItem(key,value, (0, toplam_y, 0), dot_size, threshold_low, threshold_high, x_spacing)
            toplam_y += plot_view.get_max_lenght()[1] + x_spacing
            opengl_view.addItem(plot_view)
            self.opengl_items.append(plot_view)

    def get_items(self):
        return self.opengl_items
    
    def delete_items_in_opengl_view(opengl_view: gl.GLViewWidget, opengl_items: list[TensorPlotItem]):      
        for i in opengl_items:
            opengl_view.removeItem(i)
            opengl_view.removeItem(i.tensor_name_obj)


class ViewPresetCube(pg.QtWidgets.QWidget):

    def __init__(self, opengl_view: gl.GLViewWidget):
        super().__init__()

        self.opengl_view = opengl_view

        grid = pg.QtWidgets.QGridLayout(self)
        grid.setSpacing(2)

        # ---------------- Buttons ----------------

        self.button_top = self.make_button("ÜST")
        self.button_bottom = self.make_button("ALT")
        self.button_front = self.make_button("ÖN")
        self.button_back = self.make_button("ARKA")
        self.button_left = self.make_button("SOL")
        self.button_right = self.make_button("SAĞ")

        # ---------------- Layout ----------------

        grid.addWidget(self.button_top,   0, 1)
        grid.addWidget(self.button_left,  1, 0)
        grid.addWidget(self.button_front, 1, 1)
        grid.addWidget(self.button_right, 1, 2)
        grid.addWidget(self.button_back,  2, 1)
        grid.addWidget(self.button_bottom,3, 1)

        # ---------------- Connections ----------------

        self.button_top.clicked.connect(
            lambda: self.animate_camera(0, 90)
        )

        self.button_bottom.clicked.connect(
            lambda: self.animate_camera(0, -90)
        )

        self.button_front.clicked.connect(
            lambda: self.animate_camera(-90, 0)
        )

        self.button_back.clicked.connect(
            lambda: self.animate_camera(90, 0)
        )

        self.button_left.clicked.connect(
            lambda: self.animate_camera(180, 0)
        )

        self.button_right.clicked.connect(
            lambda: self.animate_camera(0, 0)
        )

        # ---------------- Animation Timer ----------------

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_animation)

        self.target_azimuth = 0
        self.target_elevation = 0

        self.animation_speed = 0.15


    def make_button(self, text):
        button = pg.QtWidgets.QPushButton(text)
        return button

    def animate_camera(self, azimuth, elevation):

        self.target_azimuth = azimuth
        self.target_elevation = elevation
        # self.opengl_view.setCameraPosition(
        #     azimuth=self.target_azimuth,
        #     elevation=self.target_elevation
        # )
        self.timer.start(16)  # ~60 FPS

    def update_animation(self):

        opts = self.opengl_view.opts

        current_az = opts["azimuth"]
        current_el = opts["elevation"]

        # Interpolation
        new_az = current_az + (
            self.target_azimuth - current_az
        ) * self.animation_speed

        new_el = current_el + (
            self.target_elevation - current_el
        ) * self.animation_speed

        self.opengl_view.setCameraPosition(
            azimuth=new_az,
            elevation=new_el
        )

        # Stop threshold
        if (
            abs(new_az - self.target_azimuth) < 0.5
            and
            abs(new_el - self.target_elevation) < 0.5
        ):

            self.opengl_view.setCameraPosition(
                azimuth=self.target_azimuth,
                elevation=self.target_elevation
            )

            self.timer.stop()

class DragValue(pg.QtWidgets.QDoubleSpinBox):
    def __init__(self, default = 0, focus_function = None):
        super().__init__()
        self.focus_function = focus_function
        self.setRange(-1e9, 1e9)
        self.setValue(default)
        self.setDecimals(4)
        self.setSingleStep(0.1)
        
        locale = QtCore.QLocale(QtCore.QLocale.C)
        locale.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)
        self.setLocale(locale)

        self.dragging = False
        self.last_x = 0

    def focusInEvent(self, e):
        if self.focus_function is not None:
            self.focus_function()
        return super().focusInEvent(e)

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
    def __init__(self, label, default = 0, focus_function = None):
        super().__init__()
        layout = pg.QtWidgets.QHBoxLayout(self)
        self.drag_value = DragValue(default, focus_function)
        self.textChanged = self.drag_value.textChanged.connect

        self.label = pg.QtWidgets.QLabel(label)
        self.value = self.drag_value.value

        layout.addWidget(self.label)
        layout.addWidget(self.drag_value)

class ControlLayout(pg.QtWidgets.QVBoxLayout):
    def __init__(self, opengl_view):
        super().__init__()
        self.opengl_view = opengl_view

        self.repeat = pg.QtWidgets.QPushButton("repeat")
        self.repeat_button = self.repeat.clicked.connect
        self.addWidget(self.repeat)
        # Center ==
        self.slide_x = CustomDragValue("x", focus_function=self.update_inputs)
        self.slide_y = CustomDragValue("y", focus_function=self.update_inputs)
        self.slide_z = CustomDragValue("z", focus_function=self.update_inputs)

        self.addWidget(self.slide_x)
        self.addWidget(self.slide_y)
        self.addWidget(self.slide_z)

        for a in [self.slide_x, self.slide_y, self.slide_z]:
            a.textChanged(self.update_center)
        # Center --

        self.view_preset_button_group = ViewPresetCube(opengl_view)
        self.addWidget(self.view_preset_button_group)

        self.update_distance = CustomDragValue("distance", default=200, focus_function=self.update_inputs)
        self.update_distance.textChanged(self.set_distance)


        self.addWidget(self.update_distance)
        self.addStretch()

        self.threshold_slider_low = CustomDragValue("threshold low", -10, focus_function=self.update_inputs)
        self.threshold_slider_high = CustomDragValue("threshold high", 10, focus_function=self.update_inputs)

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

    def update_inputs(self):
        center : pg.Vector = self.opengl_view.opts['center']
        self.slide_x.drag_value.setValue(center.x())
        self.slide_y.drag_value.setValue(center.y())
        self.slide_z.drag_value.setValue(center.z())
        
        
        distance = self.opengl_view.opts["distance"]
        self.update_distance.drag_value.setValue(distance)
        pass

