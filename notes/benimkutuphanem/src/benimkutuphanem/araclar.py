import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import numpy as np
import sys

import torch

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns







def selam():
    return "Merhaba"

def visualize_tensor_scatter_dict(
    view_widget,
    tensor_dict,
    dot_size=10,
    threshold_low=0.0,
    threshold_high=0.0,
    x_spacing=300
):
    cmap = sns.color_palette("magma_r", as_cmap=True)

    all_scatter = []

    for i, tensor in enumerate(tensor_dict.values()):

        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()

        mask = (tensor >= threshold_low) & (tensor <= threshold_high)
        indices = np.argwhere(mask)
        # Boyutu 3'e tamamla
        if indices.shape[1] == 1:
            # 1D -> (x,0,0)
            indices = np.concatenate([
                indices,
                np.zeros((indices.shape[0], 2))
            ], axis=1)

        elif indices.shape[1] == 2:
            # 2D -> (x,y,0)
            indices = np.concatenate([
                indices,
                np.zeros((indices.shape[0], 1))
            ], axis=1)

        elif indices.shape[1] > 3:
            # Fazla boyut varsa ilk 3'ü kullan
            indices = indices[:, :3]

        values = tensor[mask]
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        colors = cmap(norm(values))

        # 🔥 X ekseninde offset
        indices = indices.copy()
        indices[..., 0] += i * x_spacing

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

def visualize_tensor_scatter(view_widget, tensor, dot_size=10, threshold_low=0.0, threshold_high=0.0):

    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()

    # Threshold maskesi
    mask = (tensor >= threshold_low) & (tensor <= threshold_high)

    # Noktalar
    indices = np.argwhere(mask)

    if indices.shape[0] == 0:
        print("Görselleştirilecek veri bulunamadı.")
        return None

    # Sadece seçilen voxel değerleri
    values = tensor[mask]

    # Normalize
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

    # Colormap
    cmap = sns.color_palette("magma_r", as_cmap=True)

    colors = cmap(norm(values))

    scatter = gl.GLScatterPlotItem(
        pos=indices,
        color=colors,
        size=dot_size,
        pxMode=True
    )

    view_widget.addItem(scatter)

    return scatter

class TensorMonitor(QtCore.QObject):
    def __init__(self):
        super().__init__()

        self.app = pg.mkQApp("Monitor")

        # =========================
        # WINDOW
        # =========================
        self.window = pg.QtWidgets.QWidget()

        self.main_layout = pg.QtWidgets.QHBoxLayout()
        self.controls_layout = pg.QtWidgets.QVBoxLayout()

        self.view = gl.GLViewWidget()

        # başlangıç kamera ayarı
        self.view.opts['distance'] = 200

        self.slider_x_label = pg.QtWidgets.QLabel("X")
        self.slider_y_label = pg.QtWidgets.QLabel("Y")
        self.slider_z_label = pg.QtWidgets.QLabel("Z")

        self.slider_x = pg.QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y = pg.QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_z = pg.QtWidgets.QSlider(QtCore.Qt.Horizontal)

        # başlangıç min/max
        for slider in [self.slider_x, self.slider_y, self.slider_z]:
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)

        # layout
        self.controls_layout.addWidget(self.slider_x_label)
        self.controls_layout.addWidget(self.slider_x)

        self.controls_layout.addWidget(self.slider_y_label)
        self.controls_layout.addWidget(self.slider_y)

        self.controls_layout.addWidget(self.slider_z_label)
        self.controls_layout.addWidget(self.slider_z)
        self.controls_layout.addStretch()

        # signals
        self.slider_x.valueChanged.connect(self.update_center)
        self.slider_y.valueChanged.connect(self.update_center)
        self.slider_z.valueChanged.connect(self.update_center)

        self.current_data = None
        self.threshold = (0.2, 0.2)

        self.threshold_label_min = pg.QtWidgets.QLabel("Threshold_min")
        self.controls_layout.addWidget(self.threshold_label_min)

        self.threshold_label_max = pg.QtWidgets.QLabel("Threshold_max")
        self.controls_layout.addWidget(self.threshold_label_max)

        self.threshold_slider_low = pg.QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider_low.setMinimum(-1000)
        self.threshold_slider_low.setMaximum(10000)
        self.threshold_slider_low.setValue(-1000)
        self.controls_layout.addWidget(self.threshold_slider_low)

        self.threshold_slider_high = pg.QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider_high.setMinimum(-1000)
        self.threshold_slider_high.setMaximum(1000)
        self.threshold_slider_high.setValue(1000)
        self.controls_layout.addWidget(self.threshold_slider_high)

        self.threshold_label_min.setText(
            f"Threshold min: {self.get_threshold_value()[0]:.2f}"
        )

        self.threshold_label_max.setText(
            f"Threshold max: {self.get_threshold_value()[1]:.2f}"
        )


        self.threshold_slider_low.valueChanged.connect(self.update_threshold)
        self.threshold_slider_high.valueChanged.connect(self.update_threshold)

        # =========================
        # MAIN LAYOUT
        # =========================
        self.main_layout.addLayout(self.controls_layout, stretch=1)
        self.main_layout.addWidget(self.view, stretch=5)

        self.window.setLayout(self.main_layout)

        self.window.resize(1200, 800)
        self.window.show()

        # =========================
        # STATE
        # =========================
        self.scatter = None
        self.tensor_shape = None
        self.min_trashold = 0
        self.max_trashold = 0

    def aget_global_min_max(self, tensor_dict):
        global_min = float("inf")
        global_max = float("-inf")

        for i, tensor in enumerate(tensor_dict.values()):

            if torch.is_tensor(tensor):
                tensor = tensor.detach().cpu().numpy()

            t_min = tensor.min()
            t_max = tensor.max()

            if t_min < global_min:
                global_min = t_min

            if t_max > global_max:
                global_max = t_max

        return global_min, global_max
    
    def compute_scene_bounds(self, tensor_dict, threshold_low=0.0, threshold_high=0.0, x_spacing=50):
        all_points = []

        for i, tensor in enumerate(tensor_dict.values()):

            if torch.is_tensor(tensor):
                tensor = tensor.detach().cpu().numpy()

            mask = (tensor >= threshold_low) & (tensor <= threshold_high)
            coords = np.argwhere(mask)

            if coords.shape[0] == 0:
                continue

            # x offset ekle
            coords = coords.copy()
            coords[:, 0] += i * x_spacing

            all_points.append(coords)

        if len(all_points) == 0:
            return (0, 0, 0)

        all_points = np.vstack(all_points)

        min_xyz = all_points.min(axis=0)
        max_xyz = all_points.max(axis=0)

        size_xyz = tuple(max_xyz - min_xyz)

        return size_xyz
    @QtCore.pyqtSlot(object)
    def guncelle(self, data_dic):
        data = None
        print("visualization started")
        if getattr(self, "_updating", False):
            return

        self._updating = True
        try:
            self.current_data = data
            self.current_data_dic = data_dic
            # if self.tensor_shape != data.shape:
            #     self.tensor_shape = data.shape
                
            #     if data_dic is not None:
            #         self.tensor_shape = self.compute_scene_bounds(data_dic)

            #     # slider limitlerini tensor boyutuna göre ayarla
            #     self.slider_x.setMaximum(self.tensor_shape[0] - 1)
            #     self.slider_y.setMaximum(self.tensor_shape[1] - 1)
            #     self.slider_z.setMaximum(self.tensor_shape[2] - 1)

            #     # slider başlangıç pozisyonları
            #     self.slider_x.setValue(self.tensor_shape[0] // 2)
            #     self.slider_y.setValue(self.tensor_shape[1] // 2)
            #     self.slider_z.setValue(self.tensor_shape[2] // 2)

            #                 # ilk center
            #     self.update_center()

            if self.current_data_dic is not None:
                min_val, max_val = self.aget_global_min_max(data_dic)
            else:
                min_val = np.min(data)
                max_val = np.max(data) 

            min_val = int(min_val * 1000)
            max_val = int(max_val * 1000)
            
            if min_val < self.min_trashold:
                self.threshold_slider_low.setMinimum(min_val)
                self.threshold_slider_high.setMinimum(min_val)
                self.min_trashold = min_val - 1000

            if max_val > self.max_trashold:
                self.threshold_slider_low.setMaximum(max_val)
                self.threshold_slider_high.setMaximum(max_val)
                self.max_trashold = max_val + 1000

            if self.scatter is not None:
                # if type(self.scatter) is list:
                #     for item in self.scatter:
                #         self.view.removeItem(item)
                # else:
                #         self.view.removeItem(self.scatter)

                self.view.clear()

            if data_dic is not None:
                self.scatter = visualize_tensor_scatter_dict(self.view, data_dic, threshold_low=self.get_threshold_value()[0], threshold_high=self.get_threshold_value()[1])
            else:
                self.scatter = visualize_tensor_scatter(self.view, data, threshold_low=self.get_threshold_value()[0], threshold_high=self.get_threshold_value()[1])


            pg.Qt.QtTest.QTest.qWait(1000)
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
        finally:
            self._updating = False

    
    
    def update_center(self):

        x = self.slider_x.value()
        y = self.slider_y.value()
        z = self.slider_z.value()

        self.view.opts['center'] = pg.Vector(x, y, z)

        self.view.update()
    def update_threshold(self):

        self.threshold_label_min.setText(
            f"Threshold min: {self.get_threshold_value()[0]:.2f}"
        )

        self.threshold_label_max.setText(
            f"Threshold max: {self.get_threshold_value()[1]:.2f}"
        )

        if self.scatter is not None:
                self.view.removeItem(self.scatter)

        if self.data_dic is not None:
            visualize_tensor_scatter_dict(self.view, self.data_dic, threshold_low=self.get_threshold_value()[0], threshold_high=self.get_threshold_value()[1])
        elif self.current_data is not None:
            self.scatter = visualize_tensor_scatter(
                self.view,
                self.current_data,
                threshold_low=self.get_threshold_value()[0],
                threshold_high=self.get_threshold_value()[1]
            )

    def get_threshold_value(self):
        return (self.threshold_slider_low.value() / 100, self.threshold_slider_high.value() / 100)
    def run(self):
        self.app.exec_()