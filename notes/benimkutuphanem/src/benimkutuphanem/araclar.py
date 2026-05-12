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
def visualize_tensor_scatter(view_widget, tensor, dot_size=10):
    """
    Tensörü GLScatterPlotItem ile görselleştirir (Seaborn magma_r colormap).
    """

    # 1. Noktalar (DOĞRU FORMAT)
    indices = np.argwhere(tensor)
    center = indices.mean(axis=0)
    # view_widget.opts['center'] = pg.Vector(center[0], center[1], center[2])
    if indices.shape[0] == 0:
        print("Görselleştirilecek veri bulunamadı.")
        return None

    # 2. Değerler
    values = tensor.flatten()

    # 3. Normalize
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

    # 4. Seaborn colormap
    cmap = sns.color_palette("magma_r", as_cmap=True)

    # 5. RGBA renkler
    colors = cmap(norm(values))

    # 6. Scatter plot (KRİTİK: pxMode=False)
    scatter = gl.GLScatterPlotItem(
        pos=indices,
        color=colors,
        size=dot_size,
        pxMode=True
    )

    view_widget.addItem(scatter)
    return scatter
def tensor_gorsellestir(data):
    app = pg.mkQApp("Scatter Tensör Görünümü")
    view = gl.GLViewWidget()
    view.show()
    view.setCameraPosition(distance=50)

    # Örnek: Yüksek boyutlu bir tensör (Örn: 50x50x50)
    # Bu 125.000 hücre demektir, Mesh ile çok kasar ama Scatter ile akıcıdır.
 

    visualize_tensor_scatter(view, data, threshold=0.2)

    if (sys.flags.interactive != 1) or not hasattr(pg, 'QtCore'):
        sys.exit(app.exec_())

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

        # =========================
        # UPDATE BUTTON
        # =========================
        self.button = pg.QtWidgets.QPushButton("Grafiği Güncelle")
        self.controls_layout.addWidget(self.button)

        # =========================
        # 3D VIEW
        # =========================
        self.view = gl.GLViewWidget()

        # başlangıç kamera ayarı
        self.view.opts['distance'] = 200

        # =========================
        # SLIDERS
        # =========================
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

        # signals
        self.slider_x.valueChanged.connect(self.update_center)
        self.slider_y.valueChanged.connect(self.update_center)
        self.slider_z.valueChanged.connect(self.update_center)

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

    @QtCore.pyqtSlot(object)
    def guncelle(self, data):

        print("visualization started")

        try:

            if self.tensor_shape != data.shape:
                self.tensor_shape = data.shape

                # slider limitlerini tensor boyutuna göre ayarla
                self.slider_x.setMaximum(self.tensor_shape[0] - 1)
                self.slider_y.setMaximum(self.tensor_shape[1] - 1)
                self.slider_z.setMaximum(self.tensor_shape[2] - 1)

                # slider başlangıç pozisyonları
                self.slider_x.setValue(self.tensor_shape[0] // 2)
                self.slider_y.setValue(self.tensor_shape[1] // 2)
                self.slider_z.setValue(self.tensor_shape[2] // 2)
                            # ilk center
                self.update_center()

            if self.scatter is not None:
                self.view.removeItem(self.scatter)

            self.scatter = visualize_tensor_scatter(self.view, data)



        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

    def update_center(self):

        x = self.slider_x.value()
        y = self.slider_y.value()
        z = self.slider_z.value()

        self.view.opts['center'] = pg.Vector(x, y, z)

        self.view.update()

    def run(self):
        self.app.exec_()