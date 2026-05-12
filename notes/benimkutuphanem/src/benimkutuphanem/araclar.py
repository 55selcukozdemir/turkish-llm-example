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

        self.window = pg.QtWidgets.QWidget()
        self.layout = pg.QtWidgets.QVBoxLayout()

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.setMinimumSize(400, 300)
        self.view.setMaximumSize(1200, 800)
        self.view.setCameraPosition(distance=50)
        self.layout.addWidget(self.view)

        self.button = pg.QtWidgets.QPushButton("Grafiği Güncelle")
        self.layout.addWidget(self.button)

        self.zoom_in_btn = pg.QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn = pg.QtWidgets.QPushButton("Zoom -")

        self.layout.addWidget(self.zoom_in_btn)
        self.layout.addWidget(self.zoom_out_btn)

        # signals
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        self.window.setLayout(self.layout)
        self.window.show()

        self.scatter = None

    @QtCore.pyqtSlot(object)
    def guncelle(self, data):
        print("visualization started")
        try:
            if self.scatter is not None:
                self.view.removeItem(self.scatter)

            self.scatter = visualize_tensor_scatter(self.view, data)

        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

    def zoom_in(self):
        cam = self.view.cameraPosition()
        new_dist = max(1, cam[2] * 5)
        self.view.setCameraPosition(distance=new_dist)

    def zoom_out(self):
        cam = self.view.cameraPosition()
        new_dist = cam[2] * 1.2
        self.view.setCameraPosition(distance=new_dist)

    def run(self):
        self.app.exec_()