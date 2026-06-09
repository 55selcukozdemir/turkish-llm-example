from benimkutuphanem.src.benimkutuphanem.components import ControlLayout, TensorPlotItemGrid
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

import torch
import numpy as np

from qt_material import apply_stylesheet
import qt_themes

def selam():
    return "Merhaba"

class TensorMonitor(QtCore.QObject):
    def __init__(self):
        super().__init__()

        # Variable
        self.current_data = None
        self.current_data_dic = None
        self.app = pg.mkQApp("Monitor")
        # self.app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # apply_stylesheet(self.app, theme='dark_blue.xml')
        qt_themes.set_theme('nord')
        # =========================
        # WINDOW
        # =========================
        self.window = pg.QtWidgets.QWidget()

        self.main_layout = pg.QtWidgets.QHBoxLayout()

        self.opengl_view = gl.GLViewWidget()

        self.controls_layout = ControlLayout(self.opengl_view)
        self.controls_layout.threshold_slider_low.textChanged(self.update_data)
        self.controls_layout.threshold_slider_high.textChanged(self.update_data)
        self.controls_layout.repeat_button

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
            TensorPlotItemGrid.delete_items_in_opengl_view(self.opengl_view, self.scatter)

        if self.current_data_dic is not None:
            tensor_grid_view = TensorPlotItemGrid(self.opengl_view, self.current_data_dic, threshold_low=self.get_threshold_value()[0], threshold_high=self.get_threshold_value()[1])
            self.scatter = tensor_grid_view.get_items()
        
    def get_threshold_value(self):
        return (self.controls_layout.threshold_slider_low.value() / 100, self.controls_layout.threshold_slider_high.value() / 100)
    
    def run(self):
        self.app.exec_()

class TrainerThread(QtCore.QThread):

    """
    Kullanım şeklim
    >>> def test (log, data_viewing):
    >>>     log("test")
    >>> tensor_monitor = TensorMonitor()
    >>> t = TrainerThread(test, tensor_monitor)
    >>> t.start()
    >>> tensor_monitor.run()
    """

    data_signal = QtCore.pyqtSignal(object)
    log_signal = QtCore.pyqtSignal(str)
    
    def __init__(self, callback, tensor_monitor: TensorMonitor):
        super().__init__()
        self.callback = callback
        self.running = True
        self.data_signal.connect(tensor_monitor.guncelle)
        self.log_signal.connect(print)

        tensor_monitor.controls_layout.repeat_button(lambda: self.run())
    
    def log(self, log_txt: str):
        self.log_signal.emit(log_txt)
    
    def data_viewing(self, data: dict[str, torch.Tensor | np.ndarray]):
        """
        params_info = dict(model.named_parameters()) şeklinde alınabilir.
        """
        self.data_signal.emit(data)
    
    def run(self):
        try:
            if callable(self.callback):
                self.callback(self.log, self.data_viewing)
            else:
                self.log("fonksiyon çalıştırılamıyor.")   
        except Exception as e:
            self.log(f"Çalıştırma hatası. {e.with_traceback()}")   
    def stop(self):
        self.running = False
    
