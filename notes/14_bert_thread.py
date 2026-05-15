
from benimkutuphanem import TensorMonitor
from benimkutuphanem.araclar import TrainerThread
import pyqtgraph as pg




def test (log, data_viewing):
    log("test")
tensor_monitor = TensorMonitor()
t = TrainerThread(test, tensor_monitor)
t.start()
tensor_monitor.run()