from madcad import *
from pyqtgraph.opengl import GLMeshItem, GLScatterPlotItem, GLViewWidget
import pyqtgraph.opengl as gl
import numpy as np
import torch

import matplotlib.colors as mcolors
import seaborn as sns

class Text3D(GLMeshItem):
    def __init__(self, text_str: str, xyz: tuple=(0,0,0)):
        super().__init__()
        label = text.text(text_str, fill=True, size=1)

        self.vertices = np.array([
            [p.x + xyz[0], p.y, p.z]
            for p in label.points
        ], dtype=np.float32)

        self.faces = np.array([
            list(face)
            for face in label.faces
        ], dtype=np.uint32)

        super().__init__(
            vertexes=self.vertices,
            faces=self.faces,
            smooth=False,
            # shader="shaded",
            drawEdges=False,
            color=(1, 1, 1, 1)
        )
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
        self.tensor_name_obj = Text3D(tensor_name, (tensor_data.shape[0] - 100, 0,0))

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

        toplam_x = 0 
        for key, value in tensor_data_dict.items():
            plot_view = TensorPlotItem(key,value, (toplam_x, 0, 0), dot_size, threshold_low, threshold_high, x_spacing)
            toplam_x += plot_view.get_max_lenght()[0]
            opengl_view.addItem(plot_view)
            self.opengl_items.append(plot_view)

    def get_items(self):
        return self.opengl_items
    
    def delete_items_in_opengl_view(opengl_view: gl.GLViewWidget, opengl_items: list[TensorPlotItem]):        
        for i in opengl_items:
            opengl_view.removeItem(i)
            opengl_view.removeItem(i.tensor_name_obj)
