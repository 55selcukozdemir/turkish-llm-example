from datetime import datetime
from rich import print

from torch import Tensor

import matplotlib.pyplot as plt
import matplotlib

import numpy as np


import os
class Logger:
    def __init__(self, folder,file_name="app.log", log_on = True):
        self.full_file_name = os.path.join(folder, file_name)
        self.folder = folder
        self.log_on = log_on
        if not os.path.exists(folder):
            os.makedirs(folder)
            self.info(f"'{folder}' başarıyla oluşturuldu.")
        else:
            self.info(f"'{folder}' zaten mevcut.")

    def log(self, level, message):
        if not self.log_on:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"

        print(log_line)

        with open(self.full_file_name, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    def info(self, message):
        self.log("INFO", message)

    def warning(self, message):
        self.log("WARNING", message)

    def error(self, message):
        self.log("ERROR", message)
        
    # Tensor hangi yönlü ise o şekilde yazılması gerekebilir
    def log_tensor(self, prefix: str, tensor: Tensor):
        
        numpy_data = tensor.cpu().numpy()
        self.log_ndarray(numpy_data)

    def log_ndarray(self, prefix: str, numpy_data: np.ndarray):
        if not self.log_on:
            return
        if(numpy_data.ndim == 1 or numpy_data.ndim == 2):
            klasor_zamani = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            full_name = os.path.join(self.folder, f"{klasor_zamani}_{prefix}")
            self.write_heatmap(full_name, numpy_data)
        elif (numpy_data.ndim == 3):
            for i, data in np.ndenumerate(numpy_data):
                klasor_zamani = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                full_name = os.path.join(self.folder, f"{klasor_zamani}_{i}_{prefix}")
                self.write_heatmap(full_name, data)
                
                
        else:
            raise ValueError("Max support 3 rang tensor") 
    
    
    def write_heatmap(self, full_name, data):
        matplotlib.use('Agg') # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it

        fig, ax = plt.subplots()
        fig.text(
            0.5, 0.98,
            str(data.shape),
            ha="center",
            va="top",
            fontsize=16
        )
        ax.imshow(data)
        fig.savefig(full_name)   
