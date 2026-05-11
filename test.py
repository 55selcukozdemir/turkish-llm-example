import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Veri Hazırlama (Örnek Fonksiyon)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) # Buraya kendi model verinizi koyun

# 2. Renk Haritasını Oluşturma
# Z değerlerini 0-1 arasına normalize edip 'viridis' gibi bir renk haritası atıyoruz
norm = plt.Normalize(Z.min(), Z.max())
cmap = cm.get_cmap('viridis') 
rgba_colors = cmap(norm(Z))  # Matris formatında (100, 100, 4)

# 3. Noktaları (Vertices) ve Renkleri Düzleştirme
# Trimesh için verileri (N, 3) ve (N, 4) formatına getirmeliyiz
vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
vertex_colors = (rgba_colors.reshape(-1, 4) * 255).astype(np.uint8)

# 4. Yüzeyleri (Faces) Oluşturma
# Izgara üzerindeki noktaları birbirine bağlayarak üçgenler oluşturuyoruz
rows, cols = X.shape
faces = []
for r in range(rows - 1):
    for c in range(cols - 1):
        v1 = r * cols + c
        v2 = r * cols + (c + 1)
        v3 = (r + 1) * cols + c
        v4 = (r + 1) * cols + (c + 1)
        # Her kare hücre için iki üçgen (Triangle Strip mantığı)
        faces.append([v1, v2, v3])
        faces.append([v2, v4, v3])

faces = np.array(faces)

# 5. Trimesh Objesini Yaratma ve Kaydetme
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

# Blender için en iyisi .obj (mtl ile birlikte) veya .glb formatıdır
mesh.export('model_blender.obj')
print("Model başarıyla dışa aktarıldı!")