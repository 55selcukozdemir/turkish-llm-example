import numpy as np
import torch
import pandas as pd


def tensor_to_blender_ply(
    tensor,
    output_file="tensor.ply",
    threshold=0.0,
    cube_size=1.0,
    normalize=True
):
    """
    N-D tensor → Blender PLY (vertex color garantili)
    """

    # -------------------------
    # PyTorch support
    # -------------------------
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
    except ImportError:
        pass

    T = np.asarray(tensor, dtype=np.float32)

    if T.ndim < 3:
        raise ValueError("Minimum 3D tensor gerekli (i,j,k,...)")

    # -------------------------
    # Flatten extra dims
    # -------------------------
    spatial = T.shape[:3]
    T = T.reshape(*spatial, -1)

    # -------------------------
    # Normalize
    # -------------------------
    if normalize:
        tmin = T.min()
        tmax = T.max()
        if tmax - tmin > 1e-8:
            T = (T - tmin) / (tmax - tmin)

    vertices = []
    colors = []

    sx, sy, sz, d = T.shape

    # -------------------------
    # VOXEL GENERATION
    # -------------------------
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):

                vec = T[i, j, k]

                value = float(np.mean(vec))

                if value < threshold:
                    continue

                # -------------------------
                # POSITION
                # -------------------------
                x = i * cube_size
                y = j * cube_size
                z = k * cube_size

                # -------------------------
                # COLOR (FIXED)
                # -------------------------
                if vec.shape[0] >= 3:
                    r, g, b = vec[0], vec[1], vec[2]
                else:
                    r = g = b = value

                # extra dims influence brightness
                brightness = float(np.mean(vec))

                r = r * brightness
                g = g * brightness
                b = b * brightness

                # safety clamp
                r = np.clip(r, 0, 1)
                g = np.clip(g, 0, 1)
                b = np.clip(b, 0, 1)

                vertices.append((x, y, z))
                colors.append((r, g, b))

    # -------------------------
    # WRITE PLY (Blender compatible)
    # -------------------------
    with open(output_file, "w") as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        # IMPORTANT: Blender reads RGB like this
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("end_header\n")

        for v, c in zip(vertices, colors):

            r = int(c[0] * 255)
            g = int(c[1] * 255)
            b = int(c[2] * 255)

            f.write(f"{v[0]} {v[1]} {v[2]} {r} {g} {b}\n")

    print("✔ Export successful:", output_file)
    print("✔ Voxels:", len(vertices))


def export_csv(tensor: torch.Tensor):
    x = tensor.detach().cpu().numpy()
    x = pd.DataFrame(x)
    x.to_csv("tensor.csv")