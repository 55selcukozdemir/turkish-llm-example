"""
TextMesh3D — 3D metin mesh oluşturucu

QPainterPath ile vektör yol çıkarımı →
earcut triangulation ile ön/arka kapak (delikler dahil) →
kontur bazlı yan duvar oluşturma →
pyqtgraph OpenGL ile görselleştirme
"""

import sys
import numpy as np
import mapbox_earcut as earcut

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QPointF
import pyqtgraph.opengl as gl


class TextMesh3D:
    def __init__(self, view, text="HELLO", pos=(0, 0, 0), depth=5):
        self.view = view
        self.pos = pos
        self.depth = depth

        vertices, faces = self.build_mesh(text)

        self.mesh = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            color=(1, 1, 1, 1),
            smooth=False,
            drawEdges=True,
        )

        self.mesh.translate(*pos)
        self.view.addItem(self.mesh)

    # -------------------------
    # TEXT → VECTOR PATH
    # -------------------------
    def text_to_path(self, text, size=100):
        font = QtGui.QFont("Arial")
        font.setPixelSize(size)

        path = QtGui.QPainterPath()
        path.addText(0, 0, font, text)
        return path

    # -------------------------
    # PATH → POLYGON CONTOURS
    # -------------------------
    def path_to_contours(self, path):
        """
        QPainterPath.toSubpathPolygons() ile her harfin dış ve iç
        konturlarını ayrı ayrı çıkarır.
        Y eksenini ters çevirir (Qt'de Y aşağı pozitif, OpenGL'de yukarı).
        Son noktayı çıkarır (ilk nokta = son nokta ise).
        """
        polygons = path.toSubpathPolygons()
        contours = []
        for poly in polygons:
            pts = []
            for i in range(poly.count()):
                p = poly.at(i)
                pts.append([p.x(), -p.y()])
            arr = np.array(pts)
            # Son nokta == ilk nokta ise çıkar (kapalı döngü)
            if len(arr) >= 2 and np.allclose(arr[0], arr[-1], atol=1e-6):
                arr = arr[:-1]
            if len(arr) >= 3:
                contours.append(arr)
        return contours

    # -------------------------
    # KONTUR YÖNÜ (CW / CCW)
    # -------------------------
    @staticmethod
    def signed_area(contour):
        """
        Shoelace formülü ile işaretli alan hesaplar.
        Pozitif = CCW (dış kontur), Negatif = CW (delik)
        """
        n = len(contour)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += contour[i][0] * contour[j][1]
            area -= contour[j][0] * contour[i][1]
        return area / 2.0

    # -------------------------
    # DIŞ KONTUR + DELİK GRUPLAMA
    # -------------------------
    def group_contours(self, contours, path):
        """
        Konturları dış sınır (outer) ve delik (hole) olarak gruplar.

        Mantık: Her konturun ilk noktasını alıp, diğer konturların
        *yalnızca kendi* oluşturdukları path içinde mi diye kontrol et.
        Eğer bir kontur başka hiçbir dış konturun içinde değilse → dış kontur.
        Eğer bir dış konturun içindeyse → o dış konturun deliği.

        Basitleştirilmiş yaklaşım: signed_area > 0 → CCW → dış,
        signed_area < 0 → CW → delik. Qt genelde dış konturları CCW,
        delikleri CW verir. Sonra her deliği en yakın (kapsayan) dışa ata.
        """
        outers = []  # (index, contour)
        holes = []   # (index, contour)

        for i, c in enumerate(contours):
            area = self.signed_area(c)
            # Y ekseni ters çevrildiği için sarım yönü de ters:
            # negatif alan = dış kontur, pozitif alan = delik
            if area < 0:
                outers.append((i, c))
            else:
                holes.append((i, c))

        # Eğer hiç dış kontur bulunamazsa, yönleri ters çevir
        if not outers:
            outers = holes
            holes = []

        # Her dış kontur için deliklerini bul
        groups = []
        for oi, outer in outers:
            # Bu dış konturun QPainterPath'ini oluştur
            outer_path = QtGui.QPainterPath()
            outer_path.addPolygon(QtGui.QPolygonF([QPointF(p[0], -p[1]) for p in outer]))
            outer_path.closeSubpath()

            my_holes = []
            for hi, hole in holes:
                # Deliğin merkez noktası dışın içinde mi?
                centroid = hole.mean(axis=0)
                if outer_path.contains(QPointF(centroid[0], -centroid[1])):
                    my_holes.append(hole)

            groups.append((outer, my_holes))

        return groups

    # -------------------------
    # EARCUT TRİANGÜLASYON
    # -------------------------
    def triangulate_polygon_with_holes(self, outer, holes):
        """
        mapbox_earcut kullanarak dış kontur + delikleri triangüle eder.
        Döner: (vertices_2d, faces)
        """
        # Tüm vertex'leri birleştir: önce outer, sonra her hole
        all_verts = [outer]
        # mapbox_earcut kümülatif bitiş indeksleri bekler
        # Örn: outer=10 nokta, hole=5 nokta → rings=[10, 15]
        cumulative = len(outer)
        ring_end_indices = [cumulative]

        for hole in holes:
            all_verts.append(hole)
            cumulative += len(hole)
            ring_end_indices.append(cumulative)

        vertices = np.vstack(all_verts).astype(np.float64)
        rings = np.array(ring_end_indices, dtype=np.uint32)

        # earcut.triangulate_float64 → düz index dizisi döner
        indices = earcut.triangulate_float64(vertices, rings)

        if len(indices) == 0:
            return vertices, np.array([]).reshape(0, 3).astype(np.uint32)

        faces = indices.reshape(-1, 3).astype(np.uint32)
        return vertices, faces

    # -------------------------
    # EXTRUDE → 3D MESH
    # -------------------------
    def build_mesh(self, text):
        path = self.text_to_path(text)
        contours = self.path_to_contours(path)

        if not contours:
            contours = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

        # Konturları dış + delik olarak grupla
        groups = self.group_contours(contours, path)

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for outer, holes in groups:
            # --- Ön/arka kapak triangülasyonu (earcut) ---
            cap_verts_2d, cap_faces = self.triangulate_polygon_with_holes(outer, holes)
            n_cap = len(cap_verts_2d)

            if n_cap < 3 or len(cap_faces) == 0:
                continue

            # 3D vertex'ler: ön (z=0) ve arka (z=depth)
            front_verts = np.c_[cap_verts_2d, np.zeros(n_cap)]
            back_verts = np.c_[cap_verts_2d, np.ones(n_cap) * self.depth]

            all_vertices.append(front_verts)
            all_vertices.append(back_verts)

            # Ön kapak yüzleri
            all_faces.append(cap_faces + vertex_offset)

            # Arka kapak yüzleri (ters winding → normal dışarı baksın)
            back_faces = cap_faces + vertex_offset + n_cap
            back_faces = back_faces[:, ::-1]
            all_faces.append(back_faces)

            # --- Yan duvarlar ---
            # Dış kontur + tüm delik konturları için yan duvar oluştur
            contour_list = [outer] + holes
            side_vert_offset = 0

            for contour in contour_list:
                nc = len(contour)
                for i in range(nc):
                    i0 = vertex_offset + side_vert_offset + i
                    i1 = vertex_offset + side_vert_offset + (i + 1) % nc

                    # Ön→arka quad (iki üçgen)
                    all_faces.append(np.array([[i0, i1, i0 + n_cap],
                                               [i1, i1 + n_cap, i0 + n_cap]]))
                side_vert_offset += nc

            vertex_offset += 2 * n_cap  # front + back

        if not all_vertices:
            # Fallback
            return (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                    np.array([[0, 1, 2]], dtype=np.uint32))

        vertices = np.vstack(all_vertices).astype(np.float32)
        faces = np.vstack(all_faces).astype(np.uint32)

        return vertices, faces


# =========================================
# ANA PROGRAM
# =========================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # OpenGL 3D görünüm penceresi
    view = gl.GLViewWidget()
    view.setWindowTitle("TextMesh3D — 3B Metin")
    view.resize(1200, 800)

    # Kamera ayarları
    view.setCameraPosition(distance=300, elevation=25, azimuth=-45)

    # Grid (zemin ızgarası)
    grid = gl.GLGridItem()
    grid.setSize(500, 500, 1)
    grid.setSpacing(10, 10, 1)
    view.addItem(grid)

    # 3D metin oluştur
    text_obj = TextMesh3D(view, text="HELLO", pos=(-100, 0, 0), depth=15)

    view.show()
    sys.exit(app.exec_())
