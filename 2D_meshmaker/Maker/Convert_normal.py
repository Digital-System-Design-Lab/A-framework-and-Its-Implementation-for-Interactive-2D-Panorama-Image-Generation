import cv2
import open3d as o3d
import open3d.cpu.pybind.io
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import xatlas
from imageio import imread
from DPT import DPT
from skimage.transform import resize
import imageio

from skimage.filters import gaussian, sobel
from skimage.color import rgb2grey

from scipy.interpolate import griddata
mesh = o3d.io.read_triangle_mesh("./2D_PCL_0801/output/d/d_mesh/111.obj")
zzz=mesh.vertex_normals
zzz= np.asarray(zzz)
zzz[:,2]=zzz[:,2]*(-1)
mesh.vertex_normals=o3d.utility.Vector3dVector(zzz)
o3d.io.write_triangle_mesh("./2D_PCL_0801/output/d/d_mesh/copy_of_crate.obj",
                               mesh,
                               write_triangle_uvs=True)
print(mesh)