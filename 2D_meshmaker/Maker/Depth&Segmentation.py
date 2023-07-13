import tensorflow as tf
import open3d
from libs import mpi
from libs import nets
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import xatlas
from imageio import imread
from DPT import DPT
def depth_exception(x,y,z,depth):
  vert = []
  vert.append(x)
  vert.append(y)
  vert.append(z)
  depth_except = 0
  for k in range(3):
    if np.round(depth[vert[k]], 1) < 0.001:
      return 1
    else :
      continue
  return 0

def display_mpi(input,layer_num):
  for i in range(layer_num):
    plt.subplot(4, 8, i + 1)
    plt.imshow(input[i])
    plt.axis('off')
    plt.title('Layer %d' % i, loc='left')
  plt.show()
def mpi_gen(in_path):
  # Layers is now a tensor of shape [L, H, W, 4].
  # This represents an MPI with L layers, each of height H and width W, and
  # each with an RGB+Alpha 4-channel image.

  input = tf.keras.Input(shape=(None, None, 3))
  output = nets.mpi_from_image(input)

  model = tf.keras.Model(inputs=input, outputs=output)
  print('Model created.')
  # Our full model, trained on RealEstate10K.
  model.load_weights('single_view_mpi_full_keras/single_view_mpi_full_keras/single_view_mpi_keras_weights')
  print('Weights loaded.')

  plt.rcParams["figure.figsize"] = (20, 10)

  # Input image
  inputfile = in_path
  input_rgb = tf.image.decode_image(tf.io.read_file(inputfile), dtype=tf.float32)
  H, W, _ = input_rgb.shape
  # Generate MPI
  layers = model(input_rgb[tf.newaxis])[0]
  depths = mpi.make_depths(1.0, 100.0, 32).numpy()

  return layers,depths,H,W

def gen_layergroup(layers,depths,layer_num):
  layer_depth = np.empty((layer_num, H, W, 1), dtype=float)
  layer_rgb = np.empty((layer_num, H, W, 3), dtype=float)
  for i in range(layer_num):
    lay = tf.stack([layers[i * 2, ...], layers[i * 2 + 1, ...]], axis=0)
    dep = tf.stack([depths[i * 2], depths[i * 2 + 1]])
    disp = mpi.disparity_from_layers(lay, dep)

    layer_depth[i, ...] = disp

    rgb = tf.stack([layers[i * 2, :, :, :3], layers[i * 2 + 1, :, :, :3]])
    layer_rgb[i, ...] = mpi.rgb_composition(lay, rgb)

  return layer_rgb,layer_depth
def gen_layer_mesh(layer_rgb,layer_depth,H,W,obj_p):
  depth_scaling_factor = 1
  rgb = layer_rgb

  depth = layer_depth * depth_scaling_factor
  depth = depth.reshape(H, W)

  x = np.linspace(1, H, W)
  y = np.linspace(1, H, W)
  z = np.linspace(1, 1, H * W)

  X, Y = np.meshgrid(x, y)
  #Y = Y = np.flip(Y, axis=0)
  pts_xyz = np.stack([X, Y, depth], -1).reshape(-1, 3)
  plt.imshow(rgb)
  plt.show()
  plt.imshow(depth)
  plt.show()

  pts_rgb = rgb.reshape(-1, 3)

  depth = depth.reshape(-1)

  faces = []

  t_factor = 1
  for i in range( 1,H-1):
    for j in range(1, W - 1):

      g = depth_exception(i * H + j, i * H + j + t_factor, (i + t_factor) * H + j, depth)
      if g == 0:
        faces.append([i * H + j, i * H + j + t_factor, (i + t_factor) * H + j])
      g = depth_exception(i * H + j, i * H + j - t_factor, (i - t_factor) * H + j, depth)
      if g == 0:
        faces.append([i * H + j, i * H + j - t_factor, (i - t_factor) * H + j])
  #pts_xyz[:,2]=pts_xyz[:,2]*depth_scaling_factor
  scene = open3d.geometry.TriangleMesh()
  scene.vertices = open3d.utility.Vector3dVector(pts_xyz)
  scene.vertex_colors = open3d.utility.Vector3dVector(pts_rgb)
  scene.triangles = open3d.utility.Vector3iVector(faces)
  scene.compute_vertex_normals()


  open3d.io.write_triangle_mesh(obj_p,
                                scene,
                                write_triangle_uvs=True,
                                write_vertex_normals=True,
                                write_vertex_colors=True
                                )
  # open3d.visualization.draw_geometries([
  #   scene,
  #   open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
  # ], mesh_show_back_face=False)
def Broad_1_to_3(img):
  H,W = img.shape
  img2 = np.zeros((H,W,3))
  img2[:, :, 0] = img
  img2[:, :, 1] = img
  img2[:, :, 2] = img
  return img2
def my_xatlas(input_p,output_p):


  # We use trimesh (https://github.com/mikedh/trimesh) to load a mesh but you can use any library.
  mesh = trimesh.load_mesh(input_p)

  # The parametrization potentially duplicates vertices.
  # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
  # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
  # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
  vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

  # Trimesh needs a material to export uv coordinates and always creates a *.mtl file.
  # Alternatively, we can use the `export` helper function to export the mesh as obj.
  xatlas.export(output_p, mesh.vertices[vmapping], indices, uvs)

def dishow(disp):
  plt.imshow(disp)
  plt.jet()
  plt.colorbar(label='Distance to Camera')
  plt.title('Depth2Disparity image')
  plt.xlabel('X Pixel')
  plt.ylabel('Y Pixel')
  plt.plot
  plt.show()

label = np.load('label.npy')
unique=np.unique(label)
layers,depths,H,W= mpi_gen('input.png')
rgb=imread('input.png')
dp=DPT('input.png',"DPT_Hybrid")
### DPT Version
dp_range = mpi.make_depths(1.0, 100.0, 32).numpy()
dp = 1/(dp+0.1)
dp = (dp-dp.min())/(dp.max()-dp.min())
dp= dp*(dp_range.max()-dp_range.min())+dp_range.min()
dishow(dp)
disparity = mpi.disparity_from_layers(layers, depths)

#refine_dis = (inf_depth - inf_depth.min()) / (inf_depth.max() - inf_depth.min())
#refine_dis = refine_dis * (back_disp.max() + back_disp.min()) + back_disp.min()
#refine_dis = refine_dis
disparity =disparity[..., 0]
dishow(disparity)
disparity=dp
### DPT version
for i in range(len(unique)):
    print(i)
    X =np.where(label==int(i),True,False)
    #X=X.reshape(X,(-1,2))
    #plt.imshow(disparity[..., 0]*X)
    #zzz=rgb * Broad_1_to_3(X)
    depth = disparity * X
    X = np.expand_dims(X, axis=2)
    #
    # plt.imshow(rgb*X)
    # plt.axis('off')
    # plt.title('Synthesized disparity')
    # plt.show()
    mesh_p = 'dpt/layer_mesh_' + str(i) + '.obj'
    xa_p = 'dpt/layer_mesh_' + str(i) + '_xa.obj'
    print(np.array(depth).max())
    #print(depth_scale)
    continue
    gen_layer_mesh(rgb*X, np.array(depth), H, W, mesh_p)
    try:
     my_xatlas(mesh_p, xa_p)
    except:
     continue
print(unique)
# layers,depths,H,W= mpi_gen('input.png')
# disparity = mpi.disparity_from_layers(layers, depths)
# plt.imshow(disparity[..., 0])
# plt.axis('off')
# plt.title('Synthesized disparity')
# plt.show()