import open3d as o3d  # open3D 모듈 추가, 이름은 o3d로 정의
import matplotlib.pyplot as plt  # 그래프 그리기 모듈 추가 plt로 정의
from PIL import Image
import os
import sys
import math # 표준 수학 함수 추가
import cv2  # openCV 모듈 추가
from imageio import imread  # image read 모듈 추가
import numpy as np  # 넘파이(행렬 계산) 추가
from skimage.transform import rescale
from skimage import io
import torch
import torch.nn as nn
import argparse
from inference import Inference
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms
# scikit-image (이미지처리 라이브러리) 에서 rescale 추가

# point cloud
def get_index(color):
    ''' Parse a color as a base-256 number and returns the index
    # 색을 256까지로 나눠서 분석, index 반환
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
        (0, 0, 0) 과 같은 배열로 나눈 것 -> 색
    Returns:
        index: an int containing the indec specified in 'color'
    '''
    return color[0] * 256 * 256 + color[1] * 256 + color[2]
    # (color0, color1, color2)라는 배열 반환식


def dishow(disp):
    plt.imshow(disp)  # disp라는 이미지를 보여줌
    plt.jet()  # Distance to Camera에 나타나는 color bar의 색 출력
    plt.colorbar(label='Distance to Camera')  # 'Distance to Camera'라는 문구 출력
    plt.title('Depth2Disparity image')  # 'Depth2Disparity image' 라는 title 출력
    plt.xlabel('X Pixel')  # 'X Pixel'이라는 문구 출력
    plt.ylabel('Y Pixel')  # 'Y Pixel'이라는 문구 출력
    plt.plot  # 그래프 그리기
    plt.show(block=False)  # 이미지 출력
    plt.pause(5)
    plt.close()

def get_uni_sphere_xyz(H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i + 0.5) / W * 2 * np.pi * -1
    v = ((j + 0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz

def mesh(d, rgb):
    H, W = d.shape[:2]
    # 정규화
    d = (d - d.min()) / (d.max() - d.min())
    d = (1 / (d + 0.1))

    xyz = np.multiply(d, get_uni_sphere_xyz(H, W))

    xyzrgb = np.concatenate([xyz, rgb / 255.], 2)
    xyzrgb = xyzrgb.reshape(-1, 6)

    pcd = o3d.geometry.PointCloud()
    print("Generating point cloud")

    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

    o3d.visualization.draw_geometries([pcd])  # 포인트 클라우드 시각화

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)  # 포인트 클라우드 점 간의 평균 거리

    radius_parameter = 1

    radius = radius_parameter * avg_dist

    bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

    bp_mesh.triangle_normals = o3d.utility.Vector3dVector([])

    o3d.io.write_triangle_mesh("result" + '.obj', bp_mesh)  # mesh를 obj 형태로 변환하여 저장

    o3d.visualization.draw_geometries([bp_mesh])  # 메쉬 시각화

    return pcd

def inference_main(config):
    inference = Inference(config)
    inference.inference()

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', help='Data_path', type=str, default='./sample')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--checkpoint_path', type=str, default='./Joint_3D60_Fres.pth')

parser.add_argument('--Input_Full', help='Use input of full angular resolution', action='store_true',
                        default='Input_Full')

parser.add_argument('--pred_height', type=int, default=512)
parser.add_argument('--pred_width', type=int, default=1024)

parser.add_argument('--output_path', help='path where inferenced samples saved', type=str, default='./output')

config = parser.parse_args()
inference_main(config)

rgb = cv2.imread('./sample\\2.png')

rgb = cv2.resize(rgb, (1024, 512)) # image resizing

depth = imread('./output\\sample\\2_disp.png', as_gray=True).astype(np.int16)  # depth 입력 - grayscale 변환

dishow(depth)

cv2.imwrite("./grayscale.png", depth)  # grayscale 이미지 저장

depth = cv2.imread('./grayscale.png')

output_path = 'C:\\Users\\imher\\PycharmProjects\\final\\3D_meshmaker\\pointcloud_test-main\\'

mesh(depth, rgb) # 메쉬 생성 함수

# o3d.visualization.draw_geometries([pcd]) # 포인트 클라우드 결과물 출력

# o3d.io.write_point_cloud("test" + ".ply", pcd)
