import cv2
import torch
import urllib.request
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
def dishow2(disp):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth2Disparity image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.plot
    plt.show()

def DPT(filename,model_type):
    #url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    #urllib.request.urlretrieve(url, filename)
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    img = cv2.imread(filename)
    #img= cv2.resize(img,(512,512))
    h,w,c= img.shape
    img = cv2.resize(img, dsize=(w//4,h//4), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.imshow(img)
    # plt.plot
    # plt.show()
    input_batch = transform(img).to(device) # 전처리 과정.

    with torch.no_grad():
        prediction = midas(input_batch) # input 이미지 output depth 이미지

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output
# output=DPT('D:/LF_DATA_Test/1/OBJ1/img_2_2.png',"DPT_Hybrid" )
#
# dishow2(output)
#plt.imshow(output)
# plt.show()