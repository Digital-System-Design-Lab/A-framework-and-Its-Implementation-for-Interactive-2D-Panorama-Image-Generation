# A-framework-and-Its-Implementation-for-Interactive-2D-Panorama-Image-Generation
## 상호작용 가능한 2차원, Panorama 영상 생성을 위한 프레임워크와 그 구현

본 프로젝트는 RTX 3090(cuda 11.3) 환경에서 개발되었음.

------------

### 1. 프로젝트 진행 과정

- 이미지 입력
	- 2D 이미지의 경우 1024x512 해상도의 임의의 이미지, 360 Panorama 이미지의 경우 360 카메라를 통해 촬영
		
- Depth Estimation
	- 위의 이미지의 Depth를 Estimate 후 Depth Map으로 변환
	- Depth Estimation에는 Joint_S3D_Fres모델 활용
		- 해당 논문 참조 : Improving 360 Monocular Depth Estimation via Non-local Dense Prediction Transformer and Joint Supervised and Self-supervised Learning,
											Ilwi Yun, Hyuk-Jae Lee, Chae Eun Rhee,
											https://github.com/yuniw18/Joint_360depth
    
- Point Cloud 추출 및 Mesh 생성
	- Depth map 기반으로 추출한 Point Cloud에서 Ball Pivoting 기법을 통하여 Mesh 생성
	
	
----------
### 2. 프로젝트 실행방법

- 해당 Repository를 git clone

- 개발 환경 복제
~~~bash
conda env create --file environment.yaml
conda activate environment
~~~

- 상단 Depth Estimation GitHub에서 Joint_S3D_Fres 파일 다운로드 후 inference 폴더로 이동

- main.py 코드 실행

