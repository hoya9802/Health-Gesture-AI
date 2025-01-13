![header](https://capsule-render.vercel.app/api?type=soft&section=header&text=Health%20Gesture%20AI&fontSize=45)
# 개인 프로젝트

## 개발 배경
 - 사회적 거리두기가 요구됨에 따라 사람들이 혼자 집에서 홈트레이닝을 많이 하는 상황
 - 정확한 운동 횟수와 운동자세를 스스로 확인하는 것이 힘듦
 - 혼자 하면 옆에서 봐주는 사람이 없어서 의욕이 저하됨
 - 과학적으로 규칙적인 홈트레이닝는 코로나19 면역력을 높임

## 파일 구조
```shell
project/
├── MP_Data_ver2/       # 제스처 손바닥 좌표가 저장되어 있는 디렉토리
│   ├── add_count       # 목표치 숫자를 추가하기 위한 손동작 좌표가 들어 있는 디렉토리
│   ├── nothing         # 무의미한 손동작 좌표가 들어 있는 디렉토리
│   └── reset_count     # 목표치 숫자를 리셋하기 위한 손동작 좌표가 들어 있는 디렉토리
├── model/              
│   └── best-model.h5   # 모델 학습 후 가중치가 저장되어 있는 .h5파일 
├── README.md
├── main.py             # 프로젝트 실행 파일 (Gesture + Squat)
├── squat.py            # Squat 테스트 파일
└── train_model.ipynb   # Gesture data를 수집 + DL 모델 학습 파일
```

## Data Collection
 - 제스처 인식을 하기 위한 손바닥의 관절 좌표 데이터가 필요
 - 운동 자세를 체크하기 위한 전신 관절 좌표 데이터가 필요
 - 데이터를 수집하기 위한 Webcam 필요

### Mediapipe Module

<img src="https://github.com/user-attachments/assets/413a0549-cfaa-44c3-8440-3702337aaaa0" alt="image" width="50%">

 - Module 안에 내장된 고속 ML를 이용하여 빠르게 원하는 결과를 얻을 수 있다.
 - 통합 솔루션을 통한 안드로이드, ios, destop/cloud등에서도 작동한다.
 - 최첨단 ML 솔루션으로 이미 입증된 프레임워크

### 손바닥 관절 데이터 수집

<img src="https://github.com/user-attachments/assets/6a3a65ba-4087-4395-8a39-75baaed94114" alt="image" width="50%">

 - 손의 좌표를 Webcam으로 실시간으로 받아와 .npy 파일로 만든 후, 해당 gesture에 해당하는 폴더에 저장
 - .npy 파일들을 모아서 폴더를 만들고 해당 폴더를 Labelling을 실시
 - 20개의 frame으로 파일을 1개의 폴더로 만들고 이 방식으로 40개의 데이터를 생성

### Data Labelling

<img src="https://github.com/user-attachments/assets/1751d59f-ad2f-4412-98bd-244d0dbb3811" alt="image" width="50%">

사진에서 볼 수 있듯이 각각의 Gesture에는 40개의 파일이 있고, 각각의 파일에는 20 Frame의 Gesture data가 들어있습니다.

### 데이터 구성

<img src="https://github.com/user-attachments/assets/907b5bff-6ea0-4ced-b934-96c75bea3205" alt="image" width="50%">

 - 위에 사진을 보면 손바닥의 좌표는 총 21개
 - 손바닥 한개의 좌표에 x,y,z 총 3개의 값이 존재
 - 손은 보통 2개(왼손, 오른손)
 - 좌표들을 .npy 파일에 저장할때 flatten을 사용하여 값을 1차원으로 만든 후 저장
 - 따라서 바로 위 사진을 보면 오른손에 해당하는 값 21x3 = 63, 왼손에 해당하는 값 21x3 = 63 그리고 이걸 코드에서 concatenate를 하기에 총 shape은 126
 
위 내용은 [train_model.ipynb](https://github.com/hoya9802/Health-Gesture-AI/blob/master/train_model.ipynb)의 Keypoints using MP Holistic과 Extract Keypoint Values에서 확인 가능

### Gesture data 수집 과정

![제스처데이터수집 (1)](https://github.com/user-attachments/assets/dd5f390b-72c2-4cb0-9ee0-4617eb7a2e46)

### 전신 데이터 수집

<img src="https://github.com/user-attachments/assets/86946bd5-cc1b-4cb1-97d8-5b2391a658a3" alt="image" width="50%">

 - Squat 동작을 판별하기 위해서 Mediapipe에서 제공하는 pose를 사용하여 신체에 관절데이터를 받음
 - 이때 23, 25, 27번 좌표의 사이 각도를 통해서 Squat를 정상적으로 하는지 판별

### 관절 사이각

<img src="https://github.com/user-attachments/assets/47600ef8-b6aa-4b49-9fe8-c3d1dc4bf10a" alt="image" width="50%">

 - 제2코사인 공식을 통해서 무릅 사이 각도를 계산
 - 처음에 관절의 사이각이 160도 이상인 상태에서 70도 이하로 내려가면 Down 그리고 다시 160도 이상으로 올라가면 Up으로 처리하여 개수 카운팅을 진행
 - 개수가 정상처리 될 때마다 beep sound를 울려서 사용자에게 알림

위 내용은 [squat.py](https://github.com/hoya9802/Health-Gesture-AI/blob/master/squat.py)에서 확인 할 수 있습니다.

### Squat 구현

![스쿼트 테스트 (1)](https://github.com/user-attachments/assets/cce21f34-48f9-4664-9d1d-c64225c9101c)

## Modeling

### LSTM
<img src="https://github.com/user-attachments/assets/7fe45707-6b22-441f-988a-daf39eb6ac93" alt="image" width="50%">

 - RNN에 비해 LSTM은 (Cell) state가 있어서 이전 Sequence에 있었던 값들이 Input Gate를 통해서 저장이 되어 값의 보존이 향상
 - Forget Gate를 통해서 과거의 의미가 없다고 판단되는 값들은 망각

### Model Summary

#### LSTM

<img src="https://github.com/user-attachments/assets/0963a2c4-0049-4e2c-9746-9b870da3dc83" alt="image" width="50%">

#### RNN

<img src="https://github.com/user-attachments/assets/87a865be-ed40-4d76-a877-c19a94605ee5" alt="image" width="50%">

#### Activation Function
Gradient Vanishing Issue을 해결하고, 데이터에는 음수의 데이터가 들어가지 않기 때문에 Relu를 사용

## Model Performance

| Error Metric    | LSTM | RNN |
|-----------------|---------|---------|
| **Accuracy** | 96.13%   | 97.22%   |
| **Loss** | 0.09998    | 0.06373    |

두 모델의 큰 성능 차이는 없었지만 데이터 수가 적고 Sequence의 길이 또한 짧아서 연산이 상대적으로 복잡한 LSTM보다 RNN에서 1%정도 향상된 결과를 얻을 수 있었습니다.

## Final Result

![final result (1) (1)](https://github.com/user-attachments/assets/7ac1489c-e3d0-4ebb-8763-a701bc3a2515)

## Conclusion

### 문제점

 - Gesture를 분류하는 종류와 데이터 수가 적어, 주먹을 쥐는 동작의 개수를 5개 늘려주는 Gesture이지만 그냥 주먹을 쥐고 있는 경우에 대한 학습이 없어 마찬가지로 5개를 늘려주는 경향이 있음
 - Squat에 경우 무릎만 보지 않고 허리, 목 등 여러 요인들도 작용하지만 이번에는 무릎 사이각으로만 모델 구현

### 기대 효과

 - Gesture 인식을 통한 수화 번역기를 만들어 귀가 들리지 않는 사람과에 소통이 가능해 질것입니다.
 - 앞으로 키보드나 마우스가 없어도 Gesture를 통한 즉각적인 화면과 상호작용을 통해 원하는 것을 보고 쓸 수 있을 것이다.
 - 4차 산업 중 하나인 VR, AR에서 지금은 손에 기구를 잡고 하지만 앞으로는 카메라만 앞에 설치하면 VR속에서 보다 더 현실감을 느낄 수 있을 것이다.

## Reference
 - https://arxiv.org/pdf/1712.10136v1.pdf
 - https://google.github.io/mediapipe/solutions/holistic.html
 - https://www.crocus.co.kr/1635
