![header](https://capsule-render.vercel.app/api?type=soft&section=header&text=Health%20Gesture%20AI&fontSize=45)

## File Structure
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

