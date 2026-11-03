# YOLOv8s_Concrete_Defect_Segmentation

지하 콘크리트 구조물 결함 탐지 프로젝트의 1주차 실습 코드입니다.
강의 코드의 환경 확인, 데이터 확인, 영상 관찰을 각각 함수로 분리했습니다.

## 실행 환경

- Windows 64-bit, Python 3.14 (검증 버전: 3.14.7)
- 패키지는 프로젝트 폴더 안의 `.venv`에 설치합니다.
- PyTorch는 CUDA 12.8 빌드를 사용합니다. NVIDIA GPU 사용 여부는 실행 시 출력됩니다.
- 첫 설치에는 패키지 다운로드를 위한 인터넷 연결이 필요합니다.

## GitHub에서 받은 후 설치

Python 3.14 64-bit와 Git을 설치한 상태에서 다음 명령을 실행합니다.

```powershell
git clone https://github.com/Juhyun1123/YOLOv8s_Concrete_Defect_Segmentation.git
cd YOLOv8s_Concrete_Defect_Segmentation
.\setup.bat
```

`setup.bat`는 `.venv`를 만들고 `requirements.txt`에 전체 버전이 고정된 패키지를 설치합니다.
가상환경 폴더 자체는 GitHub에 올리지 않으며, 다른 PC에서도 이 명령으로 생성합니다.
가상환경을 별도로 활성화하지 않고 아래의 Python 경로로 실행할 수 있습니다.

## VS Code에서 실행

1. **파일 → 폴더 열기**로 저장소의 루트 폴더를 엽니다.
2. 권장 확장인 Microsoft의 **Python** 확장을 설치하거나 활성화합니다.
3. 기본 인터프리터는 프로젝트의 `.venv/Scripts/python.exe`로 지정되어 있습니다.
   이미 다른 인터프리터를 선택한 적이 있다면 `F1` → **Python: Select Interpreter** →
   **Enter interpreter path...**에서 프로젝트의 `.venv/Scripts/python.exe`를 한 번 선택합니다.
4. `Code/main.py`를 열고 오른쪽 위 **Run Python File** 버튼으로 실행합니다.

## 데이터 배치

강의에서 제공된 실습 데이터의 이미지와 라벨을 다음 위치에 복사합니다.
원본 데이터 약 3.7GB는 `.gitignore`로 제외되어 저장소를 내려받아도 포함되지 않습니다.

```text
YOLOv8s_Concrete_Defect_Segmentation/
├── Code/
│   └── main.py
├── Data/
│   ├── data.yaml
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── requirements.txt
└── setup.bat
```

1주차 코드는 `Data/images` 아래의 `.jpg` 파일을 재귀적으로 확인합니다.
제공된 데이터 구성은 train 4,036장, val 40장, test 40장으로 총 4,116장입니다.
`Data/data.yaml`에는 다른 PC의 절대 경로를 저장하지 않습니다.

## 1주차 실습 실행

프로젝트 폴더에서 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe Code\main.py
```

| 함수 | 수행 내용 |
| --- | --- |
| `setup_environment()` | torch 버전, CUDA 사용 가능 여부, GPU 이름, YOLOv8s-Seg 모델 로드 확인 |
| `check_and_acquire_data()` | 영상 수, 예시 경로, split별 영상 수, 첫 번째 영상 해상도 출력 |
| `observe_images()` | 무작위 최대 9장으로 `samples9.png` 저장, 첫 번째 영상의 밝기 평균·표준편차 출력 |

첫 실행 시 `yolov8s-seg.pt`가 프로젝트 루트에 자동 다운로드될 수 있습니다.
샘플 그림도 프로젝트 루트에 저장됩니다.
모델 가중치, 생성된 그림, 학습 결과, 캐시, `.venv`, 원본 이미지·라벨은 Git 추적에서 제외합니다.

설치 방식 참고: [Ultralytics 설치 안내](https://docs.ultralytics.com/quickstart),
[PyTorch 설치 안내](https://pytorch.org/get-started/locally/).
