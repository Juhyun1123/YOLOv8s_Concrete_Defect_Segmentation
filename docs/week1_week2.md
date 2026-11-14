# YOLOv8s_Concrete_Defect_Segmentation

지하 콘크리트 구조물 결함 탐지 프로젝트의 1·2주차 실습 코드입니다.
PPT 코드를 항목별 함수로 분리했으며, `Code/training.py`를 실행하면 2주차 실습이 시작됩니다.
3주차 모델 비교는 [저장소 루트 README](../README.md)를 참고합니다.

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
4. `Code/training.py`를 열고 오른쪽 위 **Run Python File** 버튼으로 실행합니다.

## 데이터 배치

강의에서 제공된 실습 데이터의 이미지와 라벨을 다음 위치에 복사합니다.
원본 데이터 약 3.7GB는 `.gitignore`로 제외되어 저장소를 내려받아도 포함되지 않습니다.

```text
YOLOv8s_Concrete_Defect_Segmentation/
├── Code/
│   └── training.py
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
.\.venv\Scripts\python.exe -c "from Code.training import run_week1; run_week1()"
```

| 함수 | 수행 내용 |
| --- | --- |
| `setup_environment()` | torch 버전, CUDA 사용 가능 여부, GPU 이름, YOLOv8s-Seg 모델 로드 확인 |
| `check_and_acquire_data()` | 영상 수, 예시 경로, split별 영상 수, 첫 번째 영상 해상도 출력 |
| `observe_images()` | 무작위 최대 9장으로 `samples9.png` 저장, 첫 번째 영상의 밝기 평균·표준편차 출력 |

첫 실행 시 `yolov8s-seg.pt`가 프로젝트 루트에 자동 다운로드될 수 있습니다.
샘플 그림도 프로젝트 루트에 저장됩니다.
모델 가중치, 생성된 그림, 학습 결과, 캐시, `.venv`, 원본 이미지·라벨은 Git 추적에서 제외합니다.

## 2주차 실습 실행

`Step3_2주차_강의와실습.pdf`의 실습 코드를 `Code/training.py`에 함수로 추가했습니다.
제출물 ①의 직접 주석을 마친 뒤 프로젝트 루트에서 아래 명령으로 재학습을 실행합니다.

```powershell
.\.venv\Scripts\python.exe Code\training.py
```

실행 순서는 환경 확인 → 직접 주석 결과 확인 → 포맷 확인 → 무결성 점검 →
촬영지 단위 분할 → 모델 학습 → 학습 곡선 확인입니다.
이 명령은 PPT의 학습 설정으로 최대 60 epoch 학습을 실행합니다.

| 단계 | 함수 | PPT의 코드·내용 |
| --- | --- | --- |
| 1 폴리곤 주석 | `check_polygon_annotation_results()` | PDF 9쪽의 CVAT/Roboflow 직접 주석, 제출물 ①의 영상·인스턴스 수 확인 |
| 2 포맷 확인 | `check_label_format()` | PDF 11쪽의 첫 줄 토큰 수와 segmentation 포맷 확인 코드 |
| 3 무결성 점검 | `check_data_integrity()` | PDF 13쪽의 클래스·좌표 점검 코드와 제출물 ③의 대응·빈 파일 점검 |
| 4 분할과 data.yaml | `get_origin_id()`, `split_dataset()` | PDF 12·14·15쪽의 그룹 분할, 복사, YAML 생성 코드와 촬영지 단위 누수 방지 |
| 5 학습 실행 | `train_model()` | PDF 16쪽의 YOLOv8s-Seg 학습 코드 |
| 6 학습 곡선 확인 | `plot_training_curves()` | PDF 17·18쪽의 segmentation loss·mask mAP 그래프 코드 |

### 직접 주석 결과

CVAT 또는 Roboflow에서 제공 영상 10~20장을 직접 주석하고 **YOLOv8 Segmentation** 포맷으로 내보냅니다.
클래스 순서는 `crack`, `efflorescence`, `spall_detach`입니다.
직접 주석한 export의 이미지와 라벨만 아래 위치에 배치합니다.

```text
Annotations/
├── images/
│   └── 직접 주석한 영상.jpg
└── labels/
    └── 직접 주석한 영상.txt
```

이 함수는 직접 주석한 결과의 영상 수와 클래스별 인스턴스 수를 출력합니다.
해당 결과가 없으면 미완료로 표시하며, 제공된 전체 라벨을 직접 주석한 것으로 집계하지 않습니다.
학습용 제공 데이터 경로는 `Data/images`, `Data/labels`입니다.

### 분할과 점검 결과

같은 촬영지의 연속 프레임과 증강본이 분할 사이에 섞이지 않도록 파일명의 첫 부분을 촬영지 코드로 사용합니다.
직접 주석용 10장을 고른 촬영지는 학습셋에 배정했습니다.
seed 42로 촬영지 배정을 비교하고, 각 분할에 세 클래스가 포함되는 후보 중 증강 전 원본 수가 PPT의 80:10:10 비율에 가까운 배정을 사용합니다.
검증·테스트에는 원본 ID별 증강 전 이미지 한 장씩만 포함합니다.

제공 이미지 4,116장 중 0바이트 파일 63장은 분할에서 제외했습니다.
새 학습 경로는 `dataset_split_clean/data.yaml`입니다.
프로젝트 루트에서 분할만 다시 검사하려면 아래 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -c "from Code.training import check_data_integrity, split_dataset; split_dataset(check_data_integrity())"
```

| 분할 | 영상 수 | 증강 전 원본 수 | 촬영지 코드 수 |
| --- | ---: | ---: | ---: |
| train | 3,545 | 160 | 8 |
| val | 19 | 19 | 5 |
| test | 19 | 19 | 4 |

분할 간 중복 촬영지 코드, 원본 ID, 동일한 이미지 파일은 각각 0건입니다.
기존 `dataset_split`으로 학습한 `runs/step3` 결과에는 분할 간 중복 영상의 영향이 포함됩니다.

### PPT 코드에서 보완한 부분

- 현재의 `Data/images/train|val|test`와 `Data/labels/train|val|test` 경로를 읽도록 재귀 탐색과 대응 경로를 지정했습니다.
- PDF 13쪽의 정의되지 않은 `G()`는 `glob.glob(..., recursive=True)`로 연결했습니다.
- 제출물 ①을 위해 직접 주석 결과의 영상 수와 클래스별 인스턴스 수 출력을 추가했습니다.
- 제출물 ③을 위해 영상–라벨 대응, 빈 라벨, 잘못된 토큰 형식, 숫자 형식, 실제 발견된 0바이트 영상을 점검합니다.
- 제출물 ④를 위해 `_jpg.rf.<hash>`, `_aug`, `_augfill`, `_efflo_*` 표기를 같은 원본 ID로 묶고, 검증·테스트의 증강 영상을 제외합니다.
- 실제 분할 결과에서 촬영지 코드, 원본 ID, 동일한 이미지 파일의 중복과 증강 영상 배치를 점검합니다.
- 그래프는 `train_model()`이 반환한 실제 결과 폴더의 `results.csv`를 읽습니다.

### 학습 설정과 산출물

`epochs=60`, `imgsz=960`, `batch=4`, `optimizer="AdamW"`, `lr0=0.00015`,
`mosaic=0.0`, `mixup=0.0`, `patience=20`, `seed=42`를 PPT와 동일하게 사용합니다.
새 학습 결과의 기본 폴더는 `runs/step3_clean`입니다.
PPT의 안내에 따라 CUDA out of memory가 발생하면 `batch=2`로 조정할 수 있습니다.

- 학습 가중치: 실제 학습 결과 폴더의 `weights/best.pt`
- 학습 기록: 실제 학습 결과 폴더의 `results.csv`, `results.png`
- PPT 그래프 코드의 결과: 실제 학습 결과 폴더의 `curve.png`

제출물 ⑥의 설정 근거는 PPT의 다음 내용입니다.

| 설정 | PPT의 근거 |
| --- | --- |
| 입력 크기 960 | 가는 균열의 세부 정보 보존을 고려해 입력 크기 960을 사용한다. |
| Mosaic 비활성화 | 콘크리트 표면의 공간적 맥락 왜곡을 줄이기 위해 비활성화 |
| MixUp 비활성화 | 결함 경계의 해석 가능성을 유지하기 위해 비활성화 |

설치 방식 참고: [Ultralytics 설치 안내](https://docs.ultralytics.com/quickstart),
[PyTorch 설치 안내](https://pytorch.org/get-started/locally/).
