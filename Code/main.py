"""STEP 3 1주차: 환경 구축, 데이터 확인 및 확보, 영상 관찰."""

# 프로젝트 루트에서 실행할 환경 구축 및 실행 명령:
# .\setup.bat
# .\.venv\Scripts\python.exe Code/main.py

import glob
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_environment():
    """torch 버전, GPU 사용 가능 여부와 YOLO 모델 로드를 확인한다."""
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    model = YOLO(str(PROJECT_ROOT / "yolov8s-seg.pt"))
    print("model loaded")


def check_and_acquire_data():
    """제공된 실습용 데이터셋의 영상 수와 경로를 확인한다."""
    imgs = sorted(
        glob.glob(str(PROJECT_ROOT / "Data" / "images" / "**" / "*.jpg"), recursive=True)
    )
    print("영상 수:", len(imgs))
    print("예시:", imgs[0] if imgs else "없음")

    # 이번 주 제출물의 데이터 요약: 데이터 구성과 해상도.
    for split in ("train", "val", "test"):
        split_imgs = glob.glob(
            str(PROJECT_ROOT / "Data" / "images" / split / "**" / "*.jpg"),
            recursive=True,
        )
        print(f"{split} 영상 수:", len(split_imgs))
    if imgs:
        im = cv2.imread(imgs[0])
        height, width = im.shape[:2]
        print("첫 번째 영상 해상도:", f"{width} x {height}")

    return imgs


def observe_images(imgs):
    """최대 9장의 샘플 그림을 저장하고 첫 번째 영상의 밝기를 확인한다."""
    n = min(9, len(imgs))
    sample = random.sample(imgs, n)
    fig, ax = plt.subplots(3, 3, figsize=(10, 9))
    axes = ax.ravel()
    for a in axes:
        a.axis("off")
    for a, p in zip(axes, sample):
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        a.imshow(im)
        a.set_title(Path(p).name[:20], fontsize=7)
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "samples9.png"), dpi=110)

    # 첫 번째 영상의 밝기 통계 확인.
    if imgs:
        gray = cv2.imread(imgs[0], 0)
        print("밝기 평균:", round(gray.mean(), 1))
        print("밝기 표준편차:", round(gray.std(), 1))


if __name__ == "__main__":
    setup_environment()
    imgs = check_and_acquire_data()
    observe_images(imgs)
