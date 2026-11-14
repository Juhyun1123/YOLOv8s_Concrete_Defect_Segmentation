"""STEP 3 1·2주차 실습. 기본 실행은 2주차 주석 점검·분할·학습이다."""

# 프로젝트 루트에서 실행할 환경 구축 및 실행 명령:
# .\setup.bat
# .\.venv\Scripts\python.exe Code/training.py

import filecmp
import glob
import hashlib
import random
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_IMG_DIR = PROJECT_ROOT / "Data" / "images"
SRC_LBL_DIR = PROJECT_ROOT / "Data" / "labels"
OUT_ROOT = PROJECT_ROOT / "dataset_split_clean"
ANNOTATION_ROOT = PROJECT_ROOT / "Annotations"
ANNOTATION_SELECTION = SRC_IMG_DIR.parent / "annotation_selection.txt"
SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "val", "test")
ORIGIN_PATTERN = re.compile(
    r"([A-Z]+\d+_\d{4}_\d{2}_\d{2}_\d+_\d+)"
    r"(?:_jpg\.rf\.[0-9a-fA-F]+|_aug[A-Za-z0-9_]*|_efflo_[A-Za-z0-9_]+)?"
)
CLASS_NAMES = {
    0: "crack",
    1: "efflorescence",
    2: "spall_detach",
}


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


def check_polygon_annotation_results():
    """단계 1: 제출물 ①에 필요한 직접 주석 영상·클래스별 인스턴스 수."""
    # PPT 9쪽: CVAT 또는 Roboflow에서 10~20장을 직접 주석하고 export한다.
    # 직접 주석한 export만 Annotations/images, Annotations/labels에 배치한다.
    imgs = sorted((ANNOTATION_ROOT / "images").rglob("*.jpg"))
    lbls = sorted((ANNOTATION_ROOT / "labels").rglob("*.txt"))
    print("직접 주석 영상 수:", len(imgs))
    if not imgs or not lbls:
        print("직접 주석 결과가 없습니다. CVAT 또는 Roboflow 주석 후 다음 위치에 배치하세요:")
        print(ANNOTATION_ROOT)
        return

    counts = {class_id: 0 for class_id in CLASS_NAMES}
    for p in lbls:
        for ln in p.read_text(encoding="utf-8").splitlines():
            v = ln.split()
            if v:
                counts[int(v[0])] += 1
    for class_id, name in CLASS_NAMES.items():
        print(f"직접 주석 {name} 인스턴스 수:", counts[class_id])


def check_label_format():
    """단계 2: PPT 11쪽의 YOLO segmentation 포맷 확인."""
    lbls = sorted(
        glob.glob(str(SRC_LBL_DIR / "**" / "*.txt"), recursive=True)
    )
    print("라벨 파일 수:", len(lbls))
    if not lbls:
        print("라벨 파일이 없습니다.")
    else:
        with open(lbls[0], encoding="utf-8") as f:
            line = f.readline().split()
        print("토큰 개수:", len(line))
        print("앞부분:", line[:9])
        if len(line) == 5:
            print("Detection 포맷입니다.")
        elif len(line) >= 7 and (len(line) - 1) % 2 == 0:
            print("Segmentation 포맷입니다.")
        else:
            print("라벨 형식을 다시 확인하세요.")


def check_data_integrity():
    """단계 3: PPT 13쪽 점검 코드와 제출물 ③의 대응·빈 파일 점검."""
    # PPT의 미정의 G()는 경로에 맞춘 glob.glob(..., recursive=True)로 연결한다.
    bad = set()
    empty = []
    lbls = sorted(glob.glob(str(SRC_LBL_DIR / "**" / "*.txt"), recursive=True))
    for p in lbls:
        has_instance = False
        with open(p, encoding="utf-8") as f:
            for ln in f:
                v = ln.split()
                if not v:
                    continue
                has_instance = True
                # 제출물 ③: 잘못된 토큰 수도 이상 라벨로 기록한다.
                if len(v) < 7 or (len(v) - 1) % 2 != 0:
                    bad.add(p)
                try:
                    if int(v[0]) not in (0, 1, 2):
                        bad.add(p)
                    if any(not 0 <= float(x) <= 1 for x in v[1:]):
                        bad.add(p)
                except ValueError:
                    bad.add(p)
        if not has_instance:
            empty.append(p)
    print("이상 파일:", len(bad))

    # 제출물 ③: 영상–라벨 대응, 빈 파일 및 실제 발견된 0바이트 영상 확인.
    images = sorted(SRC_IMG_DIR.rglob("*.jpg"))
    if not images:
        raise FileNotFoundError("Data/images 폴더에 jpg 이미지가 없습니다.")
    image_ids = {p.relative_to(SRC_IMG_DIR).with_suffix("") for p in images}
    label_ids = {Path(p).relative_to(SRC_LBL_DIR).with_suffix("") for p in lbls}
    missing = image_ids - label_ids
    orphan = label_ids - image_ids
    unusable = [p for p in images if p.stat().st_size == 0]
    print("영상 수:", len(images))
    print("라벨이 없는 영상:", len(missing))
    print("영상이 없는 라벨:", len(orphan))
    print("빈 라벨 파일:", len(empty))
    print("0바이트 영상:", len(unusable))
    for p in sorted(bad):
        print("[확인 필요] 이상 라벨:", p)
    for p in unusable:
        print("[학습용 복사에서 제외] 0바이트 영상:", p.name)
    if bad or missing or orphan:
        raise ValueError("영상–라벨 대응과 이상 라벨을 확인한 뒤 다시 실행하세요.")

    # 0바이트 영상을 복구한 것으로 간주하지 않고 학습용 복사 대상에서 제외한다.
    return [p for p in images if p.stat().st_size > 0]


def get_origin_id(stem):
    """PPT 12쪽: 현재 파일명 규칙에 맞춰 같은 원본의 ID를 추출한다."""
    match = ORIGIN_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"촬영지와 원본 ID를 확인할 수 없는 영상: {stem}")
    return match.group(1)


def is_augmented(stem):
    suffix = stem[len(get_origin_id(stem)):]
    return suffix.startswith(("_aug", "_efflo_"))


def source_label(image):
    return SRC_LBL_DIR / image.relative_to(SRC_IMG_DIR).with_suffix(".txt")


def choose_split_sites(groups, originals, reserved):
    """촬영지 분리를 우선하고 원본 수와 클래스 구성을 기준으로 배정한다."""
    all_classes = set(CLASS_NAMES)
    site_classes = {}
    for site, frames in originals.items():
        site_classes[site] = {
            int(line.split()[0])
            for image in frames.values()
            for line in source_label(image).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    total = sum(len(frames) for frames in originals.values())
    targets = [int(total * ratio) for ratio in SPLIT_RATIO[1:]]
    sites = set(groups)
    eligible = sorted(sites - reserved)

    def covered_prefixes(order):
        selected = set()
        classes = set()
        count = 0
        for site in order:
            selected.add(site)
            classes.update(site_classes[site])
            count += len(originals[site])
            if classes == all_classes:
                yield count, selected.copy()

    rng = random.Random(SEED)
    best = None
    best_score = None
    for _ in range(512):
        rng.shuffle(eligible)
        test_candidates = list(covered_prefixes(reversed(eligible)))
        for val_count, val in covered_prefixes(eligible):
            for test_count, test in test_candidates:
                if val & test:
                    continue
                train = sites - val - test
                if set().union(*(site_classes[site] for site in train)) != all_classes:
                    continue
                val_error = abs(val_count - targets[0])
                test_error = abs(test_count - targets[1])
                excluded = sum(len(groups[site]) - len(originals[site]) for site in val | test)
                score = (val_error + test_error, max(val_error, test_error), excluded)
                if best_score is None or score < best_score:
                    best_score = score
                    best = dict(zip(SPLIT_NAMES, (train, val, test)))
    if best is None:
        raise ValueError("직접 주석용 촬영지를 학습셋에 두고 세 클래스를 모두 포함하는 분할을 찾지 못했습니다.")
    return best


def check_split_duplicates(split_images):
    """서로 다른 분할에 촬영지, 원본 또는 동일한 이미지가 있는지 확인한다."""
    seen_sites = {}
    seen_origins = {}
    images_by_size = {}
    for split, images in split_images.items():
        for image in images:
            origin = get_origin_id(image.stem)
            site = origin.split("_", 1)[0]
            if seen_sites.setdefault(site, split) != split:
                raise ValueError(f"분할 간 중복 촬영지: {site}")
            if seen_origins.setdefault(origin, split) != split:
                raise ValueError(f"분할 간 중복 원본: {origin}")
            if split != "train" and is_augmented(image.stem):
                raise ValueError(f"검증·테스트 증강 영상: {image.name}")
            images_by_size.setdefault(image.stat().st_size, []).append((split, image))

    for matches in images_by_size.values():
        if len({split for split, _ in matches}) < 2:
            continue
        seen_hashes = {}
        for split, image in matches:
            with image.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").digest()
            other_split, other_image = seen_hashes.setdefault(digest, (split, image))
            if other_split != split:
                raise ValueError(f"분할 간 동일한 영상: {other_image.name}, {image.name}")


def save_dataset_split(split_images):
    """분할을 임시 폴더에 복사해 검증한 뒤 학습 경로에 배치한다."""
    data_yaml = {
        "path": str(OUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    yaml_text = yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False)
    files = {}
    for split, images in split_images.items():
        for kind, sources in (("images", images), ("labels", [source_label(p) for p in images])):
            entries = {source.name: source for source in sources}
            if len({name.lower() for name in entries}) != len(sources):
                raise ValueError(f"같은 분할에 이름이 겹치는 파일이 있습니다: {kind}/{split}")
            files[Path(kind) / split] = entries

    def verify(root, compare_sources=False):
        for folder, entries in files.items():
            directory = root / folder
            actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
            if actual != entries.keys():
                raise ValueError(f"분할 파일 목록이 계획과 다릅니다: {directory}")
            if compare_sources and any(
                not filecmp.cmp(source, root / folder / name, shallow=False)
                for name, source in entries.items()
            ):
                raise ValueError(f"분할 파일 내용이 원본과 다릅니다: {root / folder}")
        if (root / "data.yaml").read_text(encoding="utf-8") != yaml_text:
            raise ValueError(f"분할 설정을 확인하세요: {root / 'data.yaml'}")
        check_split_duplicates({
            split: sorted((root / "images" / split).rglob("*.jpg"))
            for split in SPLIT_NAMES
        })

    if OUT_ROOT.resolve().parent != PROJECT_ROOT:
        raise ValueError(f"프로젝트 내부의 분할 경로를 사용하세요: {OUT_ROOT}")
    if OUT_ROOT.exists():
        verify(OUT_ROOT, compare_sources=True)
        return
    with TemporaryDirectory(dir=PROJECT_ROOT, prefix=".dataset_split_clean-") as temp_dir:
        staged = Path(temp_dir).resolve() / "split"
        if not staged.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"임시 분할 경로를 확인하세요: {staged}")
        for folder, entries in files.items():
            (staged / folder).mkdir(parents=True)
            for name, source in entries.items():
                shutil.copy2(source, staged / folder / name)
        (staged / "data.yaml").write_text(yaml_text, encoding="utf-8")
        verify(staged)
        staged.rename(OUT_ROOT)


def split_dataset(images):
    """단계 4: PPT 12·14·15쪽의 분할과 data.yaml 생성."""
    groups = {}
    originals = {}
    for image in sorted(images):
        origin = get_origin_id(image.stem)
        site = origin.split("_", 1)[0]
        groups.setdefault(site, []).append(image)
        frames = originals.setdefault(site, {})
        if not is_augmented(image.stem) and (origin not in frames or image.stem == origin):
            frames[origin] = image

    # 직접 주석용으로 골라 둔 촬영지는 학습셋에만 둔다.
    if ANNOTATION_SELECTION.is_file():
        # 원래 분할에 사용한 주석 대상 목록을 저장소에서도 재현한다.
        selected = ANNOTATION_SELECTION.read_text(encoding="utf-8").splitlines()
    else:
        selected = [p.name for p in (ANNOTATION_ROOT / "images").rglob("*.jpg")]
    reserved = {get_origin_id(Path(name).stem).split("_", 1)[0] for name in selected if name}
    split_sites = choose_split_sites(groups, originals, reserved)
    split_images = {}
    for split, sites in split_sites.items():
        # 평가셋은 증강 전 영상만 쓰고 같은 원본은 한 장만 선택한다.
        source = groups if split == "train" else {
            site: frames.values() for site, frames in originals.items()
        }
        split_images[split] = [image for site in sorted(sites) for image in source[site]]

    save_dataset_split(split_images)
    print("\n분할 완료")
    print("data.yaml:", OUT_ROOT / "data.yaml")
    for split, paths in split_images.items():
        origins = {get_origin_id(p.stem) for p in paths if not is_augmented(p.stem)}
        print(f"{split}: 영상 {len(paths)}장, 증강 전 원본 {len(origins)}장, 촬영지 {len(split_sites[split])}곳")
    print("분할 간 중복 촬영지·원본 ID·동일 영상: 0")
    print("검증·테스트 증강 영상: 0")


def train_model(pretrained="yolov8s-seg.pt", run_name="step3_clean"):
    """단계 5: PPT 16쪽의 YOLOv8s-Seg 학습 설정을 그대로 사용한다."""
    model = YOLO(str(PROJECT_ROOT / pretrained))
    model.train(
        data=str(OUT_ROOT / "data.yaml"),
        epochs=60,
        imgsz=960,
        batch=4,
        optimizer="AdamW",
        lr0=0.00015,
        mosaic=0.0,
        mixup=0.0,
        patience=20,
        seed=42,
        project=str(PROJECT_ROOT / "runs"),
        name=run_name,
    )
    # 재실행 시 실제 학습 결과 폴더에 맞춰 곡선 파일을 읽는다.
    return Path(model.trainer.save_dir)


def plot_training_curves(results_dir):
    """단계 6: PPT 17·18쪽의 segmentation loss와 mask mAP 그래프."""
    df = pd.read_csv(Path(results_dir) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for c in ["train/seg_loss", "val/seg_loss"]:
        if c in df.columns:
            ax[0].plot(df["epoch"], df[c], label=c)
    ax[0].set_title("Segmentation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    for c in ["metrics/mAP50(M)", "metrics/mAP50-95(M)"]:
        if c in df.columns:
            ax[1].plot(df["epoch"], df[c], label=c)
    ax[1].set_title("Mask mAP")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("mAP")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "curve.png", dpi=150)
    plt.show()


def run_week1():
    """1주차 실행 순서."""
    setup_environment()
    imgs = check_and_acquire_data()
    observe_images(imgs)


def run_week2():
    """2주차 실행 순서. 직접 주석은 CVAT 또는 Roboflow에서 수행한다."""
    setup_environment()
    check_polygon_annotation_results()
    check_label_format()
    images = check_data_integrity()
    split_dataset(images)
    results_dir = train_model()
    plot_training_curves(results_dir)


if __name__ == "__main__":
    run_week2()
