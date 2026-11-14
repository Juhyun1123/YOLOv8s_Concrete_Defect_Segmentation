"""3주차 모델 비교 및 결과 저장."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from ultralytics import YOLO

from Code.evaluation import evaluate_model
from Code.training import OUT_ROOT, PROJECT_ROOT, check_data_integrity, split_dataset, train_model


MODELS = ("yolov8s-seg", "yolov8n-seg")
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_weights(model_name):
    weights = MODELS_DIR / model_name / "best.pt"
    if not weights.is_file():
        run_dir = train_model(f"{model_name}.pt", f"week3_{model_name}")
        weights.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=weights.parent) as temporary:
            staged = Path(temporary) / weights.name
            shutil.copy2(run_dir / "weights" / "best.pt", staged)
            staged.replace(weights)
    return weights


def main():
    parser = argparse.ArgumentParser(description="3주차 결함 분할 및 영상 단위 ON/OFF 평가")
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    args = parser.parse_args()
    names = MODELS if args.model == "all" else (args.model,)

    split_dataset(check_data_integrity())
    rows = []
    for name in names:
        model = YOLO(str(get_weights(name)))
        row = evaluate_model(model, OUT_ROOT / "data.yaml", RESULTS_DIR / name)
        rows.append({"model": name, **row})

    comparison = pd.DataFrame(rows)
    print(comparison.round(4).to_string(index=False))
    if len(rows) == len(MODELS):
        comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
