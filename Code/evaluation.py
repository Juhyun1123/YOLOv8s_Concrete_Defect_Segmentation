"""3주차 세그멘테이션·ON/OFF 평가와 오류 사례 저장."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics.data.utils import img2label_paths

from .training import CLASS_NAMES, PROJECT_ROOT


IMGSZ = 960
THRS = (0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3)
PRED_CONF = 0.0009  # 탐색 범위의 최솟값보다 낮게 추론해 0.001 후보도 평가한다.


def load_eval_split(data_yaml, split):
    """영상과 대응 라벨에서 영상별 정답 클래스 목록을 읽는다."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent / config.get("path", ".")
    paths = sorted((root / config[split]).rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"{split} 이미지가 없습니다: {root / config[split]}")
    labels = [Path(p) for p in img2label_paths([str(p) for p in paths])]
    gt_classes = [
        {int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()}
        for label in labels
    ]
    return paths, labels, gt_classes


def evaluate_segmentation(model, data_yaml, output_dir):
    """단계 1. 테스트셋 mask mAP와 클래스별 AP를 산출한다."""
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "week3_eval"),
        name=output_dir.name,
        exist_ok=True,
    )
    mask = metrics.seg
    rows = [{"class": "all", "mask_ap50": mask.map50, "mask_ap50_95": mask.map}]
    per_class = {
        int(cid): (float(ap50), float(ap))
        for cid, ap50, ap in zip(mask.ap_class_index, mask.ap50, mask.ap, strict=True)
    }
    for cid, name in CLASS_NAMES.items():
        ap50, ap = per_class.get(cid, (None, None))
        rows.append({"class": name, "mask_ap50": ap50, "mask_ap50_95": ap})
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(output_dir / "segmentation.csv", index=False, float_format="%.6f")
    return float(mask.map50), float(mask.map)


def predict_split(model, paths):
    """각 영상을 한 번 추론하고, 클래스별 최대 confidence를 보관한다."""
    scores = np.zeros((len(paths), len(CLASS_NAMES)), dtype=np.float32)
    for i, path in enumerate(paths):
        result = model.predict(str(path), imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
        cid = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        np.maximum.at(scores[i], cid, conf)
    return scores


def onoff(scores, gts, threshold, cid_target):
    """단계 2. 클래스별 영상 단위 TP·FP·FN·TN과 평가 지표를 계산한다."""
    gt_on = np.array([cid_target in gt for gt in gts], dtype=bool)
    pred_on = scores[:, cid_target] >= threshold
    tp = int((gt_on & pred_on).sum())
    fp = int((~gt_on & pred_on).sum())
    fn = int((gt_on & ~pred_on).sum())
    tn = int((~gt_on & ~pred_on).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": (tp + tn) / len(gts),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_thresholds(val_scores, val_gts, output_dir):
    """단계 3. 검증 F1을 최대화하고 동점이면 낮은 임계값을 고른다."""
    best = {}
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for cid, name in CLASS_NAMES.items():
        candidates = [
            {"class": name, "threshold": t, **onoff(val_scores, val_gts, t, cid)}
            for t in THRS
        ]
        chosen = max(candidates, key=lambda row: row["f1"])
        best[name] = chosen["threshold"]
        for row in candidates:
            row["selected"] = row is chosen
        rows.extend(candidates)
        ax.semilogx(THRS, [row["f1"] for row in candidates], marker="o", label=name)

    pd.DataFrame(rows).to_csv(output_dir / "val_thresholds.csv", index=False, float_format="%.6f")
    ax.set(xlabel="Confidence threshold", ylabel="F1-score", title="Validation F1 by Threshold", ylim=(0, 1.03))
    ax.set_xticks(THRS, [f"{t:g}" for t in THRS])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "val_threshold_f1.png", dpi=150)
    plt.close(fig)
    return best


def evaluate_test_onoff(test_scores, test_gts, best, output_dir):
    """단계 4. 검증셋에서 선택한 값을 테스트셋에 고정 적용한다."""
    df = pd.DataFrame([
        {"class": name, "threshold": best[name], **onoff(test_scores, test_gts, best[name], cid)}
        for cid, name in CLASS_NAMES.items()
    ])
    df.to_csv(output_dir / "test_onoff.csv", index=False, float_format="%.6f")
    print(df.round(4).to_string(index=False))
    print("Macro F1:", round(df["f1"].mean(), 4))
    return df


def save_confusion_matrix(test_table, output_dir):
    """영상별 다중 라벨 평가에 맞춰 클래스마다 2×2 행렬을 저장한다."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(9, 3.5), layout="constrained")
    vmax = int(test_table[["TP", "FP", "FN", "TN"]].to_numpy().max())
    for ax, row in zip(axes, test_table.itertuples(index=False), strict=True):
        counts = np.array([[row.TN, row.FP], [row.FN, row.TP]])
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=vmax)
        ax.set(xticks=(0, 1), yticks=(0, 1), xlabel="Predicted", ylabel="Actual")
        ax.set_xticklabels(["OFF", "ON"])
        ax.set_yticklabels(["OFF", "ON"])
        ax.set_title(f"{row[0]} (conf ≥ {row.threshold:g})")
        for i in range(2):
            for j in range(2):
                color = "white" if counts[i, j] > vmax / 2 else "#10213a"
                ax.text(j, i, counts[i, j], ha="center", va="center", color=color)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_error_cases(model, paths, labels, gts, scores, best, output_dir):
    """단계 5. 클래스별 FP·FN 대표 영상 최대 3장씩 저장한다."""
    for cid, name in CLASS_NAMES.items():
        class_dir = output_dir / "error_cases" / name
        class_dir.mkdir(parents=True, exist_ok=True)
        previous = set(class_dir.glob("FP_*.jpg")) | set(class_dir.glob("FN_*.jpg"))
        gt_on = np.array([cid in gt for gt in gts])
        pred_on = scores[:, cid] >= best[name]
        cases = {"FP": pred_on & ~gt_on, "FN": ~pred_on & gt_on}
        for kind, selected in cases.items():
            for i, index in enumerate(np.flatnonzero(selected)[:3], start=1):
                path = paths[index]
                if kind == "FP":
                    result = model.predict(str(path), imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
                    boxes = result.boxes
                    keep = (boxes.cls == cid) & (boxes.conf >= best[name])
                    image = result[keep].plot()
                else:
                    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    h, w = image.shape[:2]
                    for line in labels[index].read_text(encoding="utf-8").splitlines():
                        values = line.split()
                        if values and int(values[0]) == cid:
                            xy = np.array(values[1:], dtype=float).reshape(-1, 2)
                            points = np.rint(xy * [w - 1, h - 1]).astype(np.int32)
                            cv2.polylines(image, [points], True, (0, 210, 255), 3)
                out_path = class_dir / f"{kind}_{i}_{path.name}"
                ok, encoded = cv2.imencode(".jpg", image)
                if not ok:
                    raise OSError(f"오류 사례를 저장할 수 없습니다: {out_path}")
                encoded.tofile(out_path)
                previous.discard(out_path)
        for stale in previous:
            stale.unlink()


def evaluate_model(model, data_yaml, output_dir):
    if model.names != CLASS_NAMES:
        raise ValueError(f"모델의 클래스 순서가 다릅니다: {model.names}")
    output_dir.mkdir(parents=True, exist_ok=True)

    val_paths, _, val_gts = load_eval_split(data_yaml, "val")
    best = find_thresholds(predict_split(model, val_paths), val_gts, output_dir)

    test_paths, test_labels, test_gts = load_eval_split(data_yaml, "test")
    test_scores = predict_split(model, test_paths)
    map50, map50_95 = evaluate_segmentation(model, data_yaml, output_dir)
    test_table = evaluate_test_onoff(test_scores, test_gts, best, output_dir)
    save_confusion_matrix(test_table, output_dir)
    save_error_cases(model, test_paths, test_labels, test_gts, test_scores, best, output_dir)
    return {
        "mask_mAP50": map50,
        "mask_mAP50_95": map50_95,
        "accuracy": test_table["accuracy"].mean(),
        "macro_precision": test_table["precision"].mean(),
        "macro_recall": test_table["recall"].mean(),
        "macro_f1": test_table["f1"].mean(),
    }
