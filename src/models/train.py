import argparse, yaml, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    df = pd.read_parquet(cfg["features_path"])
    X = df.drop(columns=[cfg["target_label"], "student_pid"])
    y = df[cfg["target_label"]].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["train"]["test_size"],
        stratify=y, random_state=cfg["train"]["random_state"]
    )
    steps = []
    if cfg["train"]["use_smote"]:
        steps.append(("smote", SMOTE()))
    steps.append(("lr", LogisticRegression(max_iter=1000, class_weight="balanced")))
    pipe = Pipeline(steps=steps)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    ap = average_precision_score(yte, proba)
    brier = brier_score_loss(yte, proba)
    print("AUC-PR:", ap, "Brier:", brier)

    out = Path(cfg["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "baseline.joblib")
    print("modelo salvo em", out / "baseline.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
