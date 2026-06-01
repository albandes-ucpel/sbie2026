import argparse, yaml, pandas as pd, joblib

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    df = pd.read_parquet(cfg["features_path"])
    model = joblib.load("models/baseline.joblib")
    proba = model.predict_proba(df.drop(columns=[cfg["target_label"], "student_pid"]))[:,1]
    df["risk_score"] = proba
    print(df.sort_values("risk_score", ascending=False)
            [["student_pid","class_id","risk_score"]]
            .head(cfg["scoring"]["k"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
