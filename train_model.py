"""Train the models from data/Support_tickets.csv and save them with Joblib.   Usage: python train_model.py [--data path.csv]"""
import argparse
import pandas as pd
from support_ticket_classifier import DATA_PATH, MODEL_PATH, save_model, train_all
from utils.preprocessing import prepare_dataset


def run(data_path=DATA_PATH) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Add your CSV there or run generate_sample_data.py.")
    raw = pd.read_csv(data_path)
    df = prepare_dataset(raw)
    print(f"Rows: {len(raw)} raw -> {len(df)} after removing duplicates/empty text")
    if "priority" in df:
        print("Priority distribution:\n", df["priority"].value_counts(normalize=True).round(3).to_string())
    bundle = train_all(df)
    save_model(bundle)
    for name in ["priority", "category"]:
        if bundle[name]:
            m = bundle[name]["metrics"]
            print(f"{name}: accuracy={m['accuracy']:.3f} weighted F1={m['f1']:.3f} macro F1={m['macro_f1']:.3f}")
        else:
            print(f"{name}: not trained - {bundle['category_note']}")
    print(f"Saved model to {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    from pathlib import Path
    run(Path(ap.parse_args().data))
