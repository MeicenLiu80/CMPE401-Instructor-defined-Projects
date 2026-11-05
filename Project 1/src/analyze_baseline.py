
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\CMPE401_Project1")
csv_path = ROOT / "runs" / "yolo11n_baseline" / "results.csv"

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

print("Available columns:")
for column in df.columns:
    print(column)

print("\nBest epochs by validation metrics:")

for metric in [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]:
    if metric not in df.columns:
        print(f"\nColumn not found: {metric}")
        continue

    idx = df[metric].idxmax()
    print(f"\n{metric}")
    print("Epoch:", int(idx) + 1)
    print("Best value:", round(float(df.loc[idx, metric]), 6))

print("\nLast 5 epochs:")
print(df.tail(5).to_string(index=False))
