import pandas as pd
import numpy as np

SIGNALS = ['Vpv', 'Vdc', 'Ipv', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']
STATS = ['mean', 'std', 'max', 'min']

def get_features(filepath):
    df = pd.read_csv(filepath)
    # Take the first 20 rows (one window)
    window = df.iloc[:20]
    feats = {}
    for col in SIGNALS:
        s = window[col]
        feats[f"{col}_mean"] = s.mean()
        feats[f"{col}_std"] = s.std(ddof=0)
        feats[f"{col}_max"] = s.max()
        feats[f"{col}_min"] = s.min()
    return feats

f0 = get_features('dataset/F0L.csv')
f1 = get_features('dataset/F1L.csv') # Line-Line
f5 = get_features('dataset/F5L.csv') # Arc Parallel
f7 = get_features('dataset/F7L.csv') # Short Circuit

print(f"{'Feature':<20} | {'F0 (Health)':<12} | {'F1 (L-L)':<12} | {'Diff Implies'}")
print("-" * 60)

for key in f0:
    v0 = f0[key]
    v1 = f1[key]
    # Highlight significant differences (>10%)
    if abs(v1 - v0) > (abs(v0) * 0.1 + 0.01):
        print(f"{key:<20} | {v0:<12.2f} | {v1:<12.2f} | {'<< DIFF'}")
        
print("\n" + "="*60 + "\n")

print(f"{'Feature':<20} | {'F0 (Health)':<12} | {'F7 (Short)':<12} | {'Diff Implies'}")
print("-" * 60)

for key in f0:
    v0 = f0[key]
    v7 = f7[key]
    if abs(v7 - v0) > (abs(v0) * 0.1 + 0.01):
        print(f"{key:<20} | {v0:<12.2f} | {v7:<12.2f} | {'<< DIFF'}")

print("\n" + "="*60 + "\n")

print(f"{'Feature':<20} | {'F0 (Health)':<12} | {'F5 (Arc)':<12} | {'Diff Implies'}")
print("-" * 60)

print("\n" + "="*60 + "\n")

f2 = get_features('dataset/F2L.csv') # Line-Ground

print(f"{'Feature':<20} | {'F0 (Health)':<12} | {'F2 (L-G)':<12} | {'Diff Implies'}")
print("-" * 60)

f3 = get_features('dataset/F3L.csv') # Line-Line-Ground

print("\n" + "="*60 + "\n")
print(f"{'Feature':<20} | {'F1 (L-L)':<12} | {'F7 (Short)':<12} | {'Diff Implies'}")
print("-" * 60)

for key in f0:
    v1 = f1[key]
    v7 = f7[key]
    if abs(v7 - v1) > (abs(v1) * 0.1 + 0.01):
        print(f"{key:<20} | {v1:<12.2f} | {v7:<12.2f} | {'<< DIFF'}")
        
print("\n" + "="*60 + "\n")
print(f"{'Feature':<20} | {'F7 (Short)':<12} | {'F3 (LLG)':<12} | {'Diff Implies'}")
print("-" * 60)

for key in f0:
    v7 = f7[key]
    v3 = f3[key]
    if abs(v3 - v7) > (abs(v7) * 0.1 + 0.01):
        print(f"{key:<20} | {v7:<12.2f} | {v3:<12.2f} | {'<< DIFF'}")
