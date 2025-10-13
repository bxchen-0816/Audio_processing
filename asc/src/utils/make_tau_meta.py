# asc/src/utils/make_tau_meta.py
import pandas as pd, pathlib, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="数据集根目录，里面要有 audio/ 和 meta.csv")
ap.add_argument("--audio_dir", default="audio", help="音频子目录名（默认 audio）")
ap.add_argument("--val_fold", type=int, default=9)
ap.add_argument("--test_fold", type=int, default=10)
args = ap.parse_args()

root = pathlib.Path(args.root)
meta_path = root / "meta.csv"

def choose(col_names, candidates):
    for c in candidates:
        if c in col_names: return c
    return None

if meta_path.exists():
    meta = pd.read_csv(meta_path, sep=None, engine="python")
    # 找文件名列和标签列（兼容不同数据集）
    fn_col  = choose(meta.columns, ["filename", "file"])
    lab_col = choose(meta.columns, ["scene_label", "label", "category", "class"])
    if fn_col is None or lab_col is None:
        raise RuntimeError(f"meta.csv 不含所需列，看到的列是：{list(meta.columns)}")

    # 使用官方 fold → train/val/test；若没有 fold 就 8/1/1 随机
    if "fold" in meta.columns:
        def fold_to_split(f):
            try: f = int(f)
            except: return "train"
            if f <= 8: return "train"
            if f == args.val_fold: return "val"
            if f == args.test_fold: return "test"
            return "train"
        meta["split"] = meta["fold"].apply(fold_to_split)
    else:
        meta = meta.sample(frac=1, random_state=42).reset_index(drop=True)
        n=len(meta); tr=int(0.8*n); va=int(0.9*n)
        splits = ["train"]*(tr+1) + ["val"]*(va-tr) + ["test"]*(n-va-1)
        meta["split"] = splits[:len(meta)]

    meta["path"] = meta[fn_col].apply(lambda x: str(root / args.audio_dir / x))
    df = meta[["path", lab_col, "split"]].rename(columns={lab_col: "label"})
else:
    # 兜底：没有 meta.csv 就扫描目录
    paths = list((root / args.audio_dir).rglob("*.wav"))
    if not paths:
        raise RuntimeError(f"未在 {root/args.audio_dir} 里找到 wav。")
    df = pd.DataFrame({
        "path": [str(p) for p in paths],
        "label": [p.parent.name for p in paths]
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    n=len(df); tr=int(0.8*n); va=int(0.9*n)
    df.loc[:tr,  "split"] = "train"
    df.loc[tr:va,"split"] = "val"
    df.loc[va:,  "split"] = "test"

# 将输出写到项目 data/ 目录（当前工作目录是 F:\asc）
out = pathlib.Path("data/meta.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("Saved:", out)
try:
    print(df.groupby(["split","label"]).size().head())
except Exception:
    pass
