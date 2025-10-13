# asc/src/predict_folder.py
import argparse, os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.audio import set_feature_cfg         # 复用你工程里的音频配置
from models.cnn_small import SmallCNN           # 你的模型
# 如果你用的不是 SmallCNN，请改成实际模型类

def build_id2label(label2id: dict):
    return {v: k for k, v in label2id.items()}

def melspec_from_file(path, cfg):
    # 读 wav
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim > 1:                      # 转单声道
        y = y.mean(axis=1)
    if sr != cfg["sr"]:                 # 重采样
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg["sr"])
    # 计算 mel
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg["sr"],
        n_fft=cfg["n_fft"], hop_length=cfg["hop"],
        n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"]
    )
    logS = librosa.power_to_db(S, ref=1.0)     # [n_mels, T]
    return logS

class FolderDS(Dataset):
    def __init__(self, root, exts, cfg):
        self.files = []
        for ext in exts:
            self.files += list(Path(root).rglob(f"*{ext}"))
        self.files = sorted(map(str, self.files))
        self.cfg = cfg
        self.target_frames = cfg["target_frames"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        feat = melspec_from_file(path, self.cfg)            # [n_mels, T]
        x = torch.tensor(feat).unsqueeze(0)                 # [1, n_mels, T]
        # pad/裁剪到训练时的帧长
        T = x.shape[-1]
        if T < self.target_frames:
            x = F.pad(x, (0, self.target_frames - T))
        else:
            x = x[..., :self.target_frames]
        return x.float(), path

def main():
    import argparse, os
    from pathlib import Path

    ap = argparse.ArgumentParser()
    # 让 --folder 变成可选；如果没给，就弹框选择
    ap.add_argument("--folder", default=None, help="要预测的音频文件夹（递归扫描）")
    ap.add_argument("--ckpt",   default=r".\runs\tau2020m_baseline\best.pt", help="模型权重 ckpt")
    ap.add_argument("--config", default=r".\configs\tau2020m.yaml",
                    help="仅用于定位工程；特征参数将从 ckpt['cfg'] 读取")
    ap.add_argument("--ext", nargs="+", default=[".wav"], help="音频后缀，例：--ext .wav .flac")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)  # Windows 建议 0
    ap.add_argument("--device", default="cpu")         # 或 cuda
    ap.add_argument("--out", default="folder_predictions.csv")
    args = ap.parse_args()

    # ====== 新增：如果未提供 --folder，弹出选择对话框 ======
    if not args.folder:
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk_root = tk.Tk()
            tk_root.withdraw()  # 不显示主窗口
            sel = filedialog.askdirectory(title="请选择要预测的音频文件夹")
            tk_root.update()
            tk_root.destroy()
            if sel:
                args.folder = sel
        except Exception:
            # 如果 tkinter 不可用，回退到命令行输入
            pass

    # 再次确认；如果仍然没有，使用命令行输入作为兜底
    while not args.folder:
        inp = input("请输入要预测的音频文件夹路径（直接回车退出）：").strip()
        if not inp:
            print("未选择文件夹，已退出。")
            return
        if os.path.isdir(inp):
            args.folder = inp
        else:
            print("路径不存在或不是文件夹，请重新输入。")

    # 规范化为绝对路径（可选）
    args.folder = str(Path(args.folder).resolve())
    print(f"将对该文件夹进行预测：{args.folder}")

    # ====== 下面保持你原来的逻辑（加载 ckpt、构建数据集、模型推理、保存 CSV）======

    # ========= 载入权重与配置（更健壮） =========
    from pathlib import Path
    import os
    proj_root = Path(__file__).resolve().parents[1]  # 项目根，比如 F:\asc

    # 1) 先把传入的 ckpt 解析为“相对于项目根”的绝对路径
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (proj_root / ckpt_path).resolve()

    # 2) 如果找不到，分别在 runs/ 和 asc/runs/ 下搜 best.pt
    if not ckpt_path.exists():
        candidates = []
        for sub in ["runs", "asc/runs"]:
            candidates += list((proj_root / sub).rglob("best.pt"))
        if candidates:
            ckpt_path = candidates[0]
        else:
            # 3) 仍找不到就弹出文件选择框
            try:
                import tkinter as tk
                from tkinter import filedialog
                tk_root = tk.Tk(); tk_root.withdraw()
                sel = filedialog.askopenfilename(
                    title="请选择模型权重文件（.pt/.pth）",
                    filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All Files", "*.*")]
                )
                tk_root.destroy()
                if not sel:
                    print("未选择权重，已退出。"); return
                ckpt_path = Path(sel)
            except Exception as e:
                print("无法选择权重：", e); return

    print(f"使用权重: {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location="cpu")   # 只加载这一次！
    cfg = ck["cfg"]
    label2id = ck["label2id"]
    id2label = build_id2label(label2id)


    # 让工程里的特征配置生效（保持和训练一致）
    set_feature_cfg(sr=cfg["sr"], n_fft=cfg["n_fft"], hop=cfg["hop"],
                    n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"])

    # 数据集
    ds = FolderDS(args.folder, args.ext, cfg)
    if len(ds) == 0:
        print("未在文件夹中找到指定后缀的音频：", args.ext)
        return
    print(f"将从 {args.folder} 读取 {len(ds)} 个文件，后缀：{args.ext}")
    if len(ds) == 0:
        print("未在文件夹中找到指定后缀的音频：", args.ext)
        return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # 模型
    num_classes = len(label2id)
    model = SmallCNN(num_classes)
    model.load_state_dict(ck["model"])
    model.eval().to(args.device)

    # 推理
    results = []
    with torch.no_grad():
        for x, paths in dl:
            x = x.to(args.device)
            logits = model(x)                        # [B, C]
            prob = torch.softmax(logits, dim=1)     # [B, C]
            conf, pred = prob.max(dim=1)            # [B]
            for pth, yi, cf in zip(paths, pred.cpu().tolist(), conf.cpu().tolist()):
                results.append((pth, id2label[yi], float(cf)))

    # 保存 CSV
    import csv
    out_csv = args.out
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "pred_label", "confidence"])
        w.writerows(results)
    print(f"已保存预测结果：{out_csv}（共 {len(results)} 条）")

    # 同时打印前几条
    print("\n示例：")
    for r in results[:5]:
        print(r)

if __name__ == "__main__":
    main()
