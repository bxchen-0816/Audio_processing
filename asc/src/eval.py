import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import load_config
from src.utils.audio import set_feature_cfg
from src.datasets.asc import ASCDataset
from src.models.cnn_small import SmallCNN   # 若你用的是别的模型，这里换成你的模型类

def build_dataset(cfg, label2id):
    """尽量兼容不同版本的 ASCDataset 构造函数"""
    try:
        # 你的版本可能支持 label2id/target_frames
        return ASCDataset(cfg["data_csv"], "test",
                          label2id=label2id,
                          target_frames=cfg.get("target_frames", None))
    except TypeError:
        # 回退到最简单的签名
        return ASCDataset(cfg["data_csv"], split="test")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',  default='configs/tau2020m.yaml', help='配置文件（与训练一致）')
    ap.add_argument('--ckpt',    default='runs/tau2020m_baseline/best.pt', help='最优权重路径')
    ap.add_argument('--workers', type=int, default=0, help='DataLoader 的 num_workers，Windows 默认为 0')
    ap.add_argument('--batch',   type=int, default=None, help='覆盖 cfg 的 batch_size（可选）')
    args = ap.parse_args()

    # 读取权重与配置
    ck  = torch.load(args.ckpt, map_location='cpu')
    cfg = ck.get("cfg", load_config(args.config))
    label2id = ck["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    # 设置与训练一致的特征参数
    set_feature_cfg(sr=cfg["sr"], n_fft=cfg["n_fft"], hop=cfg["hop"],
                    n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"])

    # 数据集与 DataLoader
    ds = build_dataset(cfg, label2id)
    bs = args.batch or cfg.get("batch_size", 32)
    dl = DataLoader(ds, batch_size=bs, shuffle=False,
                    num_workers=args.workers, pin_memory=False)

    # 模型
    model = SmallCNN(len(label2id))
    model.load_state_dict(ck["model"])
    model.eval()

    # 推理
    gts, preds = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            preds += logits.argmax(1).cpu().tolist()
            gts   += y.tolist()

    # 分类报告
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(gts, preds, target_names=target_names, digits=4)
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(gts, preds, labels=list(range(len(id2label))))
    os.makedirs("runs", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    cm_path = os.path.join("runs", "confmat.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # 保存报告到文本（方便写报告用）
    with open(os.path.join("runs", "eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
        f.write(f"ckpt: {args.ckpt}\n")

if __name__ == '__main__':
    # Windows 下使用多进程 DataLoader 必须加守护，哪怕 workers=0 也安全
    # import multiprocessing as mp; mp.freeze_support()
    main()
