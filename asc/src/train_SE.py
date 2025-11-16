import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.utils.config import load_config
from src.utils.audio import set_feature_cfg
from src.datasets.asc import ASCDataset
from src.models.cnn_small import SmallCNN


def resolve_from_root(p: str) -> Path:
    """
    把相对路径变成相对于项目根目录（src 的上一层）的绝对路径。
    例如 'configs/tau2020m.yaml' -> F:/asc/configs/tau2020m.yaml
    """
    p = Path(p)
    if p.is_absolute():
        return p
    proj_root = Path(__file__).resolve().parents[1]  # 比如 F:\asc
    return proj_root / p


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main(cfg_path: str):
    # 1) 先把 config 路径解析成绝对路径
    cfg_path = resolve_from_root(cfg_path)
    cfg = load_config(str(cfg_path))

    # 2) 再把 yaml 里写的 out_dir / data_csv 也解析一下（防止相对路径出问题）
    if "out_dir" in cfg:
        cfg["out_dir"] = str(resolve_from_root(cfg["out_dir"]))
    if "data_csv" in cfg:
        cfg["data_csv"] = str(resolve_from_root(cfg["data_csv"]))

    # 3) 训练前的通用准备
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg.get("seed", 42))

    # 与实验一致的 STFT / Mel 参数
    set_feature_cfg(
        sr=cfg["sr"],
        n_fft=cfg["n_fft"],
        hop=cfg["hop"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
    )

    # 4) 构建数据集与 DataLoader
    train_set = ASCDataset(
        cfg["data_csv"],
        "train",
        target_frames=cfg["target_frames"],
        augment=True,
    )
    val_set = ASCDataset(
        cfg["data_csv"],
        "val",
        label2id=train_set.label2id,
        target_frames=cfg["target_frames"],
        augment=False,
    )
    n_classes = len(train_set.label2id)

    # ★★★ 新增：构建 id2label，供打印使用 ★★★
    label2id = train_set.label2id
    id2label = {v: k for k, v in label2id.items()}

    dl_tr = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    dl_va = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # 5) 设备 & 模型（带 SE 开关）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("train device =", device)
    # 从配置中读取 SE 开关，默认 True；如果你想开关，可以在 yaml 里写 model: {use_se: false}
    use_se = cfg.get("model", {}).get("use_se", True)

    model = SmallCNN(n_classes, use_se=use_se).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, cfg["epochs"] + 1):
        # ---------- 训练 ----------
        model.train()
        loss_sum = 0.0
        num_batches = len(dl_tr)

        for step, (x, y) in enumerate(dl_tr, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)

            # 每 50 个 batch 打印一次当前 loss 和进度（你可以改成 20、100）
            if step % 50 == 0 or step == num_batches:
                print(f"[train] ep {ep:02d} step {step:03d}/{num_batches} "
                      f"batch_loss={loss.item():.4f}")

        # ---------- 验证 ----------
        model.eval()
        preds, gts = [], []
        shown = 0  # 打印前几条验证样本的预测情况

        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(1).cpu()
                preds += pred.tolist()
                gts += y.tolist()

                # 只在前几个 batch 上打印几条示例（真实标签 vs 预测）
                if shown < 5:
                    for i in range(min(len(y), 5 - shown)):
                        true_lbl = id2label[y[i].item()]
                        pred_lbl = id2label[pred[i].item()]
                        print(f"[val sample] true={true_lbl:>13} | pred={pred_lbl:>13}")
                        shown += 1

        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="macro")
        avg_loss = loss_sum / max(len(train_set), 1)

        print(f"Epoch {ep:02d} | train_loss {avg_loss:.4f} "
              f"| val_acc {acc:.4f} | val_f1 {f1:.4f}")

        # 保存最优模型
        if acc > best:
            best = acc
            torch.save(
                {"model": model.state_dict(),
                 "label2id": train_set.label2id,
                 "cfg": cfg},
                os.path.join(cfg["out_dir"], "best.pt")
            )
            print(f"[INFO] New best model saved with val_acc={best:.4f}")

 


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tau2020m.yaml")
    args = ap.parse_args()
    main(args.config)