# src/predict.py
import argparse, torch, numpy as np
from src.models.cnn_small import SmallCNN
from src.utils.audio import set_feature_cfg, load_wav, logmel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="要分类的音频文件路径")
    ap.add_argument("--ckpt", default="asc/runs/tau2020m_baseline/best.pt")  # 如你已搬到 runs/ 则改成 runs/...
    ap.add_argument("--config", default="configs/tau2020m.yaml")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg, label2id = ck["cfg"], ck["label2id"]
    id2label = {v:k for k,v in label2id.items()}

    # 与训练一致的特征设置
    set_feature_cfg(sr=cfg["sr"], n_fft=cfg["n_fft"], hop=cfg["hop"],
                    n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"])

    # 读音频 → log-Mel
    y, sr = load_wav(args.wav, cfg["sr"])
    M = logmel(y, sr)  # [n_mels, T]

    # 对齐到训练长度（中心裁剪/右侧补齐）
    L, T = cfg.get("target_frames", M.shape[1]), M.shape[1]
    if T >= L:
        s = (T - L)//2; M = M[:, s:s+L]
    else:
        M = np.pad(M, ((0,0),(0,L-T)), mode="constant", constant_values=M.min())

    x = torch.from_numpy(M).unsqueeze(0).unsqueeze(0).float()  # [1,1,n_mels,L]

    # 模型
    model = SmallCNN(len(label2id))
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        pred = model(x).argmax(1).item()
    print("Predicted:", id2label[pred])

if __name__ == "__main__":
    main()
