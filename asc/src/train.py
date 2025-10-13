import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.audio import set_feature_cfg
from src.datasets.asc import ASCDataset
from src.models.cnn_small import SmallCNN
from sklearn.metrics import accuracy_score, f1_score

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main(cfg_path):
    cfg = load_config(cfg_path)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg.get("seed",42))
    set_feature_cfg(sr=cfg["sr"], n_fft=cfg["n_fft"], hop=cfg["hop"],
                    n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"])

    train_set = ASCDataset(cfg["data_csv"], "train", target_frames=cfg["target_frames"], augment=True)
    val_set   = ASCDataset(cfg["data_csv"], "val",   label2id=train_set.label2id,
                           target_frames=cfg["target_frames"], augment=False)
    n_classes = len(train_set.label2id)
    dl_tr = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
    dl_va = DataLoader(val_set,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, cfg["epochs"]+1):
        model.train(); loss_sum=0
        for x,y in dl_tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); logits=model(x); loss=crit(logits,y); loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0)

        model.eval(); preds=[]; gts=[]
        with torch.no_grad():
            for x,y in dl_va:
                x = x.to(device); logits = model(x)
                preds += logits.argmax(1).cpu().tolist(); gts += y.tolist()
        acc = accuracy_score(gts, preds); f1 = f1_score(gts, preds, average="macro")
        print(f"Epoch {ep:02d} | train_loss {loss_sum/len(train_set):.4f} | val_acc {acc:.4f} | val_f1 {f1:.4f}")

        if acc>best:
            best=acc
            torch.save({"model":model.state_dict(),"label2id":train_set.label2id,"cfg":cfg},
                       os.path.join(cfg["out_dir"], "best.pt"))
    print("Best val_acc:", best)

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tau2020m.yaml")
    args = ap.parse_args(); main(args.config)
