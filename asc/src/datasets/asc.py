import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from src.utils.audio import cache_feature   


class ASCDataset(Dataset):
    def __init__(self, csv_path, split, label2id=None, target_frames=1000, augment=False):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"]==split].reset_index(drop=True)
        if label2id is None:
            labels = sorted(df["label"].unique())
            self.label2id = {l:i for i,l in enumerate(labels)}
        else:
            self.label2id = label2id
        self.id2label = {v:k for k,v in self.label2id.items()}
        self.target_frames = target_frames
        self.augment = augment

    def __len__(self): return len(self.df)

    def _crop_or_pad(self, m):
        T, L = m.shape[1], self.target_frames
        if T > L:
            start = np.random.randint(0, T-L+1) if self.augment else (T-L)//2
            m = m[:, start:start+L]
        elif T < L:
            pad = L - T
            m = np.pad(m, ((0,0),(0,pad)), mode="constant", constant_values=m.min())
        return m

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = cache_feature(row["path"])     # [n_mels, T]
        feat = self._crop_or_pad(feat)
        x = torch.from_numpy(feat).unsqueeze(0).float()  # [1, n_mels, T]
        y = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return x, y
