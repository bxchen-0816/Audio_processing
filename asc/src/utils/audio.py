import librosa, numpy as np, soundfile as sf, os, hashlib

CFG = dict(sr=32000, n_fft=1024, hop=320, n_mels=64, fmin=50, fmax=None)

def set_feature_cfg(sr=None, n_fft=None, hop=None, n_mels=None, fmin=None, fmax=None):
    if sr is not None:   CFG["sr"]=sr
    if n_fft is not None:CFG["n_fft"]=n_fft
    if hop is not None:  CFG["hop"]=hop
    if n_mels is not None: CFG["n_mels"]=n_mels
    if fmin is not None: CFG["fmin"]=fmin
    if fmax is not None: CFG["fmax"]=fmax

def load_wav(path, target_sr=None):
    if target_sr is None: target_sr = CFG["sr"]
    y, sr = sf.read(path, dtype="float32")
    if y.ndim>1: y=y.mean(1)
    if sr!=target_sr: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def logmel(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CFG["n_fft"], hop_length=CFG["hop"],
        n_mels=CFG["n_mels"], fmin=CFG["fmin"], fmax=CFG["fmax"] or sr//2, power=2.0)
    return librosa.power_to_db(S).astype(np.float32)  # [n_mels, T]

def cache_feature(wav_path, cache_root="asc/data/.cache"):
    os.makedirs(cache_root, exist_ok=True)
    key = hashlib.md5((wav_path+str(CFG)).encode()).hexdigest()+".npy"
    cache_path = os.path.join(cache_root, key)
    if os.path.exists(cache_path):
        return np.load(cache_path)
    y, sr = load_wav(wav_path)
    feat = logmel(y, sr)
    np.save(cache_path, feat)
    return feat