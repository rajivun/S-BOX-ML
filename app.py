"""
Modifikasi AES S-box Berbasis Machine Learning & Demo AES-128 CBC
------------------------------------------------------------------
Ruang pencarian : Eksponen power-map d  x  Matriks Afin Non-Sirkulan GL(8,2)
                  + Optimasi Eksak Konstanta C in [0..255]
Konstruksi      : S(x) = A . x^d  XOR  C    (di GF(2^8), polinomial AES 0x11B)
ML              : surrogate (Ridge / RandomForest / HistGradientBoosting) memprediksi
                  loss struktural kandidat (d, A) tanpa menghitung semua metrik secara penuh.
Jalankan        : streamlit run app.py
"""
import io
import json
import math
import os
import struct
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AES S-box + ML", page_icon="🔐", layout="wide")

# =====================================================================
# 1. ARITMATIKA GF(2^8) DAN KOMPONEN DASAR
# =====================================================================
AES_POLY = 0x11B


def gmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= AES_POLY
        b >>= 1
    return r


EXP = np.zeros(512, dtype=np.int64)
LOG = np.zeros(256, dtype=np.int64)
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x = gmul(_x, 3)  # 3 = generator primitif untuk polinomial 0x11B
EXP[255:510] = EXP[:255]

PAR = np.array([bin(i).count("1") & 1 for i in range(256)], dtype=np.uint8)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
H256 = (1.0 - 2.0 * PAR[np.arange(256)[:, None] & np.arange(256)[None, :]]).astype(np.float32)
XS = np.arange(256, dtype=np.uint8)

# topeng keluaran: 8 bit tunggal (SAC) + 28 pasangan bit (BIC-SAC)
_PAIRS = [(j, k) for j in range(8) for k in range(j + 1, 8)]
MASKS = np.array([1 << j for j in range(8)] + [(1 << j) | (1 << k) for j, k in _PAIRS], dtype=np.uint8)
PAIR_MASKS = MASKS[8:]
IDX0 = [np.array([x for x in range(256) if not (x >> i) & 1], dtype=np.intp) for i in range(8)]


def rotl8(v, k):
    k %= 8
    return ((v << k) | (v >> (8 - k))) & 0xFF


def power_map(d):
    """x -> x^d di GF(2^8) (0 -> 0). Permutasi jika gcd(d,255)=1."""
    g = np.zeros(256, dtype=np.uint8)
    nz = np.arange(1, 256)
    g[1:] = EXP[(LOG[nz] * d) % 255]
    return g


VALID_EXPS = [d for d in range(1, 255) if math.gcd(d, 255) == 1]  # 128 eksponen permutasi
G_TABLE = np.zeros((255, 256), dtype=np.uint8)
for _d in VALID_EXPS:
    G_TABLE[_d] = power_map(_d)

AES_ROWS = np.array([rotl8(0xF1, i) for i in range(8)], dtype=np.uint8)
AES_C = 0x63


def affine_batch(g, rows, C):
    """g:(n,256) uint8, rows:(n,8) uint8 (baris j = topeng bit keluaran j), C:(n,) -> S:(n,256)."""
    out = np.zeros_like(g)
    for j in range(8):
        out |= (PAR[g & rows[:, j, None]] << j).astype(np.uint8)
    return out ^ np.asarray(C, dtype=np.uint8)[:, None]


AES_SBOX = affine_batch(G_TABLE[254][None, :], AES_ROWS[None, :], np.array([AES_C]))[0]
assert AES_SBOX[0] == 0x63 and AES_SBOX[1] == 0x7C and AES_SBOX[255] == 0x16


def invertible_batch(rows):
    """Cek keterbalikan matriks 8x8 atas GF(2) secara batch (eliminasi Gauss)."""
    M = rows.copy()
    n = M.shape[0]
    ar = np.arange(n)
    ok = np.ones(n, dtype=bool)
    for c in range(8):
        bit = (M >> c) & 1
        cand = bit[:, c:]
        has = cand.any(axis=1)
        ok &= has
        piv = c + cand.argmax(axis=1)
        rp = M[ar, piv].copy()
        rc = M[:, c].copy()
        M[ar, piv] = rc
        M[:, c] = rp
        pr = M[:, c][:, None]
        mask = ((M >> c) & 1).astype(bool)
        mask[:, c] = False
        M ^= np.where(mask, pr, 0).astype(np.uint8)
    return ok


def circulant_batch(rows):
    r0 = rows[:, 0]
    ok = np.ones(rows.shape[0], dtype=bool)
    for i in range(1, 8):
        rot = (((r0.astype(np.uint16) << i) | (r0.astype(np.uint16) >> (8 - i))) & 0xFF).astype(np.uint8)
        ok &= rows[:, i] == rot
    return ok


def gen_candidates(n, exps, rng):
    """Kandidat acak (d, A non-sirkulan & invertibel, C)."""
    ds, rs = [], []
    got = 0
    while got < n:
        m = int((n - got) * 4) + 64
        rows = rng.integers(0, 256, (m, 8), dtype=np.uint8)
        ok = invertible_batch(rows) & ~circulant_batch(rows)
        rows = rows[ok]
        ds.append(rng.choice(exps, len(rows)))
        rs.append(rows)
        got += len(rows)
    d = np.concatenate(ds)[:n].astype(np.int64)
    rows = np.concatenate(rs)[:n]
    C = rng.integers(0, 256, n, dtype=np.uint8)
    return d, rows, C


def gen_unique(n, exps, rng):
    d, rows, C = gen_candidates(int(n * 1.02) + 10, exps, rng)
    while True:
        key = np.column_stack([d.astype(np.uint8), rows, C])
        _, idx = np.unique(key, axis=0, return_index=True)
        idx = np.sort(idx)
        if len(idx) >= n:
            idx = idx[:n]
            return d[idx], rows[idx], C[idx]
        d2, r2, c2 = gen_candidates(n - len(idx) + 100, exps, rng)
        d, rows, C = np.concatenate([d[idx], d2]), np.concatenate([rows[idx], r2]), np.concatenate([C[idx], c2])


# =====================================================================
# 2. METRIK KRIPTOGRAFI
# =====================================================================
def walsh_spectrum(S):
    S = np.asarray(S, dtype=np.uint8)
    F = 1.0 - 2.0 * PAR[np.arange(1, 256, dtype=np.uint8)[:, None] & S[None, :]]
    return F.astype(np.float32) @ H256  # (255,256)


def ddt_counts(S):
    S = np.asarray(S, dtype=np.uint8)
    dx = np.arange(1, 256)
    D = S[None, :] ^ S[np.arange(256)[None, :] ^ dx[:, None]]
    flat = (np.arange(255)[:, None] * 256 + D).ravel()
    return np.bincount(flat, minlength=255 * 256).reshape(255, 256)


def algebraic_degrees(S):
    T = PAR[np.arange(1, 256, dtype=np.uint8)[:, None] & np.asarray(S, dtype=np.uint8)[None, :]].astype(np.uint8)
    for i in range(8):
        T = T.reshape(255, -1, 2, 2 ** i)
        T[:, :, 1, :] ^= T[:, :, 0, :]
        T = T.reshape(255, 256)
    return (T * POP[None, :]).max(axis=1)


def cycle_info(S):
    seen = np.zeros(256, dtype=bool)
    lens = []
    for s in range(256):
        if not seen[s]:
            n, x = 0, s
            while not seen[x]:
                seen[x] = True
                x = int(S[x])
                n += 1
            lens.append(n)
    return len(lens), max(lens)


def avalanche_batch(S):
    """S:(n,256) -> R:(n,8,36) ; R[:,i,:8]=SAC (bit keluaran j), R[:,i,8:]=BIC-SAC (28 pasangan)."""
    n = S.shape[0]
    R = np.empty((n, 8, 36), dtype=np.float32)
    for i in range(8):
        D = S[:, IDX0[i]] ^ S[:, IDX0[i] ^ (1 << i)]
        P = PAR[D[:, :, None] & MASKS]
        R[:, i, :] = P.sum(axis=1, dtype=np.int32) / 128.0
    return R


def summarize_avalanche(R):
    sac, bic = R[:, :, :8], R[:, :, 8:]
    return dict(
        sac_mean=sac.mean(axis=(1, 2)),
        sac_max_dev=np.abs(sac - 0.5).max(axis=(1, 2)),
        sac_dev_mean=np.abs(sac - 0.5).mean(axis=(1, 2)),
        sac_std=sac.std(axis=(1, 2)),
        bic_sac_mean=bic.mean(axis=(1, 2)),
        bic_sac_max_dev=np.abs(bic - 0.5).max(axis=(1, 2)),
    )


# bobot fungsi loss (semakin kecil semakin baik)
LW = dict(sac_max=0.40, sac_dev=0.15, bic_max=0.20, nl=0.15, du=0.10, fp=0.004)


def loss_struct(m, nl, du):
    nl_gap = np.clip((112 - nl) / 112.0, 0, None)
    du_gap = np.clip((du - 4) / 252.0, 0, None)
    return (LW["sac_max"] * m["sac_max_dev"] + LW["sac_dev"] * m["sac_dev_mean"]
            + LW["bic_max"] * m["bic_sac_max_dev"] + LW["nl"] * nl_gap + LW["du"] * du_gap)


def loss_total(ls, fp, ofp):
    return ls + LW["fp"] * (fp + ofp)


@st.cache_resource(show_spinner=False)
def get_exp_info():
    """NL & DU invarian terhadap transformasi afin -> cukup dihitung sekali per eksponen d."""
    nl = np.zeros(255)
    du = np.zeros(255)
    lap = np.zeros(255)
    gsm = np.zeros(255)
    gss = np.zeros(255)
    for d in VALID_EXPS:
        g = G_TABLE[d]
        mx = np.abs(walsh_spectrum(g)).max()
        nl[d], lap[d] = 128 - mx / 2, mx / 512
        du[d] = ddt_counts(g).max()
        m = summarize_avalanche(avalanche_batch(g[None, :]))
        gsm[d], gss[d] = m["sac_max_dev"][0], m["sac_std"][0]
    table = pd.DataFrame({"d": VALID_EXPS, "NL": nl[VALID_EXPS], "DU": du[VALID_EXPS], "LAP": lap[VALID_EXPS]})
    return dict(nl=nl, du=du, lap=lap, gsac_max=gsm, gsac_std=gss, table=table)


def select_exponents(k):
    t = get_exp_info()["table"].sort_values(["NL", "DU", "d"], ascending=[False, True, True])
    return np.array(sorted(t["d"].head(k).tolist()))


def evaluate_batch(d, rows, C, chunk=500):
    """Hitung semua metrik kandidat (d, A, C) -> DataFrame."""
    info = get_exp_info()
    parts = []
    for s in range(0, len(d), chunk):
        sl = slice(s, s + chunk)
        S = affine_batch(G_TABLE[d[sl]], rows[sl], C[sl])
        m = summarize_avalanche(avalanche_batch(S))
        m["fp"] = (S == XS[None, :]).sum(axis=1)
        m["ofp"] = (S == (XS ^ 255)[None, :]).sum(axis=1)
        parts.append(pd.DataFrame(m))
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "d", d)
    for j in range(8):
        df[f"A{j}"] = rows[:, j]
    df["C"] = C
    df["nl"], df["du"] = info["nl"][d], info["du"][d]
    df["loss_struct"] = loss_struct(df, df["nl"].values, df["du"].values)
    df["loss_score"] = loss_total(df["loss_struct"].values, df["fp"].values, df["ofp"].values)
    return df


def full_metrics(S):
    """Semua metrik untuk satu S-box (dipakai tab Evaluasi)."""
    S = np.asarray(S, dtype=np.uint8)
    W = walsh_spectrum(S)
    comp_nl = 128 - np.abs(W).max(axis=1) / 2
    ddt = ddt_counts(S)
    R = avalanche_batch(S[None, :])
    m = summarize_avalanche(R)
    deg = algebraic_degrees(S)
    ncyc, maxcyc = cycle_info(S)
    pair_idx = [int(mk) - 1 for mk in PAIR_MASKS]
    fp = int((S == XS).sum())
    ofp = int((S == (XS ^ 255)).sum())
    out = {
        "Bijektif (permutasi 0..255)": bool(len(set(S.tolist())) == 256),
        "Nonlinearity (NL)": int(comp_nl.min()),
        "BIC-NL (min NL pasangan bit)": int(comp_nl[pair_idx].min()),
        "Differential uniformity (DU)": int(ddt.max()),
        "DAP maks": float(ddt.max() / 256),
        "LAP maks": float(np.abs(W).max() / 512),
        "SAC rata-rata (ideal 0.5)": float(m["sac_mean"][0]),
        "SAC dev maks": float(m["sac_max_dev"][0]),
        "SAC std": float(m["sac_std"][0]),
        "BIC-SAC rata-rata (ideal 0.5)": float(m["bic_sac_mean"][0]),
        "BIC-SAC dev maks": float(m["bic_sac_max_dev"][0]),
        "Derajat aljabar (min)": int(deg.min()),
        "Fixed point": fp,
        "Opposite fixed point": ofp,
        "Jumlah siklus": ncyc,
        "Siklus terpanjang": maxcyc,
    }
    return out, R[0], ddt


@st.cache_data(show_spinner=False)
def cached_metrics(sbox_bytes):
    return full_metrics(np.frombuffer(sbox_bytes, dtype=np.uint8))


# =====================================================================
# 3. PIPELINE ML
# =====================================================================
def make_features(d, rows):
    info = get_exp_info()
    n = len(d)
    bits = (rows[:, :, None] >> np.arange(8)) & 1
    rw, cw = bits.sum(axis=2), bits.sum(axis=1)
    names = (["d", "nl_d", "du_d", "lap_d", "gsac_max_d", "gsac_std_d"]
             + [f"bit_r{j}c{k}" for j in range(8) for k in range(8)]
             + [f"rowW{j}" for j in range(8)] + [f"colW{k}" for k in range(8)]
             + ["rowW_mean", "rowW_std", "rowW_min", "rowW_max", "colW_std", "colW_min", "colW_max"])
    X = np.column_stack([d, info["nl"][d], info["du"][d], info["lap"][d], info["gsac_max"][d], info["gsac_std"][d],
                         bits.reshape(n, 64), rw, cw, rw.mean(1), rw.std(1), rw.min(1), rw.max(1),
                         cw.std(1), cw.min(1), cw.max(1)]).astype(np.float32)
    return X, names


def best_C_scan(base, ls):
    """base: S tanpa C (256,), ls: loss struktural. Scan eksak seluruh C=0..255 (bincount)."""
    fpC = np.bincount(base ^ XS, minlength=256)              # S0[x]^x == C  -> fixed point
    ofpC = np.bincount(base ^ XS ^ 255, minlength=256)       # S0[x]^x^255 == C -> opposite fixed point
    loss = loss_total(ls, fpC, ofpC)
    best = np.flatnonzero(np.isclose(loss, loss.min()))
    cyc = [cycle_info(base ^ np.uint8(c))[1] for c in best]  # tie-break: siklus terpanjang
    c_best = int(best[int(np.argmax(cyc))])
    return c_best, loss


@st.cache_resource(show_spinner=False)
def run_pipeline(n_samples, seed, n_exp, pool_size, top_k):
    t_all = time.time()
    rng = np.random.default_rng(seed)
    exps = select_exponents(n_exp)

    # --- dataset ---
    t0 = time.time()
    d, rows, C = gen_unique(n_samples, exps, rng)
    df = evaluate_batch(d, rows, C)
    t_data = time.time() - t0
    n_dup = int(n_samples - len(np.unique(np.column_stack([d.astype(np.uint8), rows, C]), axis=0)))

    # --- baseline AES ---
    aes_m, _, _ = full_metrics(AES_SBOX)
    aes_df = evaluate_batch(np.array([254]), AES_ROWS[None, :], np.array([AES_C], dtype=np.uint8))
    aes_loss = float(aes_df["loss_score"][0])

    # --- surrogate ML ---
    X, fnames = make_features(d, rows)
    y = df["loss_struct"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=5,
                                              max_features=0.5, n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                              random_state=seed),
    }
    bench, preds = [], {}
    for name, mdl in models.items():
        t1 = time.time()
        mdl.fit(Xtr, ytr)
        t_fit = time.time() - t1
        p = mdl.predict(Xte)
        preds[name] = p
        top = np.argsort(p)[:100]
        rho, pv = stats.spearmanr(p, yte)
        bench.append(dict(Model=name, R2=r2_score(yte, p), MAE=mean_absolute_error(yte, p),
                          Spearman=rho, Spearman_p=pv, **{"Lift@100": yte.mean() / yte[top].mean()},
                          Waktu_latih_s=t_fit))
    bench = pd.DataFrame(bench)
    best_name = bench.sort_values("Spearman", ascending=False)["Model"].iloc[0]
    best_model = models[best_name]
    imp = None
    if hasattr(models["RandomForest"], "feature_importances_"):
        imp = pd.Series(models["RandomForest"].feature_importances_, index=fnames).sort_values(ascending=False)

    # --- statistik EDA ---
    groups = [g["loss_struct"].values for _, g in df.groupby("d") if len(g) > 1]
    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    best_rho = float(bench.set_index("Model").loc[best_name, "Spearman"])
    best_rho_p = float(bench.set_index("Model").loc[best_name, "Spearman_p"])

    # --- pencarian pool kandidat ---
    pd_, pr_, pc_ = gen_candidates(pool_size, exps, rng)
    t1 = time.time()
    Xp, _ = make_features(pd_, pr_)
    pred_pool = best_model.predict(Xp)
    t_sur = time.time() - t1
    order = np.argsort(pred_pool)
    sel = order[:top_k]

    # estimasi biaya evaluasi eksak seluruh pool
    sub = rng.choice(pool_size, size=min(1000, pool_size), replace=False)
    t1 = time.time()
    evaluate_batch(pd_[sub], pr_[sub], pc_[sub])
    t_exact_est = (time.time() - t1) / len(sub) * pool_size

    # evaluasi eksak finalis + scan eksak C
    fin = evaluate_batch(pd_[sel], pr_[sel], np.zeros(len(sel), dtype=np.uint8))
    base = affine_batch(G_TABLE[pd_[sel]], pr_[sel], np.zeros(len(sel), dtype=np.uint8))
    best_cs, best_loss, loss_curves = [], [], []
    for i in range(len(sel)):
        c, lc = best_C_scan(base[i], fin["loss_struct"].values[i])
        best_cs.append(c)
        best_loss.append(lc[c])
        loss_curves.append(lc)
    fin["C"] = best_cs
    S_fin = base ^ np.array(best_cs, dtype=np.uint8)[:, None]
    fin["fp"] = (S_fin == XS[None, :]).sum(axis=1)
    fin["ofp"] = (S_fin == (XS ^ 255)[None, :]).sum(axis=1)
    fin["loss_score"] = best_loss
    fin["pred_struct"] = pred_pool[sel]
    fin = fin.sort_values("loss_score").reset_index(drop=True)

    # pembanding acak (jumlah evaluasi eksak sama)
    ridx = rng.choice(pool_size, size=top_k, replace=False)
    rnd = evaluate_batch(pd_[ridx], pr_[ridx], np.zeros(top_k, dtype=np.uint8))
    lift_pool = float(rnd["loss_struct"].mean() / fin["loss_struct"].mean())

    w = fin.iloc[0]
    w_rows = np.array([int(w[f"A{j}"]) for j in range(8)], dtype=np.uint8)
    w_d, w_C = int(w["d"]), int(w["C"])
    w_S = affine_batch(G_TABLE[[w_d]], w_rows[None, :], np.array([w_C], dtype=np.uint8))[0]
    base_w = affine_batch(G_TABLE[[w_d]], w_rows[None, :], np.zeros(1, dtype=np.uint8))[0]
    _, w_curve = best_C_scan(base_w, float(w["loss_struct"]))
    w_m, _, _ = full_metrics(w_S)

    return dict(
        exps=exps, df=df, n_dup=n_dup, bench=bench, best_name=best_name, imp=imp, anova=anova, kruskal=kruskal,
        best_rho=best_rho, best_rho_p=best_rho_p, yte=yte, pred_te=preds[best_name],
        t_sur=t_sur, t_exact_est=t_exact_est, t_data=t_data, lift_pool=lift_pool,
        fin=fin, rnd_loss=float(rnd["loss_struct"].mean()),
        winner=dict(d=w_d, rows=w_rows, C=w_C, S=w_S, loss=float(w["loss_score"]), metrics=w_m, curve=w_curve),
        aes=dict(loss=aes_loss, metrics=aes_m), pool_size=pool_size, top_k=top_k, total_time=time.time() - t_all,
    )


# =====================================================================
# 4. AES-128 (S-box kustom) + CBC
# =====================================================================
def _ror(v, k):
    return ((v >> k) | (v << (32 - k))) & 0xFFFFFFFF


class AES128:
    def __init__(self, sbox, key):
        S = [int(v) for v in sbox]
        Si = [0] * 256
        for i, v in enumerate(S):
            Si[v] = i
        self.S, self.Si = S, Si
        self.Te0 = [(gmul(s, 2) << 24) | (s << 16) | (s << 8) | gmul(s, 3) for s in S]
        self.Td0 = [(gmul(s, 14) << 24) | (gmul(s, 9) << 16) | (gmul(s, 13) << 8) | gmul(s, 11) for s in Si]
        self.Te1 = [_ror(v, 8) for v in self.Te0]
        self.Te2 = [_ror(v, 16) for v in self.Te0]
        self.Te3 = [_ror(v, 24) for v in self.Te0]
        self.Td1 = [_ror(v, 8) for v in self.Td0]
        self.Td2 = [_ror(v, 16) for v in self.Td0]
        self.Td3 = [_ror(v, 24) for v in self.Td0]
        self.ek = self._expand(key)
        dk = []
        for r in range(11):
            for i in range(4):
                w = self.ek[4 * (10 - r) + i]
                if 0 < r < 10:
                    w = (self.Td0[S[w >> 24]] ^ self.Td1[S[(w >> 16) & 255]]
                         ^ self.Td2[S[(w >> 8) & 255]] ^ self.Td3[S[w & 255]])
                dk.append(w)
        self.dk = dk

    def _expand(self, key):
        S = self.S
        w = list(struct.unpack(">4I", key))
        rcon = 1
        for i in range(4, 44):
            t = w[i - 1]
            if i % 4 == 0:
                t = ((t << 8) & 0xFFFFFFFF) | (t >> 24)
                t = (S[t >> 24] << 24) | (S[(t >> 16) & 255] << 16) | (S[(t >> 8) & 255] << 8) | S[t & 255]
                t ^= rcon << 24
                rcon = gmul(rcon, 2)
            w.append(w[i - 4] ^ t)
        return w

    def _enc(self, s0, s1, s2, s3):
        rk, T0, T1, T2, T3, S = self.ek, self.Te0, self.Te1, self.Te2, self.Te3, self.S
        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3]
        for r in range(1, 10):
            k = 4 * r
            t0 = T0[s0 >> 24] ^ T1[(s1 >> 16) & 255] ^ T2[(s2 >> 8) & 255] ^ T3[s3 & 255] ^ rk[k]
            t1 = T0[s1 >> 24] ^ T1[(s2 >> 16) & 255] ^ T2[(s3 >> 8) & 255] ^ T3[s0 & 255] ^ rk[k + 1]
            t2 = T0[s2 >> 24] ^ T1[(s3 >> 16) & 255] ^ T2[(s0 >> 8) & 255] ^ T3[s1 & 255] ^ rk[k + 2]
            t3 = T0[s3 >> 24] ^ T1[(s0 >> 16) & 255] ^ T2[(s1 >> 8) & 255] ^ T3[s2 & 255] ^ rk[k + 3]
            s0, s1, s2, s3 = t0, t1, t2, t3
        o0 = ((S[s0 >> 24] << 24) | (S[(s1 >> 16) & 255] << 16) | (S[(s2 >> 8) & 255] << 8) | S[s3 & 255]) ^ rk[40]
        o1 = ((S[s1 >> 24] << 24) | (S[(s2 >> 16) & 255] << 16) | (S[(s3 >> 8) & 255] << 8) | S[s0 & 255]) ^ rk[41]
        o2 = ((S[s2 >> 24] << 24) | (S[(s3 >> 16) & 255] << 16) | (S[(s0 >> 8) & 255] << 8) | S[s1 & 255]) ^ rk[42]
        o3 = ((S[s3 >> 24] << 24) | (S[(s0 >> 16) & 255] << 16) | (S[(s1 >> 8) & 255] << 8) | S[s2 & 255]) ^ rk[43]
        return o0, o1, o2, o3

    def _dec(self, s0, s1, s2, s3):
        rk, T0, T1, T2, T3, Si = self.dk, self.Td0, self.Td1, self.Td2, self.Td3, self.Si
        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3]
        for r in range(1, 10):
            k = 4 * r
            t0 = T0[s0 >> 24] ^ T1[(s3 >> 16) & 255] ^ T2[(s2 >> 8) & 255] ^ T3[s1 & 255] ^ rk[k]
            t1 = T0[s1 >> 24] ^ T1[(s0 >> 16) & 255] ^ T2[(s3 >> 8) & 255] ^ T3[s2 & 255] ^ rk[k + 1]
            t2 = T0[s2 >> 24] ^ T1[(s1 >> 16) & 255] ^ T2[(s0 >> 8) & 255] ^ T3[s3 & 255] ^ rk[k + 2]
            t3 = T0[s3 >> 24] ^ T1[(s2 >> 16) & 255] ^ T2[(s1 >> 8) & 255] ^ T3[s0 & 255] ^ rk[k + 3]
            s0, s1, s2, s3 = t0, t1, t2, t3
        o0 = ((Si[s0 >> 24] << 24) | (Si[(s3 >> 16) & 255] << 16) | (Si[(s2 >> 8) & 255] << 8) | Si[s1 & 255]) ^ rk[40]
        o1 = ((Si[s1 >> 24] << 24) | (Si[(s0 >> 16) & 255] << 16) | (Si[(s3 >> 8) & 255] << 8) | Si[s2 & 255]) ^ rk[41]
        o2 = ((Si[s2 >> 24] << 24) | (Si[(s1 >> 16) & 255] << 16) | (Si[(s0 >> 8) & 255] << 8) | Si[s3 & 255]) ^ rk[42]
        o3 = ((Si[s3 >> 24] << 24) | (Si[(s2 >> 16) & 255] << 16) | (Si[(s1 >> 8) & 255] << 8) | Si[s0 & 255]) ^ rk[43]
        return o0, o1, o2, o3

    def encrypt_cbc(self, data, iv):
        assert len(data) % 16 == 0 and len(iv) == 16
        n = len(data) // 16
        w = struct.unpack(f">{4 * n}I", data)
        p0, p1, p2, p3 = struct.unpack(">4I", iv)
        out = []
        enc = self._enc
        for k in range(0, 4 * n, 4):
            p0, p1, p2, p3 = enc(w[k] ^ p0, w[k + 1] ^ p1, w[k + 2] ^ p2, w[k + 3] ^ p3)
            out += (p0, p1, p2, p3)
        return struct.pack(f">{4 * n}I", *out)

    def decrypt_cbc(self, data, iv):
        assert len(data) % 16 == 0 and len(iv) == 16
        n = len(data) // 16
        w = struct.unpack(f">{4 * n}I", data)
        p0, p1, p2, p3 = struct.unpack(">4I", iv)
        out = []
        dec = self._dec
        for k in range(0, 4 * n, 4):
            d0, d1, d2, d3 = dec(w[k], w[k + 1], w[k + 2], w[k + 3])
            out += (d0 ^ p0, d1 ^ p1, d2 ^ p2, d3 ^ p3)
            p0, p1, p2, p3 = w[k], w[k + 1], w[k + 2], w[k + 3]
        return struct.pack(f">{4 * n}I", *out)


def pkcs7_pad(b):
    n = 16 - len(b) % 16
    return b + bytes([n]) * n


def pkcs7_unpad(b):
    if not b or len(b) % 16:
        raise ValueError("Panjang data tidak valid")
    n = b[-1]
    if n < 1 or n > 16 or b[-n:] != bytes([n]) * n:
        raise ValueError("Padding tidak valid (kunci/IV salah atau data rusak)")
    return b[:-n]


def parse_key(mode, text):
    if mode == "Teks (dipotong/dipad 16 byte)":
        return (text.encode("utf-8") + b"\x00" * 16)[:16]
    h = text.strip().replace(" ", "")
    if len(h) != 32:
        raise ValueError("Kunci hex harus 32 digit (16 byte)")
    return bytes.fromhex(h)


def parse_iv(text):
    h = text.strip().replace(" ", "")
    if len(h) != 32:
        raise ValueError("IV hex harus 32 digit (16 byte)")
    return bytes.fromhex(h)


@st.cache_resource(show_spinner=False)
def get_cipher(sbox_bytes, key):
    return AES128(np.frombuffer(sbox_bytes, dtype=np.uint8), key)


def aes_selftest():
    """Vektor uji FIPS-197 (AES-128) dengan S-box standar + round-trip dengan S-box usulan."""
    key = bytes(range(16))
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    ct = AES128(AES_SBOX, key).encrypt_cbc(pt, bytes(16))
    return ct.hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"


# =====================================================================
# 5. UTILITAS TAMPILAN
# =====================================================================
def sbox_df(S):
    a = np.asarray(S).reshape(16, 16)
    return pd.DataFrame([[f"{v:02X}" for v in r] for r in a],
                        index=[f"{i:X}_" for i in range(16)], columns=[f"_{i:X}" for i in range(16)])


def rows_hex(rows):
    return " ".join(f"{int(r):02X}" for r in rows)


def entropy(arr):
    c = np.bincount(np.asarray(arr).ravel(), minlength=256).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def adjacent_corr(img):
    g = img if img.ndim == 2 else img.mean(axis=2)
    a, b = g[:, :-1].ravel(), g[:, 1:].ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sample_image(size=128):
    im = Image.new("RGB", (size, size))
    px = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    px[..., 0] = (xx * 255 // size)
    px[..., 1] = (yy * 255 // size)
    px[..., 2] = 160
    im = Image.fromarray(px)
    dr = ImageDraw.Draw(im)
    dr.ellipse([size * .15, size * .15, size * .6, size * .6], fill=(250, 220, 40), outline=(0, 0, 0), width=3)
    dr.rectangle([size * .45, size * .5, size * .9, size * .85], fill=(220, 40, 60), outline=(255, 255, 255), width=3)
    dr.polygon([(size * .1, size * .9), (size * .3, size * .6), (size * .5, size * .9)], fill=(30, 150, 80))
    return im


def fig_hist(ax, data, title, xlabel, aes_val=None, prop_val=None):
    ax.hist(data, bins=50, color="#C8102E", alpha=.85)
    if aes_val is not None:
        ax.axvline(aes_val, color="#333333", ls="--", lw=1.5, label="AES")
    if prop_val is not None:
        ax.axvline(prop_val, color="#0B6E3A", lw=1.5, label="Proposed")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)
    if aes_val is not None or prop_val is not None:
        ax.legend(fontsize=7)



THEME_CSS = """
<style>
:root { --merah:#E53935; --merah-muda:#FF8A80; --panel:#2A1214; --garis:#5A2226; }
.block-container { padding-top: 1.2rem; }
.kopdes-banner {
  background: linear-gradient(90deg, #7F0D1B 0%, #C8102E 60%, #E03A3E 100%);
  border-radius: 14px; padding: 18px 26px; color: #fff; margin-bottom: 6px;
  border-bottom: 6px solid #fff; box-shadow: 0 3px 0 #C8102E;
}
.kopdes-banner .tag { font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; opacity: .95; }
.kopdes-banner h1 { color:#fff !important; font-size: 1.75rem; line-height: 1.25; margin: 4px 0 0 0; padding: 0; }
[data-testid="stSidebar"] { background: #1E0E10; border-right: 4px solid var(--merah); }
[data-testid="stSidebar"] h2 { color: #fff; }
[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid var(--garis); border-left: 6px solid var(--merah);
  border-radius: 10px; padding: 10px 14px;
}
[data-testid="stMetricValue"] { color: #fff; }
[data-testid="stMetricLabel"] { color: #F5C6C6; }
button[role="tab"][aria-selected="true"] { color: var(--merah-muda) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--merah) !important; }
h2, h3 { color: var(--merah-muda); }
.stButton > button, .stDownloadButton > button {
  background: var(--merah); color: #fff; border: 0; border-radius: 8px;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: #B71C1C; color: #fff; }
hr { border-color: var(--garis); }
</style>
"""

BANNER = """
<div class="kopdes-banner">
  <div class="tag">🇮🇩 Kopdes Merah Putih &nbsp;•&nbsp; Tugas Kriptografi</div>
  <h1>Modifikasi AES S-box Berbasis Machine Learning &amp; Demo AES-128 CBC</h1>
</div>
"""

# =====================================================================
# 6. APLIKASI
# =====================================================================
def main():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    # ---------------- sidebar: pengaturan ----------------
    with st.sidebar:
        st.markdown("## 🔐 AES S-box + ML")
        with st.expander("⚙️ Pengaturan pipeline", expanded=False):
            n_samples = st.select_slider("Sampel dataset unik", [5000, 10000, 20000, 30000], value=20000)
            n_exp = st.select_slider("Jumlah eksponen d (terbaik by NL/DU)", [16, 30, 64, 128], value=30)
            pool_size = st.select_slider("Kandidat pencarian", [10000, 50000, 100000], value=50000)
            top_k = st.select_slider("Finalis (evaluasi eksak)", [100, 250, 500, 1000, 2000], value=500)
            seed = st.number_input("Seed", 0, 10_000, 42)

    with st.spinner("Menjalankan pipeline ML (pertama kali ±1 menit, lalu di-cache)…"):
        R = run_pipeline(int(n_samples), int(seed), int(n_exp), int(pool_size), int(top_k))
    win, aes = R["winner"], R["aes"]
    wm, am = win["metrics"], aes["metrics"]
    prop_sbox = win["S"].astype(np.uint8)

    with st.sidebar:
        st.caption(f"Opsi: {n_exp} eksponen power-map × Matriks Afin Non-Sirkulan GL(8,2) + "
                   "Optimasi Eksak C & Surrogate ML")
        st.divider()
        st.metric("Proposed ML loss_score", f"{win['loss']:.6f}",
                  f"{win['loss'] - aes['loss']:+.6f} vs AES", delta_color="inverse")
        st.metric("Proposed SAC Mean", f"{wm['SAC rata-rata (ideal 0.5)']:.5f}",
                  f"err {abs(wm['SAC rata-rata (ideal 0.5)'] - 0.5):.5f} (ideal 0.5)", delta_color="off")
        st.metric("Proposed BIC-SAC", f"{wm['BIC-SAC rata-rata (ideal 0.5)']:.5f}",
                  f"err {abs(wm['BIC-SAC rata-rata (ideal 0.5)'] - 0.5):.5f}", delta_color="off")
        st.divider()
        st.markdown("**Konfigurasi Terpilih:**")
        st.code(f"d = {win['d']}\nA (baris, hex) = {rows_hex(win['rows'])}\nC = 0x{win['C']:02X}", language="text")
        with st.expander("Self-test AES"):
            st.write("Vektor FIPS-197 (S-box standar):", "✅ lulus" if aes_selftest() else "❌ gagal")

    st.markdown(BANNER, unsafe_allow_html=True)
    st.caption(f"Ruang pencarian {n_exp} eksponen power-map d × Matriks Afin Non-Sirkulan GL(8,2) "
               "+ Optimasi Eksak Konstanta C ∈ [0..255]  |  S(x) = A·x^d ⊕ C pada GF(2⁸)")

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Pipeline ML & Benchmark", "2️⃣ Evaluasi & Inspeksi S-box",
                                       "3️⃣ Demo Enkripsi Teks (AES-128 CBC)", "4️⃣ Demo Enkripsi Gambar (AES-128 CBC)"])

    # ================= TAB 1 =================
    with tab1:
        df = R["df"]
        st.header(f"1. Ringkasan Dataset & Temuan EDA ({len(df):,} Sampel Unik)".replace(",", "."))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sampel Dataset Unik", f"{len(df):,}".replace(",", "."), f"{R['n_dup']} duplikat kanonik",
                  delta_color="off")
        c2.metric(f"ANOVA {len(R['exps'])} Eksponen", f"F = {R['anova'].statistic:.2f}", f"p = {R['anova'].pvalue:.2e}",
                  delta_color="off")
        c3.metric(f"Spearman ρ ({R['best_name']})", f"{R['best_rho']:.4f}", f"p = {R['best_rho_p']:.2e}",
                  delta_color="off")
        c4.metric(f"Kecepatan {R['pool_size'] // 1000}k Kandidat", f"{R['t_sur']:.2f}s",
                  f"vs eksak ≈ {R['t_exact_est']:.1f}s", delta_color="off")

        colA, colB = st.columns(2)
        with colA:
            fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
            fig_hist(axs[0], df["loss_score"], "Distribusi loss_score", "loss_score", aes["loss"], win["loss"])
            fig_hist(axs[1], df["sac_max_dev"], "Distribusi sac_max_dev", "sac_max_dev",
                     am["SAC dev maks"], wm["SAC dev maks"])
            fig_hist(axs[2], df["sac_std"], "Distribusi sac_std", "sac_std", am["SAC std"], wm["SAC std"])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Distribusi loss_score, sac_max_dev, & sac_std ({len(df):,} kandidat)".replace(",", "."))
        with colB:
            order = df.groupby("d")["loss_struct"].mean().sort_values().index
            fig, ax = plt.subplots(figsize=(11, 3.2))
            ax.boxplot([df.loc[df["d"] == d_, "loss_struct"] for d_ in order], showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor="#E57373"))
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels([str(d_) for d_ in order], rotation=90, fontsize=6)
            ax.set_xlabel("Eksponen d (diurutkan berdasarkan mean loss_struct)", fontsize=8)
            ax.set_ylabel("loss_struct", fontsize=8)
            ax.set_title(f"Pengaruh eksponen d terhadap loss (ANOVA p={R['anova'].pvalue:.2e}, "
                         f"Kruskal p={R['kruskal'].pvalue:.2e})", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Pengaruh eksponen power-map terhadap loss struktural")

        st.divider()
        st.header("2. Surrogate ML & Benchmark")
        st.markdown(
            "Target surrogate adalah **loss struktural** (SAC, BIC-SAC, NL, DU) yang bergantung pada `(d, A)`. "
            "Bagian yang bergantung pada `C` (fixed point & opposite fixed point) **tidak** diprediksi ML karena "
            "dapat dihitung eksak untuk ke-256 nilai C — itulah tahap *Optimasi Eksak C*.")
        b = R["bench"].copy()
        st.dataframe(b.style.format({"R2": "{:.4f}", "MAE": "{:.5f}", "Spearman": "{:.4f}", "Spearman_p": "{:.2e}",
                                     "Lift@100": "{:.2f}×", "Waktu_latih_s": "{:.1f}"}), hide_index=True,
                     width="stretch")
        st.caption(f"Model terbaik (Spearman tertinggi): **{R['best_name']}**. "
                   "Lift@100 = rata-rata loss seluruh data uji ÷ rata-rata loss 100 kandidat teratas versi surrogate.")
        colC, colD = st.columns(2)
        with colC:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hexbin(R["yte"], R["pred_te"], gridsize=40, cmap="Reds", mincnt=1)
            lo, hi = R["yte"].min(), R["yte"].max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlabel("loss_struct aktual (data uji)")
            ax.set_ylabel("loss_struct prediksi")
            ax.set_title(f"Prediksi vs aktual — {R['best_name']}", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with colD:
            if R["imp"] is not None:
                top = R["imp"].head(12)[::-1]
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.barh(top.index, top.values, color="#C8102E")
                ax.set_title("Feature importance (RandomForest)", fontsize=10)
                ax.tick_params(labelsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.divider()
        st.header("3. Pencarian Kandidat & Optimasi Eksak C")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Kandidat diskor surrogate", f"{R['pool_size']:,}".replace(",", "."))
        k2.metric("Finalis dievaluasi eksak", R["top_k"])
        k3.metric("Lift@K (pool)", f"{R['lift_pool']:.2f}×", "vs acak dgn K sama", delta_color="off")
        k4.metric("Total waktu pipeline", f"{R['total_time']:.1f}s")
        st.markdown("**10 finalis terbaik** (setelah scan eksak seluruh C ∈ [0..255]):")
        fin = R["fin"].head(10).copy()
        fin["A (baris hex)"] = [rows_hex([r[f"A{j}"] for j in range(8)]) for _, r in fin.iterrows()]
        fin["C"] = fin["C"].map(lambda v: f"0x{int(v):02X}")
        st.dataframe(fin[["d", "A (baris hex)", "C", "nl", "du", "sac_max_dev", "bic_sac_max_dev", "fp", "ofp",
                          "loss_struct", "loss_score"]].style.format(
            {"sac_max_dev": "{:.4f}", "bic_sac_max_dev": "{:.4f}", "loss_struct": "{:.6f}", "loss_score": "{:.6f}"}),
            hide_index=True, width="stretch")

        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.bar(range(256), win["curve"], color="#E9A3A8")
        ax.bar([win["C"]], [win["curve"][win["C"]]], color="#0B6E3A", label=f"C terbaik = 0x{win['C']:02X}")
        ax.set_xlabel("Konstanta C")
        ax.set_ylabel("loss_score")
        ax.set_title("Scan eksak C untuk konfigurasi pemenang", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.header("4. Hasil vs AES")
        cmp_ = pd.DataFrame({
            "Metrik": ["loss_score", "NL", "DU", "SAC dev maks", "BIC-SAC dev maks", "Fixed point", "Opposite FP"],
            "AES": [aes["loss"], am["Nonlinearity (NL)"], am["Differential uniformity (DU)"], am["SAC dev maks"],
                    am["BIC-SAC dev maks"], am["Fixed point"], am["Opposite fixed point"]],
            "Proposed": [win["loss"], wm["Nonlinearity (NL)"], wm["Differential uniformity (DU)"],
                         wm["SAC dev maks"], wm["BIC-SAC dev maks"], wm["Fixed point"], wm["Opposite fixed point"]],
        })
        st.dataframe(cmp_.style.format({"AES": "{:.6g}", "Proposed": "{:.6g}"}), hide_index=True,
                     width="stretch")
        st.info("Catatan: formula loss = 0.40·SAC_maxdev + 0.15·SAC_meandev + 0.20·BICSAC_maxdev + "
                "0.15·gap_NL + 0.10·gap_DU + 0.004·(fixed point + opposite fixed point). "
                "NL & DU tidak berubah oleh transformasi afin (A, C), jadi ditentukan oleh eksponen d.")

    # ================= TAB 2 =================
    with tab2:
        st.header("Evaluasi & Inspeksi S-box")
        choice = st.radio("S-box yang diinspeksi", ["Proposed (ML)", "AES standar"], horizontal=True)
        S_sel = prop_sbox if choice.startswith("Proposed") else AES_SBOX
        st.subheader("Tabel S-box (heksadesimal)")
        st.dataframe(sbox_df(S_sel), width="stretch")

        m_sel, R_sel, ddt_sel = cached_metrics(S_sel.tobytes())
        m_aes, _, _ = cached_metrics(AES_SBOX.tobytes())
        m_prop, _, _ = cached_metrics(prop_sbox.tobytes())
        keys = list(m_prop.keys())
        tbl = pd.DataFrame({"Metrik": keys, "AES": [m_aes[k] for k in keys], "Proposed (ML)": [m_prop[k] for k in keys]})
        tbl["AES"] = tbl["AES"].map(lambda v: f"{v:.5f}" if isinstance(v, float) else str(v))
        tbl["Proposed (ML)"] = tbl["Proposed (ML)"].map(lambda v: f"{v:.5f}" if isinstance(v, float) else str(v))
        st.subheader("Perbandingan metrik kriptografi")
        st.dataframe(tbl, hide_index=True, width="stretch")

        ca, cb = st.columns(2)
        with ca:
            sac = R_sel[:, :8].T  # (bit keluaran j, bit masukan i)
            fig, ax = plt.subplots(figsize=(5, 4.2))
            im = ax.imshow(sac, cmap="RdBu_r", vmin=0.4, vmax=0.6)
            for j in range(8):
                for i in range(8):
                    ax.text(i, j, f"{sac[j, i]:.3f}", ha="center", va="center", fontsize=6.5)
            ax.set_xlabel("bit masukan i")
            ax.set_ylabel("bit keluaran j")
            ax.set_title(f"Matriks SAC — {choice}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with cb:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            vals, cnt = np.unique(ddt_sel, return_counts=True)
            ax.bar([str(v) for v in vals], cnt, color="#C8102E")
            ax.set_yscale("log")
            ax.set_xlabel("Nilai entri DDT")
            ax.set_ylabel("Jumlah (log)")
            ax.set_title(f"Distribusi DDT — {choice}", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        inv = np.zeros(256, dtype=np.uint8)
        inv[S_sel] = np.arange(256, dtype=np.uint8)
        st.success("Uji invers: S⁻¹(S(x)) = x untuk semua x ✅" if bool((inv[S_sel] == XS).all())
                   else "Uji invers gagal ❌")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Unduh S-box (CSV)", ",".join(map(str, S_sel.tolist())), "sbox.csv", "text/csv")
        d2.download_button("⬇️ Unduh parameter konfigurasi (JSON)", json.dumps(
            dict(d=win["d"], A_rows_hex=[f"{int(r):02X}" for r in win["rows"]], C=f"{win['C']:02X}",
                 polynomial="0x11B", sbox=win["S"].tolist()), indent=2), "konfigurasi_sbox.json", "application/json")

    # ================= TAB 3 =================
    with tab3:
        st.header("Demo Enkripsi Teks — AES-128 CBC")
        sb_choice = st.radio("S-box yang dipakai", ["Proposed (ML)", "AES standar"], horizontal=True, key="sb_text")
        sb = prop_sbox if sb_choice.startswith("Proposed") else AES_SBOX
        k1, k2 = st.columns(2)
        with k1:
            kmode = st.radio("Format kunci", ["Teks (dipotong/dipad 16 byte)", "Hex (32 digit)"], horizontal=True)
            ktext = st.text_input("Kunci", "kriptografi-2026" if kmode.startswith("Teks") else "000102030405060708090a0b0c0d0e0f")
        with k2:
            if "iv_hex" not in st.session_state:
                st.session_state.iv_hex = os.urandom(16).hex()
            if st.button("🎲 Buat IV baru"):
                st.session_state.iv_hex = os.urandom(16).hex()
            iv_text = st.text_input("IV (hex 32 digit)", key="iv_hex")
        plain = st.text_area("Plaintext", "Kriptografi itu seru! S-box ini dibangun dengan bantuan machine learning.")
        try:
            key, iv = parse_key(kmode, ktext), parse_iv(iv_text)
            cipher = get_cipher(sb.tobytes(), key)
            std_cipher = get_cipher(AES_SBOX.tobytes(), key)
            pt = plain.encode("utf-8")
            ct = cipher.encrypt_cbc(pkcs7_pad(pt), iv)
            import base64
            st.subheader("Ciphertext")
            st.code(ct.hex(), language="text")
            st.caption("Base64: " + base64.b64encode(ct).decode())
            back = pkcs7_unpad(cipher.decrypt_cbc(ct, iv)).decode("utf-8", errors="replace")
            st.success(f"Dekripsi kembali: {back}" if back == plain else "Dekripsi tidak cocok ❌")
            ct_std = std_cipher.encrypt_cbc(pkcs7_pad(pt), iv)
            st.caption("Ciphertext dengan S-box AES standar (kunci & IV sama): " + ct_std.hex()[:64] + "…")

            st.subheader("Efek avalanche (ubah 1 bit plaintext)")
            if pt:
                pt2 = bytearray(pt)
                pt2[0] ^= 1
                ct2 = cipher.encrypt_cbc(pkcs7_pad(bytes(pt2)), iv)
                n_diff = sum(bin(a ^ b).count("1") for a, b in zip(ct, ct2))
                st.write(f"Bit ciphertext berubah: **{n_diff} dari {len(ct) * 8}** ({100 * n_diff / (len(ct) * 8):.1f}%) "
                         "— idealnya ≈ 50%.")

            st.subheader("Dekripsi manual")
            ct_in = st.text_input("Tempel ciphertext (hex)", ct.hex())
            if st.button("Dekripsi"):
                try:
                    out = pkcs7_unpad(cipher.decrypt_cbc(bytes.fromhex(ct_in.strip()), iv))
                    st.success("Hasil: " + out.decode("utf-8", errors="replace"))
                except Exception as e:
                    st.error(f"Gagal: {e}")
        except Exception as e:
            st.error(str(e))

    # ================= TAB 4 =================
    with tab4:
        st.header("Demo Enkripsi Gambar — AES-128 CBC")
        sb_choice = st.radio("S-box yang dipakai", ["Proposed (ML)", "AES standar"], horizontal=True, key="sb_img")
        sb = prop_sbox if sb_choice.startswith("Proposed") else AES_SBOX
        up = st.file_uploader("Unggah gambar (PNG/JPG) — kosongkan untuk gambar contoh", type=["png", "jpg", "jpeg"])
        g1, g2, g3 = st.columns(3)
        max_side = g1.select_slider("Sisi maksimum (px)", [64, 96, 128, 192, 256], value=128)
        gray = g2.checkbox("Grayscale", value=False)
        ikey = g3.text_input("Kunci (teks)", "kunci-gambar-2026")
        iv_img = bytes(range(16))
        im = Image.open(up) if up else sample_image(max_side)
        im = im.convert("L" if gray else "RGB")
        im.thumbnail((max_side, max_side))
        arr = np.array(im, dtype=np.uint8)
        key = (ikey.encode() + b"\x00" * 16)[:16]
        cipher = get_cipher(sb.tobytes(), key)
        raw = arr.tobytes()
        t0 = time.time()
        enc = cipher.encrypt_cbc(pkcs7_pad(raw), iv_img)
        t_enc = time.time() - t0
        enc_img = np.frombuffer(enc[:len(raw)], dtype=np.uint8).reshape(arr.shape)
        dec = np.frombuffer(pkcs7_unpad(cipher.decrypt_cbc(enc, iv_img)), dtype=np.uint8).reshape(arr.shape)

        i1, i2, i3 = st.columns(3)
        i1.image(arr, caption=f"Asli ({arr.shape[1]}×{arr.shape[0]})", width="stretch")
        i2.image(enc_img, caption="Terenkripsi (CBC)", width="stretch")
        i3.image(dec, caption="Hasil dekripsi", width="stretch")
        st.success("Dekripsi identik dengan gambar asli ✅" if np.array_equal(arr, dec) else "Dekripsi tidak identik ❌")

        # uji diffusi: ubah 1 piksel pertama
        arr2 = arr.copy()
        arr2.flat[0] ^= 1
        enc2 = cipher.encrypt_cbc(pkcs7_pad(arr2.tobytes()), iv_img)
        e1 = np.frombuffer(enc, dtype=np.uint8).astype(int)
        e2 = np.frombuffer(enc2, dtype=np.uint8).astype(int)
        npcr = float((e1 != e2).mean() * 100)
        uaci = float((np.abs(e1 - e2) / 255).mean() * 100)

        st.subheader("Analisis keamanan visual")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Entropi asli", f"{entropy(arr):.4f}")
        s2.metric("Entropi cipher", f"{entropy(enc_img):.4f}", "ideal ≈ 8", delta_color="off")
        s3.metric("Korelasi piksel (asli → cipher)", f"{adjacent_corr(arr):.3f} → {adjacent_corr(enc_img):.3f}")
        s4.metric("NPCR", f"{npcr:.2f}%", "ideal ≈ 99.6%", delta_color="off")
        s5.metric("UACI", f"{uaci:.2f}%", "ideal ≈ 33.46%", delta_color="off")
        st.caption(f"Waktu enkripsi: {t_enc:.2f}s untuk {len(raw) // 16} blok. NPCR/UACI dihitung dengan mengubah 1 bit "
                   "piksel pertama (CBC menyebarkan perubahan ke seluruh blok berikutnya).")
        fig, axs = plt.subplots(1, 2, figsize=(10, 2.8), sharey=False)
        axs[0].hist(arr.ravel(), bins=256, range=(0, 255), color="#8E0F1F")
        axs[0].set_title("Histogram gambar asli", fontsize=9)
        axs[1].hist(enc_img.ravel(), bins=256, range=(0, 255), color="#C8102E")
        axs[1].set_title("Histogram gambar terenkripsi", fontsize=9)
        for a in axs:
            a.tick_params(labelsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        buf = io.BytesIO()
        Image.fromarray(enc_img).save(buf, format="PNG")
        st.download_button("⬇️ Unduh gambar terenkripsi (PNG)", buf.getvalue(), "terenkripsi.png", "image/png")


if __name__ == "__main__":
    main()
