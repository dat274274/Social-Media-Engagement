"""
De tai 20 - Muc 3.2: Dac trung mau (X-bar, S, R, |S|)
            Muc 3.3: Kiem dinh chuan da bien (Mardia + Henze-Zirkler)
Input: data_clean.csv (n = 460, da qua xu ly o muc 3.1)
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

DATA_DIR = Path(__file__).resolve().parent.parent / 'DULIEU'

DV = [
    "comment",
    "like",
    "share",
    "Lifetime Post Total Reach",
    "Lifetime Post Total Impressions",
]
GROUP = "Type"
NHOM = ["Photo", "Status", "Link"]

df = pd.read_csv(DATA_DIR / "data_clean.csv")


def mardia_test(X):
    """
    Kiem dinh Mardia ve do lech (skewness) va do nhon (kurtosis) da bien.
    Tra ve thong ke va p-value cho ca hai thanh phan.
    """
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = np.cov(Xc, rowvar=False, bias=True)  # uoc luong MLE (chia cho n)
    S_inv = np.linalg.inv(S)
    A = Xc @ S_inv @ Xc.T  # ma tran khoang cach Mahalanobis cheo cap

    # Do lech da bien (multivariate skewness)
    b1p = np.sum(A ** 3) / n ** 2
    df_skew = p * (p + 1) * (p + 2) / 6
    stat_skew = n * b1p / 6
    p_skew = 1 - stats.chi2.cdf(stat_skew, df_skew)

    # Do nhon da bien (multivariate kurtosis)
    b2p = np.sum(np.diag(A) ** 2) / n
    mean_k = p * (p + 2)
    var_k = 8 * p * (p + 2) / n
    z_kurt = (b2p - mean_k) / np.sqrt(var_k)
    p_kurt = 2 * (1 - stats.norm.cdf(abs(z_kurt)))

    return {
        "skew_stat": stat_skew, "skew_df": df_skew, "skew_p": p_skew,
        "kurt_z": z_kurt, "kurt_p": p_kurt,
    }


# ============================================================
# MUC 3.2 - DAC TRUNG MAU THEO TUNG NHOM
# ============================================================
print("=" * 70)
print("MUC 3.2 - DAC TRUNG MAU (X-bar, S, R, |S|) THEO TUNG NHOM")
print("=" * 70)

for nhom in NHOM:
    sub = df[df[GROUP] == nhom][DV].values
    n = len(sub)
    xbar = sub.mean(axis=0)
    S = np.cov(sub, rowvar=False)
    R = np.corrcoef(sub, rowvar=False)
    detS = np.linalg.det(S)

    print(f"\n--- Nhom {nhom} (n={n}) ---")
    print("Vector trung binh X-bar:")
    for ten, gia_tri in zip(DV, xbar):
        print(f"  {ten:35s} = {gia_tri:,.2f}")
    print(f"Dinh thuc ma tran hiep phuong sai |S| = {detS:.4e}")
    print("Ma tran tuong quan mau R:")
    print(pd.DataFrame(R, index=DV, columns=DV).round(2))

# ============================================================
# MUC 3.3 - KIEM DINH CHUAN DA BIEN
# ============================================================
print("\n" + "=" * 70)
print("MUC 3.3 - KIEM DINH CHUAN DA BIEN (Mardia + Henze-Zirkler)")
print("=" * 70)

for nhom in NHOM:
    sub = df[df[GROUP] == nhom][DV].values
    print(f"\n--- Nhom {nhom} (n={len(sub)}) ---")

    m = mardia_test(sub)
    print(f"Mardia - do lech (skewness): thong ke = {m['skew_stat']:.3f}, "
          f"df = {m['skew_df']:.0f}, p-value = {m['skew_p']:.5f}")
    print(f"Mardia - do nhon (kurtosis): z = {m['kurt_z']:.3f}, "
          f"p-value = {m['kurt_p']:.5f}")

    hz, pval, normal = pg.multivariate_normality(sub, alpha=0.05)
    print(f"Henze-Zirkler: HZ = {hz:.4f}, p-value = {pval:.5f}, "
          f"chuan da bien = {normal}")
