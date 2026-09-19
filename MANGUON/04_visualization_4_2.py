# -*- coding: utf-8 -*-
"""
Muc 4.2 - Truc quan hoa: Heatmap tuong quan, Pairplot theo loai noi dung, Elip tin cay 95%
De tai 20 - Danh gia tuong tac da chieu theo loai noi dung bai dang
Input : data_clean.csv (n = 460, dataset Facebook Metrics - UCI, da qua xu ly
        missing values, loai nhom Video va loc outlier Mahalanobis o buoc 3.1)
Output: 3 hinh PNG luu trong thu muc hien hanh
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / 'DULIEU'

sns.set_theme(style="whitegrid", font_scale=1.0)

# ---------------------------------------------------------------
# 1. DOC DU LIEU (data_clean.csv da duoc xu ly: loai Video, loc outlier)
# ---------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "data_clean.csv")

# Bien phan loai chinh thuc cua de tai: Type, 3 muc Photo / Status / Link
# (nhom Video da bi loai o buoc tien xu ly do co mau qua nho, n=7)
thu_tu_nhom = ["Photo", "Status", "Link"]
print("So quan sat moi nhom dinh dang:")
print(df["Type"].value_counts().reindex(thu_tu_nhom))

# ---------------------------------------------------------------
# 2. VECTOR BIEN p = 5 CHINH THUC (dung chung voi muc 3.4 - MANOVA)
# ---------------------------------------------------------------
bien_5 = ["like", "comment", "share",
          "Lifetime Post Total Reach", "Lifetime Post Total Impressions"]
ten_hien_thi = ["Like", "Comment", "Share", "Reach", "Impressions"]
df_plot = df[bien_5 + ["Type"]].rename(columns=dict(zip(bien_5, ten_hien_thi)))

# ---------------------------------------------------------------
# 3. HEATMAP MA TRAN TUONG QUAN (chung + theo tung nhom loai noi dung)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

corr_all = df_plot[ten_hien_thi].corr()
sns.heatmap(corr_all, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, cbar=True, ax=axes[0])
axes[0].set_title("Toan bo du lieu")

for ax, nhom in zip(axes[1:], thu_tu_nhom):
    sub = df_plot[df_plot["Type"] == nhom][ten_hien_thi]
    corr = sub.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, cbar=True, ax=ax)
    ax.set_title(f"Nhom: {nhom} (n={len(sub)})")

fig.suptitle("Ma tran tuong quan R giua 5 bien tuong tac theo tung loai noi dung", y=1.05, fontsize=14)
plt.tight_layout()
plt.savefig(DATA_DIR / "heatmap_tuongquan.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 4. PAIRPLOT THEO LOAI NOI DUNG
# ---------------------------------------------------------------
g = sns.pairplot(
    df_plot, vars=ten_hien_thi, hue="Type", hue_order=thu_tu_nhom,
    palette="Set2", diag_kind="kde", plot_kws={"alpha": 0.6, "s": 25},
    corner=True,
)
g.fig.suptitle("Pairplot 5 bien tuong tac phan theo loai noi dung", y=1.02, fontsize=14)
g.savefig(DATA_DIR / "pairplot_loainoidung.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 5. ELIP TIN CAY 95% (tren mat phang 2 bien dai dien: Like vs Comment)
# ---------------------------------------------------------------
def ve_elip_tin_cay(x, y, ax, n_std=2.4477, **kwargs):
    """
    Ve elip tin cay 95% cho phan phoi chuan 2 chieu (Np(mu, Sigma)).
    n_std duoc tinh chinh xac tu chi2.ppf(0.95, df=2) ** 0.5 ben duoi.
    """
    if x.size < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, facecolor="none", **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(np.mean(x), np.mean(y))
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


chi2_95 = stats.chi2.ppf(0.95, df=2)
n_std_95 = np.sqrt(chi2_95)

bien_x, bien_y = "Like", "Comment"  # co the doi sang cap bien khac de minh hoa
fig, ax = plt.subplots(figsize=(7.5, 6.5))
palette = dict(zip(thu_tu_nhom, sns.color_palette("Set2", 3)))

for nhom in thu_tu_nhom:
    sub = df_plot[df_plot["Type"] == nhom]
    ax.scatter(sub[bien_x], sub[bien_y], s=22, alpha=0.55, color=palette[nhom],
               label=f"{nhom} (n={len(sub)})")
    ve_elip_tin_cay(sub[bien_x].values, sub[bien_y].values, ax, n_std=n_std_95,
                     edgecolor=palette[nhom], linewidth=2.2, linestyle="--")
    ax.scatter(sub[bien_x].mean(), sub[bien_y].mean(), color=palette[nhom],
               marker="X", s=140, edgecolor="black")

ax.set_xlabel(bien_x)
ax.set_ylabel(bien_y)
ax.set_title(f"Elip tin cay 95% cho vector trung binh ({bien_x}, {bien_y}) theo loai noi dung")
ax.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "elip_tincay_95.png", dpi=150, bbox_inches="tight")
plt.close()

print("Da xuat 3 hinh: heatmap_tuongquan.png, pairplot_loainoidung.png, elip_tincay_95.png")
