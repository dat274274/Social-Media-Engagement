"""
De tai 20 - Phan 3.4: Thuc thi kiem dinh MANOVA tren Python
Input: data_clean.csv (n = 460, da duoc xu ly hoan chinh o muc 3.1:
       loai missing/trung lap, loai nhom Video, loc outlier Mahalanobis)
"""

from pathlib import Path

import pandas as pd
from statsmodels.multivariate.manova import MANOVA
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

# ============================================================
# 1. DOC DU LIEU
# ============================================================
df = pd.read_csv(DATA_DIR / "data_clean.csv")
print("Phan bo theo loai noi dung:\n", df[GROUP].value_counts(), "\n")

# ============================================================
# 2. THONG KE MO TA THEO NHOM (doi chieu voi 3.2)
# ============================================================
desc = df.groupby(GROUP)[DV].agg(["mean", "std"]).round(2)
print("Thong ke mo ta theo Type:\n", desc, "\n")

# ============================================================
# 3. KIEM DINH BOX'S M (dong nhat ma tran hiep phuong sai)
# ============================================================
box_m = pg.box_m(data=df, dvs=DV, group=GROUP)
print("Kiem dinh Box's M:\n", box_m, "\n")
print("=> Neu Box's M vi pham (p < 0.05): uu tien doc ket qua tu Pillai's Trace,")
print("   day la tieu chuan robust hon Wilks' Lambda khi vi pham gia dinh dong nhat Sigma.\n")

# ============================================================
# 4. CHAY MANOVA 1 NHAN TO
# ============================================================
formula = " + ".join([f"Q('{v}')" for v in DV]) + f" ~ {GROUP}"
maov = MANOVA.from_formula(formula, data=df)
result = maov.mv_test()
stats_table = result.results[GROUP]["stat"]
print("Bang 4 tieu chuan kiem dinh da bien:\n", stats_table, "\n")

pillai_val = stats_table.loc["Pillai's trace", "Value"]
pillai_F = stats_table.loc["Pillai's trace", "F Value"]
pillai_p = stats_table.loc["Pillai's trace", "Pr > F"]

alpha = 0.05
print(f"--- Ket luan (alpha = {alpha}) ---")
print(f"Pillai's Trace = {pillai_val:.4f} | F = {pillai_F:.4f} | p-value = {pillai_p:.6f}")
if pillai_p < alpha:
    print("=> Bac bo H0: co su khac biet co y nghia thong ke ve vector trung binh")
    print("   cac chi so tuong tac giua 3 loai noi dung (Photo/Status/Link).")
else:
    print("=> Chua du co so bac bo H0.")

# ============================================================
# 5. ANOVA TUNG BIEN + POST-HOC TUKEY (bo tro cho 4.1/4.3)
# ============================================================
print("\n--- ANOVA tung bien ---")
for var in DV:
    aov = pg.anova(data=df, dv=var, between=GROUP, detailed=False)
    print(f"{var:35s} F={aov['F'].values[0]:.3f}  p={aov['p_unc'].values[0]:.5f}")

print("\n--- Post-hoc Tukey HSD tung bien ---")
for var in DV:
    tukey = pg.pairwise_tukey(data=df, dv=var, between=GROUP)
    print(f"\n[{var}]")
    print(tukey[["A", "B", "diff", "p_tukey"]])
