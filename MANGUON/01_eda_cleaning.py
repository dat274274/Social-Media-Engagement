from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'DULIEU'
INPUT_CSV = DATA_DIR / 'dataset.csv'
OUTPUT_CSV = DATA_DIR / 'data_clean.csv'

# 1. Doc du lieu goc
df = pd.read_csv(INPUT_CSV)
print(f"Kich thuoc ban dau: {df.shape}")

# 2. Xoa cot thua neu co
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 3. Xu ly gia tri bi thieu (Missing values)
df['Paid'] = df['Paid'].fillna(df['Paid'].mode()[0])
df['like'] = df['like'].fillna(df['like'].median())
df['share'] = df['share'].fillna(df['share'].median())

# 4. Xoa dong trung lap
df = df.drop_duplicates()

# 5. Chuan hoa kieu du lieu so
numeric_cols = ['Paid', 'like', 'share']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

print(f"Sau xu ly missing/trung lap: {df.shape}")
print(df['Type'].value_counts())

# 6. Loai nhom dinh dang co co mau qua nho (Video: 7 quan sat)
# Voi p = 5 bien, moi nhom can toi thieu p+1 = 6 quan sat de uoc luong on dinh
# ma tran hiep phuong sai; Box's M can nhieu hon the moi cho ket qua tin cay.
df = df[df['Type'] != 'Video'].reset_index(drop=True)
print(f"\nSau loai nhom Video: {df.shape}")

# 7. Phat hien va loai outlier da bien bang khoang cach Mahalanobis
# Tinh tren 5 bien phan tich chinh thuc cua de tai
DV = ['comment', 'like', 'share', 'Lifetime Post Total Reach', 'Lifetime Post Total Impressions']
X = df[DV].values
mean_vec = X.mean(axis=0)
cov_mat = np.cov(X, rowvar=False)
inv_cov = np.linalg.inv(cov_mat)
diff = X - mean_vec
mahal_dist = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)

threshold = chi2.ppf(0.975, df=len(DV))  # nguong chi-binh phuong, muc y nghia 2,5%
n_outlier = (mahal_dist > threshold).sum()
print(f"\nNguong Mahalanobis D^2 (chi2, df={len(DV)}, 97.5%) = {threshold:.2f}")
print(f"So outlier phat hien va loai bo: {n_outlier}/{len(df)}")

df = df[mahal_dist <= threshold].reset_index(drop=True)
print(f"\nCo mau chinh thuc sau lam sach: {df.shape}")
print(df['Type'].value_counts())

# 8. Luu ket qua ra tep moi
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDa lam sach va xuat file thanh cong ra '{OUTPUT_CSV.name}' tai '{OUTPUT_CSV.parent}'!")
