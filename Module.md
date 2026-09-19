# HƯỚNG DẪN CHẠY MÃ NGUỒN PYTHON — ĐỀ TÀI 20

## 1. Yêu cầu môi trường

- Python phiên bản 3.9 trở lên (đã kiểm thử trên Python 3.12.3)

## 2. Các thư viện (module) cần cài đặt

| Thư viện | Phiên bản đã kiểm thử | Vai trò |
|---|---|---|
| pandas | 2.x trở lên | Đọc, xử lý, thao tác dữ liệu dạng bảng |
| numpy | 1.x trở lên | Tính toán ma trận, đại số tuyến tính (hiệp phương sai, Mahalanobis) |
| scipy | 1.x trở lên | Phân phối chi-bình phương, phân phối chuẩn (dùng trong kiểm định Mardia, ngưỡng outlier) |
| statsmodels | 0.14 trở lên | Chạy mô hình MANOVA (`statsmodels.multivariate.manova`) |
| pingouin | 0.6 trở lên | Kiểm định Henze-Zirkler, Box's M, ANOVA, hậu định Tukey HSD |
| matplotlib | 3.x trở lên | Vẽ biểu đồ (heatmap, pairplot, elip tin cậy) |
| seaborn | 0.12 trở lên | Vẽ biểu đồ thống kê (dựa trên matplotlib) |

### Cài đặt nhanh (1 dòng lệnh)

```bash
pip install pandas numpy scipy statsmodels pingouin matplotlib seaborn
```

> Nếu dùng Google Colab: chỉ cần cài thêm `pingouin` (các thư viện còn lại đã có sẵn):
> ```python
> !pip install pingouin
> ```

## 3. Cấu trúc thư mục bắt buộc

Các file code dùng đường dẫn tương đối dựa trên vị trí thư mục, **bắt buộc giữ nguyên cấu trúc** sau (đặt 2 thư mục `DULIEU` và `MANGUON` cùng cấp với nhau):

```
├── DULIEU/
│   ├── dataset.csv          (dữ liệu gốc)
│   └── data_clean.csv       (sinh ra sau khi chạy file 01)
└── MANGUON/
    ├── 01_eda_cleaning.py
    ├── 02_code_3_2_3_3.py
    ├── 03_code_manova_3_4.py
    └── 04_visualization_4_2.py
```

## 4. Thứ tự chạy và chức năng từng file

Phải chạy **đúng thứ tự 01 → 02 → 03 → 04**, vì file 01 sinh ra `data_clean.csv` mà 3 file sau đều phụ thuộc vào.

### 01_eda_cleaning.py
**Chức năng:** Tiền xử lý dữ liệu thô (tương ứng Mục 3.1 của báo cáo).
- Đọc `dataset.csv` (500 bài đăng gốc, 20 cột).
- Loại cột thừa (`Unnamed: 0` nếu có).
- Xử lý giá trị khuyết thiếu: cột `Paid` điền theo giá trị xuất hiện nhiều nhất (mode), cột `like`/`share` điền theo trung vị (median) — chọn trung vị vì hai biến này có phân phối lệch, tránh bị outlier kéo lệch giá trị điền.
- Xóa các dòng trùng lặp hoàn toàn.
- Chuẩn hóa kiểu dữ liệu số nguyên cho `Paid`, `like`, `share`.
- Loại nhóm định dạng "Video" (chỉ có 7 quan sát — không đủ để ước lượng ổn định ma trận hiệp phương sai riêng của nhóm khi có 5 biến phân tích).
- Tính khoảng cách Mahalanobis D² của từng quan sát so với vector trung bình chung (trên 5 biến phân tích chính), loại các quan sát vượt ngưỡng phân vị 97,5% của phân phối chi-bình phương (outlier đa biến).
- Xuất kết quả ra `DULIEU/data_clean.csv` (460 quan sát, 3 nhóm Photo/Status/Link).

### 02_code_3_2_3_3.py
**Chức năng:** Tính đặc trưng mẫu và kiểm định giả định chuẩn đa biến (tương ứng Mục 3.2 và 3.3).
- Đọc `data_clean.csv`.
- Với từng nhóm (Photo, Status, Link): tính vector trung bình mẫu X̄, ma trận hiệp phương sai mẫu S, ma trận tương quan mẫu R, và định thức |S| (phương sai tổng quát).
- Cài đặt thủ công kiểm định Mardia (đo độ lệch và độ nhọn đa biến) vì thư viện pingouin không có sẵn hàm này.
- Gọi hàm có sẵn của pingouin để chạy kiểm định Henze-Zirkler (kiểm định chuẩn đa biến).
- In ra toàn bộ kết quả để đối chiếu với số liệu trình bày trong báo cáo.

### 03_code_manova_3_4.py
**Chức năng:** Thực thi kiểm định MANOVA chính (tương ứng Mục 3.4).
- Đọc `data_clean.csv`.
- Thống kê mô tả trung bình/độ lệch chuẩn theo từng nhóm (đối chiếu chéo với file 02).
- Chạy kiểm định Box's M kiểm tra giả định đồng nhất ma trận hiệp phương sai giữa 3 nhóm.
- Chạy MANOVA một nhân tố trên 5 biến phụ thuộc (comment, like, share, Reach, Impressions), xuất đầy đủ 4 tiêu chuẩn kiểm định (Wilks' Lambda, Pillai's Trace, Hotelling-Lawley Trace, Roy's Greatest Root).
- Kết luận giả thuyết H0 dựa trên Pillai's Trace (ưu tiên vì robust hơn khi Box's M vi phạm).
- Chạy ANOVA riêng từng biến và hậu định Tukey HSD để xác định cụ thể cặp nhóm nào khác biệt ở biến nào — phục vụ phần biện luận Mục 4.1 và 4.3.

### 04_visualization_4_2.py
**Chức năng:** Trực quan hóa dữ liệu đa biến (tương ứng Mục 4.2).
- Đọc `data_clean.csv`.
- Vẽ ma trận tương quan dạng heatmap: 1 biểu đồ tổng thể + 3 biểu đồ riêng cho từng nhóm.
- Vẽ pairplot (ma trận biểu đồ phân tán từng cặp biến, phân biệt màu theo nhóm).
- Vẽ elip tin cậy 95% cho vector trung bình (minh họa trên cặp biến Like–Comment).
- Xuất 3 file ảnh PNG: `heatmap_tuongquan.png`, `pairplot_loainoidung.png`, `elip_tincay_95.png` (lưu vào thư mục `DULIEU/`).