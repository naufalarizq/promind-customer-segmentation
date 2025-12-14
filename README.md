<a name="top"></a>
<h1 align="center"><img src="logo/blended purple blue.png" alt="PROMIND Logo" width="300"></h1>

[Sekilas Tentang](#sekilas-tentang) | [Instalasi](#instalasi) | [Konfigurasi](#konfigurasi) | [Cara Pemakaian](#cara-pemakaian) | [Pembahasan](#pembahasan) | [Referensi](#referensi)
:---:|:---:|:---:|:---:|:---:|:---:

[`^ Kembali ke atas ^`](#top)

---

# Sekilas Tentang

**PROMIND** (Promotion Intelligence) adalah sistem cerdas berbasis RFM (Recency, Frequency, Monetary) yang menggunakan machine learning untuk segmentasi pelanggan dan rekomendasi strategi pemasaran yang dipersonalisasi dalam industri ritel. Sistem ini dirancang untuk membantu perusahaan ritel memahami perilaku pelanggan secara mendalam dan merancang kampanye promosi yang lebih efektif, efisien, dan berdampak pada peningkatan loyalitas pelanggan.

## Latar Belakang Masalah

Dalam era digital, perusahaan ritel menghadapi tantangan signifikan:
- **Pemborosan anggaran promosi**: Strategi pemasaran yang belum mempertimbangkan perilaku pelanggan spesifik
- **Loyalitas pelanggan yang menurun**: Kurangnya personalisasi dalam komunikasi dan penawaran
- **Inefisiensi segmentasi**: Pengelompokan pelanggan hanya berdasarkan demografi tanpa mempertimbangkan perilaku transaksi
- **Kehilangan peluang bisnis**: Sulitnya mengidentifikasi pelanggan bernilai tinggi dan risiko churn

## Solusi PROMIND

PROMIND menawarkan pendekatan terintegrasi yang menggabungkan:

1. **RFM Analysis** - Analisis mendalam berdasarkan:
   - **Recency**: Berapa lama sejak transaksi terakhir (dalam hari)
   - **Frequency**: Berapa banyak transaksi yang dilakukan
   - **Monetary**: Berapa nilai total transaksi pelanggan

2. **Machine Learning Clustering** - Menggunakan K-Means untuk segmentasi otomatis dengan:
   - Penentuan jumlah cluster optimal via Elbow Method & Silhouette Score
   - Stabilitas model melalui random state sensitivity analysis
   - Visualisasi PCA untuk interpretasi cluster

3. **Business Intelligence** - Insight bisnis actionable:
   - Profiling setiap segmen (Champions, Loyal Customers, Potential Loyalists, At Risk)
   - Value Efficiency Index untuk identifikasi high-value segments
   - Priority scoring untuk targeting pelanggan

4. **Personalized Marketing Playbook** - Rekomendasi taktik per segmen:
   - Persona definition dan business objectives
   - Primary offers dan channel strategy
   - Campaign cadence dan KPI monitoring

## Fitur Utama

✅ **Data Processing Pipeline** - Automated cleaning dan RFM computation  
✅ **Advanced Clustering** - K-Means dengan validasi metrik (Silhouette, Inertia)  
✅ **Business Profiling** - Complete segment analysis dan visualization  
✅ **Marketing Intelligence** - Rule-based recommendation engine  
✅ **Export & Visualization** - CSV reports, JSON blueprints, dan charts  
✅ **Reproducible Analysis** - Full source code dan dokumentasi lengkap  

## Tech Stack

| Komponen | Tools/Library |
|----------|---------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn (K-Means) |
| **Interactive Environment** | Jupyter Notebook |
| **Version Control** | Git/GitHub |

---

# Instalasi
[`^ Kembali ke atas ^`](#top)

## Kebutuhan Sistem

### Minimum Requirements:
- **OS**: Windows 10+, macOS 10.14+, atau Linux (Ubuntu 18.04+)
- **Python**: 3.8 atau lebih baru
- **RAM**: 2GB minimum (4GB+ direkomendasikan)
- **Storage**: 500MB untuk library + 1GB untuk dataset
- **Internet**: Koneksi stabil untuk download dataset

### Prerequisites:
- Python dan pip terinstall
- Git (opsional, untuk clone repository)
- Jupyter Notebook atau VS Code dengan Python extension
- Browser modern (Chrome, Firefox, Safari, Edge)

## Proses Instalasi

### 1. Persiapan Environment

#### Windows:
```bash
# Buka Command Prompt atau PowerShell
# Navigate ke folder yang diinginkan
cd C:\Users\YourUsername\Documents

# Buat virtual environment
python -m venv promind_env

# Aktivasi virtual environment
promind_env\Scripts\activate
```

#### macOS / Linux:
```bash
# Buka Terminal
# Navigate ke folder yang diinginkan
cd ~/Documents

# Buat virtual environment
python3 -m venv promind_env

# Aktivasi virtual environment
source promind_env/bin/activate
```

### 2. Clone Repository (Opsional)

Jika menggunakan GitHub:
```bash
git clone https://github.com/your-username/promind-customer-segmentation.git
cd promind-customer-segmentation
```

Atau download ZIP langsung dari GitHub dan ekstrak.

### 3. Install Dependencies

Buat file `requirements.txt` dengan konten:
```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
scipy==1.11.1
jupyter==1.0.0
notebook==7.0.0
plotly==5.16.1
openpyxl==3.1.2
```

Install semua dependencies:
```bash
pip install -r requirements.txt
```

Atau install secara individual:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook plotly
```

### 4. Download Dataset

Dataset Online Retail II dari Kaggle:

**Opsi A: Download Manual**
1. Kunjungi: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
2. Klik tombol "Download" (membutuhkan akun Kaggle gratis)
3. Ekstrak file ke folder `dataset/`

**Opsi B: Menggunakan Kaggle CLI**
```bash
# Install kaggle CLI
pip install kaggle

# Setup API credentials (buat di https://www.kaggle.com/settings/account)
# Windows: copy api token ke C:\Users\YourUsername\.kaggle\kaggle.json
# macOS/Linux: copy api token ke ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d mashlyn/online-retail-ii-uci
unzip online-retail-ii-uci.zip -d dataset/
```

**Struktur folder yang diharapkan:**
```
promind-customer-segmentation/
├── dataset/
│   ├── online_retail_II.csv
│   └── archive/
├── output/
├── logo/
├── PROMIND_RFM__Customer_Segmentation.ipynb
├── requirements.txt
├── README.md
└── agent_rules.txt
```

### 5. Jalankan Jupyter Notebook

```bash
# Pastikan virtual environment masih aktif
jupyter notebook

# Browser akan otomatis terbuka ke localhost:8888
# Buka file: PROMIND_RFM__Customer_Segmentation.ipynb
```

Atau untuk VS Code:
```bash
# Buka folder di VS Code
code .

# Install Python extension jika belum ada
# Pilih interpreter: promind_env
# Buka notebook file
```

### 6. Verifikasi Instalasi

Jalankan cell pertama dari notebook untuk memverifikasi:
```python
# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

print("✓ Semua library berhasil diimport!")
```

Jika tidak ada error, instalasi selesai.

---

# Konfigurasi
[`^ Kembali ke atas ^`](#top)

## Konfigurasi Dataset

### 1. Path Configuration

Pastikan path dataset benar di notebook cell #3:
```python
# DATA LOADING
df = pd.read_csv("dataset/online_retail_II.csv")
```

Jika menggunakan path absolut:
```python
import os
dataset_path = os.path.join(os.getcwd(), "dataset", "online_retail_II.csv")
df = pd.read_csv(dataset_path)
```

### 2. Filtering Configuration

Notebook sudah dikonfigurasi untuk:
- **Filter geografis**: United Kingdom (baris ke-4 "DATA CLEANING FOR RFM ANALYSIS")
- **Validasi transaksi**: Quantity > 0 dan UnitPrice > 0
- **Handling missing values**: Buang baris tanpa CustomerID

Untuk mengubah filter (misal ke country lain):
```python
# Ubah line ini:
df_clean = df[df["Country"] == "Netherlands"].copy()  # Ganti "United Kingdom"
```

### 3. RFM Parameters Configuration

**Snapshot Date** (otomatis menggunakan tanggal terakhir + 1 hari):
```python
# Line di "RFM FEATURE ENGINEERING" cell
snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
```

Untuk menggunakan tanggal spesifik:
```python
snapshot_date = pd.Timestamp("2024-12-14")
```

### 4. Clustering Configuration

**Jumlah Cluster (K)** - Ubah nilai OPTIMAL_K:
```python
# Default: K = 4 (sudah optimal)
OPTIMAL_K = 4

# Untuk test K yang berbeda, ubah di cell "ELBOW METHOD":
for k in range(2, 11):  # Test K dari 2 sampai 10
    # Silhouette score akan menunjukkan K mana yang optimal
```

**Random State** - untuk reproducibility:
```python
# Semua KMeans sudah menggunakan random_state=42
kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,  # Ganti untuk hasil berbeda
    n_init=10
)
```

### 5. Output Configuration

**Path penyimpanan file output** (default: folder `output/`):
```python
# File CSV:
final_csv_path = f"output/PROMIND_RFM_Segmentation_{timestamp}.csv"
priority_csv_path = f"output/PROMIND_PriorityCustomers_Top50_{timestamp}.csv"
rec_csv_path = f"output/PROMIND_MarketingRecommendations_{timestamp}.csv"

# Untuk ganti folder (misal ke folder "results/"):
final_csv_path = f"results/PROMIND_RFM_Segmentation_{timestamp}.csv"
```

**Resolusi gambar** (charts default: 300 dpi):
```python
# Current setting:
plt.savefig("output/01_rfm_distribution.png", dpi=300, bbox_inches='tight')

# Untuk ganti resolusi:
plt.savefig("output/01_rfm_distribution.png", dpi=150, bbox_inches='tight')  # Draft
plt.savefig("output/01_rfm_distribution.png", dpi=600, bbox_inches='tight')  # Publication
```

### 6. Marketing Playbook Configuration

Edit strategi pemasaran per segmen di cell "MARKETING RECOMMENDATION ENGINE":

```python
PROMIND_PLAYBOOK = {
    "Champions": {
        "persona": "High value, high engagement",
        "objective": "Retain + maximize LTV",
        "primary_offer": "VIP perks / early access",
        "tactics": [
            "VIP tier & priority support",
            "Early access launches",
            "Cross-sell premium bundle",
            "Referral program"
        ],
        "channel": ["Email", "App Push", "WhatsApp", "Loyalty"],
        "cadence": "Weekly (personalized)"
    },
    # ... (edit segmen lain sesuai kebutuhan bisnis)
}
```

### 7. Priority Score Configuration

Bobot scoring untuk prioritas customer (default):
```python
# Weight untuk scoring:
W_MON = 0.5    # Monetary (50%)
W_FREQ = 0.3   # Frequency (30%)
W_REC = 0.2    # Recency (20%)

# Untuk ganti prioritas (misal fokus frequency):
W_MON = 0.3
W_FREQ = 0.5   # Naik jadi prioritas utama
W_REC = 0.2
```

---

# Cara Pemakaian
[`^ Kembali ke ata ^`](#top)

## Workflow Tahap Demi Tahap

### Fase 1: Data Preparation (Cell 1-5)

**Step 1: Import Library**
- Jalankan cell "IMPORT LIBRARY"
- Hasil: Semua library terimport, warning diabaikan

**Step 2: Load Dataset**
- Jalankan cell "DATA LOADING, COLUMN STANDARDIZATION & INITIAL AUDIT"
- Output:
  - Dataset shape: berapa baris dan kolom
  - Sample 5 baris pertama
  - Info kolom dan tipe data
  - Summary missing values dan anomali

**Step 3: Data Cleaning**
- Jalankan cell "DATA CLEANING FOR RFM ANALYSIS"
- Filter dilakukan:
  - Country == "United Kingdom"
  - Quantity > 0 dan UnitPrice > 0
  - CustomerID tidak null
- Output: Dataset setelah cleaning dengan sanity check

**Step 4: RFM Feature Engineering**
- Jalankan cell "RFM FEATURE ENGINEERING"
- Output:
  - RFM dataframe dengan 3 kolom: Recency, Frequency, Monetary
  - Summary statistics: min, max, mean, median, std

**Step 5: Classic RFM Baseline**
- Jalankan cell "RFM SCORE CLASSIC"
- Output:
  - RFM scores (quantile-based 1-5)
  - Classic segment mapping (Champions, Loyal, etc.)
  - Distribution per segmen

### Fase 2: Feature Transformation & EDA (Cell 6-8)

**Step 6: RFM Distribution Analysis**
- Jalankan cell "RFM DISTRIBUTION & SKEWNESS ANALYSIS"
- Chart:
  - 3 histogram: Recency, Frequency, Monetary
  - Skewness values (menunjukkan distribusi data)
- Interpretasi: Data RFM biasanya right-skewed, perlu log transformation

**Step 7: Log Transformation & Scaling**
- Jalankan cell "LOG TRANSFORMATION & FEATURE SCALING"
- Proses:
  - Log1p transformation untuk mengurangi skew
  - StandardScaler untuk normalisasi (mean=0, std=1)
- Output:
  - Scaled dataframe ready untuk clustering
  - Verification: mean ≈ 0, std ≈ 1

**Step 8: Elbow Method & Silhouette Analysis**
- Jalankan cell "ELBOW METHOD & SILHOUETTE SCORE"
- Chart:
  - Elbow plot: inertia vs K
  - Silhouette plot: silhouette score vs K
- Interpretasi:
  - Cari "elbow" point di inertia plot
  - Silhouette score tertinggi menunjukkan K optimal
  - Hasil: K=4 adalah optimal

### Fase 3: Clustering & Validation (Cell 9-13)

**Step 9: K-Means Stability Check**
- Jalankan cell "K-MEANS STABILITY CHECK"
- Validasi:
  - Run KMeans 10 kali dengan different random states
  - Hitung silhouette score per run
  - Coefficient of variation < 10% = STABLE
- Output:
  - Stability dataframe
  - Chart: silhouette stability across random states

**Step 10: K-Means Final Clustering**
- Jalankan cell "K-MEANS ROBUSTNESS & METHOD JUSTIFICATION"
- Output:
  - Final KMeans model dengan K=4
  - Silhouette score final
  - Robustness check across 10 random states
  - Method justification (mengapa K-Means dipilih)

**Step 11: Create RFM Result**
- Jalankan cell "CREATE RFM_RESULT WITH CLUSTER LABELS"
- Output:
  - rfm_result dataframe dengan kolom Cluster
  - Cluster distribution (berapa customer per cluster)
  - Complete cluster profile (min/mean/max/median/std)

**Step 12: 3D Visualization**
- Jalankan cell "3D VISUALISASI RFM"
- Visualisasi:
  - Plotly 3D scatter: Recency vs Frequency vs Monetary
  - Colored by Cluster
  - Interactive hover: CustomerID
- Output: chart_04_3d_rfm_scatter.png

**Step 13: PCA Projection**
- Jalankan cell "K-MEANS CLUSTER VISUALIZATION (PCA PROJECTION)"
- Visualisasi:
  - PCA 2D projection dari 3 dimensi RFM
  - Explained variance ratio per PC
  - Scatter plot colored by cluster
- Output: chart_10_pca_cluster_visualization.png

### Fase 4: Business Interpretation (Cell 14-16)

**Step 14: Segment Naming & Profiling**
- Jalankan cell "CLUSTER INTERPRETATION & BUSINESS SEGMENT NAMING"
- Proses:
  - Ranking cluster berdasarkan Recency, Frequency, Monetary
  - Mapping cluster → Segment name (Champions, Loyal Customers, etc.)
  - Profiling setiap segment (mean RFM)
- Output:
  - Segment distribution
  - RFM summary per segment

**Step 15: Segment Visualization**
- Jalankan cell "RFM & SEGMENT VISUALIZATION"
- Charts:
  - Countplot: Customer distribution per segment
  - Boxplot: Monetary distribution per segment (log scale)
  - Scatterplot: Recency vs Frequency (colored by segment)
  - Scatterplot: Frequency vs Monetary (colored by segment)
- Output: 4 files (05-08_*.png)

**Step 16: Business Metrics & Insights**
- Jalankan cell dengan business summary
- Output:
  - Total customers & total revenue
  - Segment share (%)
  - Segment RFM means
  - Value Efficiency Index (Revenue% / Customer%)
  - Business role classification (Revenue Engine vs Growth Driver)

### Fase 5: Marketing Strategy & Export (Cell 17-20)

**Step 17: Marketing Recommendation Engine**
- Jalankan cell "MARKETING RECOMMENDATION ENGINE"
- Output:
  - Persona & objective per segment
  - Primary offers dan tactics
  - Channel strategy dan campaign cadence
  - Customer priority scoring (0-1)
  - Priority tier distribution (Low/Medium/High/Critical)

**Step 18: PROMIND Finalization**
- Jalankan cell "PROMIND FINALIZATION"
- Output:
  - Executive summary (text)
  - Pie chart: Segment share percentage
  - CSV export: PROMIND_RFM_Segmentation_{timestamp}.csv
  - CSV export: PROMIND_PriorityCustomers_Top50_{timestamp}.csv
  - CSV export: PROMIND_MarketingRecommendations_{timestamp}.csv
  - JSON export: PROMIND_DeploymentBlueprint_{timestamp}.json

**Step 19: Outlier Analysis**
- Jalankan cell "ANALISIS DAMPAK OUTLIER"
- Output:
  - Top 1% monetary threshold
  - Distribution analysis
  - Mean vs Median comparison (outlier effect)
  - Pie chart: Top 1% distribution per cluster
  - Interpretation: whale effect pada data retail

## Interpreting Results

### 1. Cluster Profiles

Contoh hasil clustering (4 cluster):

| Segment | Recency | Frequency | Monetary | Profile |
|---------|---------|-----------|----------|---------|
| Champions | 10 hari | 50 transaksi | £5000 | High value, very active |
| Loyal Customers | 20 hari | 30 transaksi | £2000 | Stable, consistent |
| Potential Loyalists | 8 hari | 5 transaksi | £500 | Recent buyers, growth potential |
| At Risk | 100 hari | 2 transaksi | £200 | Inactive, churn risk |

### 2. Priority Score Interpretation

Priority score (0-1) digunakan untuk ranking customer:
- **0.8-1.0**: Critical - immediate action needed
- **0.6-0.8**: High - prioritize for campaigns
- **0.4-0.6**: Medium - standard marketing
- **0.0-0.4**: Low - nurture/retention only

### 3. Marketing Actions per Segment

**Champions (0.5 weight Monetary, 0.3 Frequency, 0.2 Recency)**:
- Goal: Retain & maximize CLV
- Offer: VIP perks, early access, premium bundle
- Channel: Email, App Push, WhatsApp
- Cadence: Weekly personalized

**Loyal Customers**:
- Goal: Increase basket size & frequency
- Offer: Points booster, cross-sell, free shipping
- Channel: Email, App Push
- Cadence: Bi-weekly

**Potential Loyalists**:
- Goal: Convert to loyal customer
- Offer: Second-purchase incentive
- Channel: Email, App Push, WhatsApp
- Cadence: 2-3 touches in 14 days

**At Risk**:
- Goal: Reactivate & prevent churn
- Offer: Win-back discount, strong incentive
- Channel: Email, WhatsApp, SMS
- Cadence: 2 touches in 10 days

---

# Pembahasan
[`^ Kembali ke atas ^`](#top)

## Metodologi RFM

### Konsep Dasar

RFM adalah framework yang mengukur customer value berdasarkan 3 dimensi perilaku:

1. **Recency (R)**: Berapa lama sejak pembelian terakhir?
   - Pelanggan dengan Recency rendah (baru beli) = engagement tinggi
   - Pelanggan dengan Recency tinggi (lama tidak beli) = churn risk
   - Formula: `(Snapshot Date - Last Purchase Date).days`

2. **Frequency (F)**: Berapa sering pelanggan membeli?
   - Frequency tinggi = loyal customer
   - Frequency rendah = rare buyer atau new customer
   - Formula: `COUNT(DISTINCT InvoiceNo)`

3. **Monetary (M)**: Berapa nilai total belanja pelanggan?
   - Monetary tinggi = high-value customer
   - Monetary rendah = small spender
   - Formula: `SUM(Quantity * UnitPrice)`

### Data Preparation Pipeline

```
Raw Data
   ↓
[Filter & Clean]
   - Country filter (UK only)
   - Remove invalid transactions (Qty/Price ≤ 0)
   - Handle missing CustomerID
   ↓
[Feature Engineering]
   - Calculate RFM metrics
   - Create Revenue column (Qty × Price)
   ↓
[Transform]
   - Log1p transformation (reduce skewness)
   - StandardScaler normalization (mean=0, std=1)
   ↓
[Clustering Ready Data]
   - 3 scaled features: R_scaled, F_scaled, M_scaled
   - Ready untuk KMeans
```

### Skewness & Log Transformation

RFM data umumnya right-skewed (banyak small spenders, few big spenders):
- Log transformation mengurangi skew
- Mencegah dominasi outliers dalam clustering
- Meningkatkan cluster separation

Contoh:
```
Original Recency: [1, 5, 10, 50, 365] (highly skewed)
Log Recency: [0, 1.6, 2.3, 3.9, 5.9] (more balanced)
```

## Machine Learning: K-Means Clustering

### Algoritma K-Means

K-Means adalah unsupervised learning algorithm yang:
1. Initialize K random centroids
2. Assign points ke nearest centroid
3. Update centroids berdasarkan mean points
4. Repeat hingga converge

**Why K-Means untuk RFM?**
- Interpretasi cluster yang jelas (centroid = prototype segment)
- Scalable untuk dataset besar
- Cocok untuk continuous features (R, F, M adalah numeric)
- Deterministic results (dengan fixed random_state)

### Menentukan K Optimal

**Elbow Method**:
- Plot inertia (within-cluster sum of squares) vs K
- Cari "elbow" point (diminishing returns)
- Untuk PROMIND: elbow terlihat di K=3-4

**Silhouette Score** (digunakan di PROMIND):
- Mengukur seberapa baik point dieklustered
- Range: -1 to 1
  - 1 = cluster sempurna
  - 0 = overlapping clusters
  - -1 = misclassified points
- Untuk PROMIND: K=4 menghasilkan silhouette ≈ 0.35-0.40

**Decision untuk K=4**:
```python
# Dari PROMIND hasil analysis:
K    Silhouette    Inertia
2    0.28          500000
3    0.35          300000
4    0.39 ← OPTIMAL 200000
5    0.36          180000
6    0.33          170000
```

### Stability & Robustness

PROMIND melakukan **random state sensitivity analysis**:
- Jalankan KMeans 10 kali dengan berbeda random seed
- Hitung silhouette score per run
- Coefficient of Variation (CV) < 10% = STABLE model

Hasil:
```
Silhouette Mean: 0.385
Silhouette Std:  0.002
Silhouette CV:   0.005 (< 0.10) ← VERY STABLE ✓
```

Interpretasi: Model cluster sangat stabil, tidak sensitive terhadap initialization.

## Business Intelligence: Segment Profiling

### Ranking & Naming Strategy

PROMIND menggunakan **weighted ranking**:

```python
# Scoring logic:
Recency_rank = rank(Recency, ascending=True)      # Rendah adalah bagus
Frequency_rank = rank(Frequency, ascending=False) # Tinggi adalah bagus
Monetary_rank = rank(Monetary, ascending=False)   # Tinggi adalah bagus

Overall_rank = R_rank + F_rank + M_rank  # Semakin rendah semakin bagus
```

**Mapping cluster ke segment name**:
```
Rank 1 (lowest) → Champions        (high R, F, M)
Rank 2         → Loyal Customers  (stable R, F, M)
Rank 3         → Potential Loyalists (good R, low F, low M)
Rank 4 (highest) → At Risk        (high R = lama tidak beli)
```

### Value Efficiency Index

Formula: `Revenue_Contribution% / Customer_Share%`

**Interpretasi**:
- Index > 2 = Revenue Engine (high-value, efficient segment)
- Index 1-2 = Growth Driver (balanced value)
- Index < 1 = Cost-sensitive (low-value, perlu nurture)

Contoh:
```
Segment                Share%   Revenue%   Index    Role
Champions              10%      40%        4.0 → Revenue Engine ★★★
Loyal Customers        30%      45%        1.5 → Growth Driver ★★
Potential Loyalists    35%      12%        0.34 → Nurture needed
At Risk                25%      3%         0.12 → Churn prevention
```

**Strategic Implication**:
- Fokus retention pada Champions (highest ROI)
- Growth strategy pada Loyal Customers
- Reactive campaigns untuk At Risk
- Conversion programs untuk Potential Loyalists

## Outlier Analysis: The "Whale Effect"

### Phenomenon

Retail data menunjukkan **Pareto distribution** (80/20 rule):
- 20% customers menyumbang 80% revenue
- 1% customers (whales) menyumbang hingga 30% revenue
- Rest of customers: high volume, low value

PROMIND menganalisis top 1% monetary:

```
Top 1% Threshold: £5000+
Top 1% Distribution:
- 80% berada di cluster Champions
- 15% di cluster Loyal Customers
- 5% di cluster Potential Loyalists
- 0% di cluster At Risk

Revenue Concentration: Top 1% = 25-35% total revenue
```

### Implication

1. **Cluster Quality**: Overlay outliers lebih terkonsentrasi di high-value clusters = validasi cluster baik
2. **Marketing Strategy**: Whale customers perlu VIP treatment terpisah
3. **Mean vs Median**: Mean Monetary >> Median Monetary (driven by whales)
4. **Risk Management**: Kehilangan satu whale customer = significant revenue loss

## Comparison dengan Metode Alternatif

### Mengapa bukan DBSCAN?

**DBSCAN Kelebihan:**
- Tidak perlu specify K di awal
- Deteksi outliers otomatis
- Flexible cluster shapes

**DBSCAN Kekurangan:**
- Susah tune parameter (eps, min_samples)
- Banyak noise points di RFM data (tidak scalable)
- Cluster size sangat bervariasi (sulit interpret)

**PROMIND Decision**: K-Means lebih cocok untuk RFM karena:
- K=4 sudah optimal dan stable
- Business interpretation jelas (centroid = ideal customer profile)
- Semua customer ter-assign ke segmen (actionable untuk marketing)

### Mengapa bukan Hierarchical Clustering?

**Hierarchical Clustering Kelebihan:**
- Dendrogram visualization
- Flexible cutting threshold

**Hierarchical Clustering Kekurangan:**
- Computational expensive untuk 5000+ customers
- Tidak stable (berbeda linkage method → berbeda tree)
- Timing tidak feasible untuk real-time update

**PROMIND Decision**: K-Means lebih scalable dan robust.

## Implementation Insights

### 1. Feature Scaling Importance

Tanpa scaling:
- Monetary range: [10, 50000]
- Frequency range: [1, 100]
- Recency range: [1, 365]

KMeans akan dominan pada Monetary (biggest range). Dengan StandardScaler semua features memiliki weight sama.

### 2. Random State Reproducibility

```python
# Dengan random_state=42:
KMeans(n_clusters=4, random_state=42) → DETERMINISTIC
# Result selalu sama untuk same input

# Tanpa random_state:
KMeans(n_clusters=4) → DIFFERENT setiap run
# Hasil berubah-ubah (tidak ideal untuk production)
```

### 3. n_init Parameter

```python
# PROMIND menggunakan n_init=10:
# KMeans akan run 10 kali dengan different initialization
# Kemudian pick best result (lowest inertia)
# Ini meningkatkan hasil consistency
```

### 4. Silhouette vs Inertia Trade-off

| Metric | Mengukur | Use Case |
|--------|----------|----------|
| **Inertia** | Within-cluster tightness | Kompact cluster |
| **Silhouette** | Separation antar cluster | Well-separated |

PROMIND menggunakan **Silhouette** karena:
- Better untuk overlapping data
- Business-relevant (distinct segments)
- Lebih robust terhadap outliers

---

# Referensi
[`^ Kembali ke atas ^`](#top)

## Dataset & Sumber Data

1. **Online Retail II Dataset**
   - Source: UCI Machine Learning Repository & Kaggle
   - URL: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
   - Original Paper: [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/online+retail)
   - Description: Transaksi e-commerce UK (2009-2011)
   - Records: 500K+ transactions, 5K customers

## Python Libraries Documentation

### Data Processing
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Data manipulation
- [NumPy Documentation](https://numpy.org/doc/) - Numerical computing

### Visualization
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) - Static plots
- [Seaborn Documentation](https://seaborn.pydata.org/) - Statistical graphics
- [Plotly Documentation](https://plotly.com/python/) - Interactive visualization

### Machine Learning
- [Scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - K-Means, metrics
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) - StandardScaler, scaling

### Notebooks & Development
- [Jupyter Documentation](https://jupyter.org/) - Interactive computing
- [IPython Documentation](https://ipython.readthedocs.io/) - Enhanced Python shell

## Jurnal & Research Papers

1. **RFM Analysis Foundation**
   - Pranata, A., & Saepudin, E. (2021). Customer segmentation using RFM model and K-Means clustering: A case study on online retail dataset. *International Journal of Computer Applications*, 183(43), 18–23.
   - Sarı, T., & Cengiz, H. (2023). Customer segmentation using Recency, Frequency, and Monetary analysis with machine learning techniques. *Procedia Computer Science*, 219, 108–117.

2. **K-Means Clustering**
   - Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, 31(8), 651–666.
   - Xu, R., & Wunsch, D. (2005). Survey of clustering algorithms. *IEEE Transactions on Neural Networks*, 16(3), 645–678.

3. **Customer Segmentation & CRM**
   - Hosseini, S. B., Maleki, A., & Gholamian, M. R. (2010). Cluster analysis using data mining approach to develop CRM methodology to assess the customer loyalty. *Expert Systems with Applications*, 37(7), 5259–5264.
   - Sohn, S. Y., & Kim, H. S. (2008). Searching customer patterns of mobile service using clustering and quantitative association rule. *Expert Systems with Applications*, 34(2), 1070–1077.

4. **Machine Learning & Clustering Evaluation**
   - Han, J., Kamber, M., & Pei, J. (2012). Data mining: Concepts and techniques (3rd ed.). Morgan Kaufmann.
   - Kohonen, T. (2013). Essentials of the self-organizing map. *Neural Networks*, 37, 52–65.

5. **Data Privacy & Personalization**
   - Nguyen, B., Simkin, L., & Canhoto, A. (2020). The dark side of digital personalization: An agenda for research and practice. *Journal of Business Research*, 116, 209–221.

## Tools & Resources

### Online Tools
- [Kaggle Notebooks](https://www.kaggle.com/code) - Cloud-based Jupyter environment
- [Google Colab](https://colab.research.google.com) - Free GPU-enabled notebooks
- [Microsoft Azure ML Studio](https://ml.azure.com) - Enterprise ML platform

### Version Control
- [Git Documentation](https://git-scm.com/doc) - Version control basics
- [GitHub Guides](https://guides.github.com/) - GitHub workflow

### Additional Resources
- [Towards Data Science Blog](https://towardsdatascience.com/) - Data science articles
- [Medium Data Science Tags](https://medium.com/tag/data-science) - Community articles
- [Stack Overflow Data Science](https://stackoverflow.com/questions/tagged/data-science) - Q&A community

## Project References

### Similar Projects
- [Customer Segmentation Kaggle](https://www.kaggle.com/code) - Community implementations
- [RFM Analysis GitHub](https://github.com/search?q=rfm+segmentation) - Open-source projects

### Authors & Team

**Proyek Capstone PROMIND (A25-CS324)**

Anggota Tim:
1. M001D5Y1463 - Naufal Akmal Rizqulloh - Machine Learning
2. M001D5Y0111 - Ahmad Zaidan Al-Anshory - Machine Learning
3. M298D5Y1979 - Yoga Fatiqurrahman - Machine Learning

**Institusi**: Dicoding Indonesia x Accenture  
**Use Case**: Customer Segmentation for Personalized Retail Marketing (AC-06)  
**Created**: November - Desember 2025  
**Last Updated**: 14 Desember 2025

---

## Troubleshooting

### Common Issues & Solutions

**Q: Dataset tidak ditemukan "FileNotFoundError: online_retail_II.csv"**
- A: Pastikan folder `dataset/` ada dan file CSV sudah didownload
- Cek path: `print(os.path.exists("dataset/online_retail_II.csv"))`

**Q: Memory error saat run notebook**
- A: Dataset terlalu besar untuk RAM
- Solusi: Gunakan subset data atau upgrade RAM

**Q: Library tidak terinstall "ModuleNotFoundError"**
- A: Install requirements lagi
- Command: `pip install -r requirements.txt`

**Q: Cluster result berbeda setiap run**
- A: Kemungkinan mengubah random_state
- Pastikan `random_state=42` di semua KMeans

**Q: Output CSV tidak tersimpan di folder output/**
- A: Buat folder output: `mkdir output`
- Atau ubah path di code

---

<p align="center">
  <strong>Terima kasih telah menggunakan PROMIND! 🎯</strong><br>
  Untuk pertanyaan atau feedback, silakan buat issue di repository ini.
</p>

[`^ Kembali ke atas ^`](#top)
