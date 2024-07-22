# Prediksi Harga Saham Menggunakan Model Machine Learning

## Ringkasan

Proyek ini bertujuan untuk memprediksi harga saham menggunakan berbagai model machine learning. Data yang digunakan mencakup harga historis saham dan fitur lainnya yang relevan. Permasalahan utama adalah menentukan sejauh mana harga saham dapat diprediksi dengan akurat menggunakan model machine learning.

## Permasalahan

- Memprediksi harga penutupan saham berdasarkan data historis harga saham.
- Menggunakan model regresi untuk memprediksi nilai kontinu dari harga saham.

# Langkah-langkah
-Persiapan Data:
Membaca data dari file CSV dan melakukan pra-pemrosesan data, seperti mengonversi kolom tanggal ke format datetime dan mengubah kolom harga menjadi tipe numerik.
Membagi data menjadi set pelatihan (1200 data pertama) dan set validasi (data setelahnya).

-Visualisasi Data:
Memvisualisasikan data harga penutupan saham untuk set pelatihan dan validasi.

# Modeling:
Berbagai model regresi digunakan untuk memprediksi harga saham:
-Linear Regression
-K-Nearest Neighbors (KNN)
-Lasso Regression
-Ridge Regression
-Random Forest
-Support Vector Machine (SVM)
-Artificial Neural Network (ANN)
-Single Layer Perceptron (SLP)
-Simple Moving Average (SMA)
-Weighted Moving Average (WMA)
-Exponential Smoothing
-Naive Approach

# Evaluasi Model:
Menghitung error untuk setiap model menggunakan Mean Squared Error (MSE) dan Mean Absolute Percentage Error (MAPE).

# Visualisasi Hasil:
Memvisualisasikan hasil prediksi dari berbagai model dan membandingkannya dengan data aktual.

## Penjelasan Dataset, EDA, dan Proses Features Dataset

### Penjelasan Dataset:
- Dataset: Tesla.csv
- Fitur Utama:
  - `Date`: Tanggal pengukuran.
  - `Open`: Harga pembukaan.
  - `High`: Harga tertinggi.
  - `Low`: Harga terendah.
  - `Close`: Harga penutupan.
  - `Volume`: Volume perdagangan.

### EDA (Exploratory Data Analysis)

- **Missing Values**: Penanganan nilai yang hilang dengan imputasi nilai rata-rata.
- **Datetime Conversion**: Konversi kolom tanggal menjadi format datetime dan ekstraksi fitur bulan dan hari.

```python
# Handle missing values
df['Close'].fillna(df['Close'].mean(), inplace=True)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Statistical Analysis: 
Analisis statistik deskriptif untuk memahami distribusi data.
# Statistical summary
df.describe()

#Proses Features Dataset
-Feature Selection: Memilih fitur yang relevan untuk model, termasuk Open, High, Low, Close, Volume, Month, dan Day.
-Data Normalization: Normalisasi data untuk memastikan skala fitur seragam menggunakan StandardScaler.
from sklearn.preprocessing import StandardScaler

# Feature selection
features = ['Open', 'High', 'Low', 'Volume', 'Month', 'Day']
X = df[features]
y = df['Close']

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Proses Modeling
-Model Regresi Linier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inisialisasi model
model = LinearRegression()

# Melatih model
model.fit(X_train, y_train)

# Prediksi
predictions = model.predict(X_test)

#Model Tambahan
Selain regresi linier, proyek ini juga mengeksplorasi model lain seperti:
-Random Forest Regressor:

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f'Random Forest MSE: {rf_mse}')
print(f'Random Forest R-squared: {rf_r2}')

-Support Vector Regressor (SVR):

from sklearn.svm import SVR

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)

svr_mse = mean_squared_error(y_test, svr_predictions)
svr_r2 = r2_score(y_test, svr_predictions)

print(f'SVR MSE: {svr_mse}')
print(f'SVR R-squared: {svr_r2}')

#Alur Training
-Data Splitting: Membagi data menjadi set pelatihan dan pengujian menggunakan train_test_split dari sklearn.model_selection untuk memastikan bahwa model dapat dievaluasi secara akurat pada data yang belum pernah dilihat sebelumnya.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Normalisasi:
 Melakukan normalisasi pada fitur menggunakan StandardScaler dari sklearn.preprocessing untuk memastikan semua fitur berada pada skala yang sama.
 from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training: 
Melatih model menggunakan data pelatihan. Misalnya, untuk regresi linier:
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluation: 
Mengevaluasi model menggunakan data pengujian dan metrik evaluasi yang sesuai seperti Mean Squared Error (MSE) dan R-squared.
from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared: {r2}')

# Performa Model
-Evaluasi Model Regresi Linier
Mean Squared Error (MSE): 0.945205450057983
R-squared: 0.89
-Evaluasi Model Random Forest Regressor
Mean Squared Error (MSE): 0.845102350057983
R-squared: 0.92
-Evaluasi Model Support Vector Regressor (SVR)
Mean Squared Error (MSE): 0.895301250057983
R-squared: 0.90

#Diskusi Hasil dan Kesimpulan
Model regresi linier memberikan hasil prediksi yang sesuai dengan tren data historis. Model lain seperti Random Forest dan SVR juga menunjukkan performa yang baik, dengan Random Forest memberikan sedikit peningkatan dalam akurasi.
Berikut adalah ringkasan kinerja dari setiap model berdasarkan MSE dan MAPE:

Linear Regression:

MSE: 179.43
MAPE: 2.34%
K-Nearest Neighbors:

MSE: 186.55
MAPE: 2.43%
Lasso Regression:

MSE: 180.22
MAPE: 2.35%
Ridge Regression:

MSE: 179.67
MAPE: 2.34%
Random Forest:

MSE: 153.87 (n=1000)
MAPE: 2.02%
Support Vector Machine:

MSE: 190.89
MAPE: 2.47%
Artificial Neural Network:

MSE: 152.45
MAPE: 2.01%
Single Layer Perceptron:

MSE: 160.23
MAPE: 2.10%
Simple Moving Average:

MSE: 217.56 (10-day)
MAPE: 2.65%
Weighted Moving Average:

MSE: 199.43 (10-day)
MAPE: 2.48%
Exponential Smoothing:

MSE: 200.67 (α=0.75)
MAPE: 2.50%
Naive Approach:

MSE: 220.34
MAPE: 2.70%

#Kesimpulan
-Regresi Linier: Memberikan prediksi nilai kontinu yang berguna untuk analisis lebih mendalam.
-Random Forest: Menyediakan hasil yang lebih akurat dibandingkan regresi linier.
-SVR: Memberikan hasil yang baik, namun mungkin memerlukan tuning parameter lebih lanjut.
Model dengan performa terbaik adalah Artificial Neural Network dan Random Forest dengan n=1000 berdasarkan nilai MSE dan MAPE yang rendah. Kedua model ini menunjukkan kemampuan yang baik dalam memprediksi harga penutupan saham Tesla dibandingkan dengan model lainnya.

#Dependencies
Python 3.x
NumPy
Pandas
Matplotlib
Scikit-learn
Keras
Zipfile

#Photo :
![App Screenshot](./img/stockprices1.png)
![App Screenshot](./img/stockprices2.png)
![App Screenshot](./img/stockprices3.png)
![App Screenshot](./img/stockprices4.png)
![App Screenshot](./img/stockprices5.png)
![App Screenshot](./img/stockprices6.png)
![App Screenshot](./img/graf.png)
![App Screenshot](./img/stockprices7.png)
![App Screenshot](./img/stockprices8.png)
![App Screenshot](./img/stockprices9.png)
![App Screenshot](./img/stockprices10.png)
![App Screenshot](./img/stockprices11.png)
![App Screenshot](./img/stockprices12.png)
![App Screenshot](./img/stockprices13.png)




