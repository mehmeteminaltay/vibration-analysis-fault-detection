import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# --- 1. VERİ YÜKLEME FONKSİYONU ---
def veri_yukle(dosya_yolu):
    try:
        mat = scipy.io.loadmat(dosya_yolu)
        # .mat dosyası içinde 'DE_time' (Drive End Time) ile biten anahtarı bul
        for key in mat.keys():
            if key.endswith("DE_time"):
                print(f"{dosya_yolu} başarıyla yüklendi. Veri anahtarı: {key}")
                return mat[key]
        print(f"Hata: {dosya_yolu} içinde titreşim verisi bulunamadı!")
        return None
    except FileNotFoundError:
        print(f"Hata: {dosya_yolu} dosyası bulunamadı. Dosya adını kontrol et!")
        return None

# --- 2. DOSYALARI OKU ---
# Dosya isimlerin farklıysa burayı değiştirmeyi unutma!
saglam_sinyal = veri_yukle('normal.mat') 
arizali_sinyal = veri_yukle('arizali.mat')

# Eğer veriler yüklenemezse kodu durdur
if saglam_sinyal is None or arizali_sinyal is None:
    print("Veriler yüklenemediği için işlem durduruldu.")
else:
    # Veriyi tek boyuta indir (örneğin [[1], [2]] -> [1, 2])
    saglam_sinyal = saglam_sinyal.reshape(-1)
    arizali_sinyal = arizali_sinyal.reshape(-1)

    # --- 3. GÖRSELLEŞTİRME: ZAMAN BÖLGESİ (TIME DOMAIN) ---
    plt.figure(figsize=(15, 8))
    
    # Sağlam Rulman
    plt.subplot(2, 1, 1)
    plt.plot(saglam_sinyal[:2000], color='blue', linewidth=1)
    plt.title("SAĞLAM Rulman - Zaman Sinyali / HEALTHY Bearing - Time Domain", fontsize=12, fontweight='bold')
    plt.ylabel("Genlik / Amplitude", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Arızalı Rulman
    plt.subplot(2, 1, 2)
    plt.plot(arizali_sinyal[:2000], color='red', linewidth=1)
    plt.title("ARIZALI Rulman - Zaman Sinyali / FAULTY Bearing - Time Domain", fontsize=12, fontweight='bold')
    plt.ylabel("Genlik / Amplitude", fontsize=10)
    plt.xlabel("Zaman / Time (Sample Points)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    # --- 4. HESAPLAMA: FFT (FREKANS ANALİZİ) ---
    fs = 12000 # Örnekleme Frekansı (Hz)
    N = len(arizali_sinyal) # Toplam veri sayısı
    
    # FFT Al
    yf_saglam = fft(saglam_sinyal)
    yf_arizali = fft(arizali_sinyal)
    
    # Frekans eksenini oluştur
    xf = fftfreq(N, 1/fs)
    
    # Pozitif tarafı al
    idx_max = N // 2
    xf_plot = xf[:idx_max]
    yf_saglam_plot = np.abs(yf_saglam[:idx_max])
    yf_arizali_plot = np.abs(yf_arizali[:idx_max])

    # --- 5. GÖRSELLEŞTİRME: FREKANS BÖLGESİ (FREQUENCY DOMAIN) ---
    plt.figure(figsize=(15, 8))
    
    # Sağlam FFT
    plt.subplot(2, 1, 1)
    plt.plot(xf_plot, yf_saglam_plot, color='blue')
    plt.title("SAĞLAM Rulman - Frekans Spektrumu / HEALTHY Bearing - FFT Spectrum", fontsize=12, fontweight='bold')
    plt.ylabel("Genlik / Amplitude")
    plt.grid(True)
    plt.xlim(0, 2000) # İlk 2000 Hz'e odaklan
    
    # Arızalı FFT
    plt.subplot(2, 1, 2)
    plt.plot(xf_plot, yf_arizali_plot, color='red')
    plt.title("ARIZALI Rulman - Frekans Spektrumu / FAULTY Bearing - FFT Spectrum", fontsize=12, fontweight='bold')
    plt.ylabel("Genlik / Amplitude")
    plt.xlabel("Frekans / Frequency (Hz)")
    plt.grid(True)
    plt.xlim(0, 2000) # İlk 2000 Hz'e odaklan

    plt.tight_layout()
    plt.show()
    import pandas as pd
from scipy.stats import skew, kurtosis

# --- ÖZNİTELİK ÇIKARMA FONKSİYONU ---
def ozellik_cikar(sinyal, etiket, parca_boyutu=1000):
    ozellikler = []
    
    # Sinyali parçalara bölüyoruz (Örn: her 1000 veride bir analiz yap)
    for i in range(0, len(sinyal) - parca_boyutu, parca_boyutu):
        parca = sinyal[i : i + parca_boyutu]
        
        # Zaman Bölgesi Özellikleri (Time Domain Features)
        rms = np.sqrt(np.mean(parca**2))
        max_val = np.max(np.abs(parca))
        kurt = kurtosis(parca)
        skw = skew(parca)
        
        # Listeye ekle (Etiket: 0=Sağlam, 1=Arızalı)
        ozellikler.append([rms, max_val, kurt, skw, etiket])
        
    return ozellikler

# --- VERİ SETİNİ OLUŞTURMA ---
parca_boyutu = 1000  # Her 1000 örneklemde bir özellik çıkaracağız

# Sağlam veriden özellik çıkar (Etiket = 0)
saglam_ozellikler = ozellik_cikar(saglam_sinyal, etiket=0, parca_boyutu=parca_boyutu)

# Arızalı veriden özellik çıkar (Etiket = 1)
arizali_ozellikler = ozellik_cikar(arizali_sinyal, etiket=1, parca_boyutu=parca_boyutu)

# Hepsini birleştirip bir Tablo (DataFrame) yapalım
tum_veri = saglam_ozellikler + arizali_ozellikler
df = pd.DataFrame(tum_veri, columns=['RMS', 'Max_Deger', 'Kurtosis', 'Skewness', 'Durum'])

# --- SONUCU GÖSTER ---
print(f"Toplam Veri Sayısı: {len(df)}")
print("\nVeri Setinden İlk 5 Satır (Durable):")
print(df.head())
print("\nVeri Setinden Son 5 Satır (Defective):")
print(df.tail())

# --- KÜÇÜK BİR GÖRSELLEŞTİRME ---
# RMS değerlerinin farkını görelim
plt.figure(figsize=(10, 6))
plt.scatter(df[df['Durum']==0].index, df[df['Durum']==0]['RMS'], label='Durable (0)', color='blue', alpha=0.6)
plt.scatter(df[df['Durum']==1].index, df[df['Durum']==1]['RMS'], label='Defective (1)', color='red', alpha=0.6)
plt.title("Comparison of RMS Values ​​of Healthy and Faulty Data")
plt.xlabel("Sample Number")
plt.ylabel("RMS Value")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. DATA PREPARATION (Converting to English) ---
# Renaming columns to English for the final report
df.columns = ['RMS', 'Max_Value', 'Kurtosis', 'Skewness', 'Status']

X = df[['RMS', 'Max_Value', 'Kurtosis', 'Skewness']]
y = df['Status']

# Split Data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. MODEL TRAINING (Random Forest) ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# --- 3. RESULTS & VISUALIZATION ---

# A. Accuracy Score
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# B. Visualization
plt.figure(figsize=(14, 6))

# Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
plt.title("Confusion Matrix (Model Performance)", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.xticks([0.5, 1.5], ['Healthy', 'Faulty'], fontsize=11)
plt.yticks([0.5, 1.5], ['Healthy', 'Faulty'], fontsize=11, rotation=0)

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] # Sort features by importance
feature_names = X.columns

# Reorder feature names based on importance
sorted_features = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

# Bar Plot
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'] # Custom colors
plt.bar(sorted_features, sorted_importances, color=colors, edgecolor='black')
plt.title("Feature Importance Analysis", fontsize=14, fontweight='bold')
plt.ylabel("Importance Score", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()