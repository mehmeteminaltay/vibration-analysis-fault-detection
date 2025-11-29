# AI-Driven Predictive Maintenance for Armored Vehicle Drivetrains

![Status](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🌍 Project Overview (English)
In high-stakes defense and industrial environments, mechanical failures in rotating machinery (e.g., helicopter transmissions, tank track systems) can lead to mission abortion or safety hazards. The goal of this project was to develop a **Condition-Based Monitoring (CBM)** system capable of detecting bearing faults before catastrophic failure occurs.

Using the **Case Western Reserve University (CWRU)** dataset, raw vibration signals were analyzed using **Signal Processing (FFT)** and **Machine Learning (Random Forest)**.

### Key Results
* **100% Classification Accuracy** on the test set.
* Successfully identified specific fault frequencies (**BPFI**) in the frequency spectrum.
* Determined **Max Value** and **RMS** as the most critical indicators for fault detection.

---

## 🇹🇷 Proje Raporu ve Teknik Detaylar (Turkish)

### 1.1 Projenin Amacı
Savunma sanayii ve ağır sanayide kullanılan araçların yürüyen aksamlarının ve dönen parçalarının (rulman, şanzıman vb.) sürdürülebilirliği için analiz edilmesi hedeflenmektedir. Geleneksel bakım yerine sensör verileri kullanılarak durum bazlı kestirimci bakım (Predictive Maintenance) yapılması planlanmaktadır.

### 2.1 Sinyal İşleme ve Frekans Analizi
Analiz kısmında CWRU sitesinden indirilen 12k Drive End verileri kullanılmıştır. Sinyallerin analizi ve işlenmesi için Python programı kullanılmıştır.
Sinyal analizi kısmında ilk işlem olarak datalar yüklenmiştir. Daha sonra ise zaman bağlı grafikler çizdirilmiştir. Figure 1 de görüldüğü gibi arızalı olan rulmanda darbeler olduğu gözlemlenmiştir. Bu darbelerin frekansını bulmak için sinyallere FFT (Hızlı Fourier Dönüşümü) uygulanmıştır. Figure 2 de görüldüğü üzere arızalı sinyalde belirli frekanslarda (160 Hz civarı) enerji artışı olduğu tespit edilmiştir.

### 3.1 Öznitelik Çıkarımı ve Yapay Zeka
Ham sinyallerin direk analizi zor olduğu için sinyallerden öznitelik çıkarımı yapılması hedeflenmiştir. Sinyaller 1000 lik parçalara bölünerek her parça için RMS, Kurtosis ve Maksimum değerleri hesaplanmıştır.
Figure 3 te görüldüğü üzere sağlam ve arızalı veriler RMS değerleri üzerinden hesaplandığında birbirinden ayrıldığı görülmüştür.

### 4.1 Sonuçların Değerlendirilmesi
Elde edilen öznitelik dataları Random Forest algoritiması ile eğitilmiştir. Verilerin %80 i eğitim %20 si test verisi olarak ayrılmıştır.
Modelin analizi sonucunda %100 doğruluk (Accuracy) elde edilmiştir. Karmaşıklık matrisi (Confusion Matrix) Figure 4 te verilmiştir. Matriste görüldüğü gibi model sağlam ve arızalı parçaları hatasız ayırmıştır.

---

### 💻 How to Run (Nasıl Çalıştırılır?)

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the main script:
    ```bash
    python main.py
    ```

---

![Figure 1](Figure_1.png)
*Created by Mehmet Emin Altay - Mechanical Engineer *
