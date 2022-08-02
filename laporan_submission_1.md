# Laporan Proyek Machine Learning - Lutfan Zainul Haq

  

## Domain Proyek

  

Jamur merupakan salah-satu jenis tumbuhan yang banyak dijumpai di alam, sehingga sejak dahulu jamur banyak dijadikan bahan konsumsi utama. di alam terbuka ada jenis jamur yang beracun dan yang dapat dikonsumsi. dan untuk membedakannya dapat dilihat berdasarkan bentuk, sifat, dan keadaanya yang mana itu sangat sulit dilakukan oleh masyarakat biasa. oleh karena itu dengan dibuatnya model machine learning ini dapat membantu masyarakat dalam membedakan antara jamur beracun dan jamur yang dapat dikonsumsi. dataset yang digunakan berupa kumpulan data jamur yang didapat dari UCI repository of machine learning. Dataset ini mencakup deskripsi sampel hipotesis yang sesuai dengan 23 spesies jamur insang di Agaricus dan Jamur Keluarga Lepiota yang diambil dari The Audubon Society Field Guide to North American Mushrooms (1981). Setiap spesies diidentifikasi sebagai pasti dapat dimakan, pasti beracun, atau tidak diketahui dapat dimakan dan tidak direkomendasikan. Kelas terakhir ini digabungkan dengan yang beracun.

  

**Rubrik/Kriteria Tambahan (Opsional)**:

- Permasalahan dan Solusi
dengan adanya berbagai jenis jamur yang ada di alam, akan sulit bagi masyarakat awam untuk mengidentifikasi jenis jamur yang bisa dimakan atau tidak. dengan adanya dataset jamur, bisa kita manfaatkan untuk membuat sebuah model machine learning untuk melakukan klasifikasi agar dapat membantu masyarakat dalam mengidentifikasi jamur di alam.

- Referensi :
[Klasifikasi Jenis Jamur Menggunakan SVM dengan Fitur HSV dan HOG](https://stt-pln.e-journal.id/petir/article/view/1101)
[dataset Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
[Implementasi Algoritma Random Forest Classifier dalam Melakukan Klasifikasi Kelayakan Edibilitas pada Jamur](https://kc.umn.ac.id/13543/)

  

## Business Understanding

  

### Problem Statements

Latar belakang masalah pada proyek ini adalah sebagai berikut :
- Algoritma machine learning apa yang bagus untuk melakukan klasifikasi jamur?
- Bagaimana cara membangun machine learning untuk melakukan klasifikasi?
- Bagaimana cara melakukan evaluasi model machine learning?

  
  

### Goals

  

Tujuan dari pernyataan masalah:
- Mencari beberapa algoritma machine learning untuk melakukan klasifikasi
- Dapat membangun machine learning untuk melakukan klasifikasi
- Melakukan evaluasi model yang telah dibuat

  

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution statements

- Pada proyek ini, saya akan menggunakan 3 algoritma machine learning yaitu **Linear Regression** (sebagai baseline model), **SVM** dan **Random forest**. dari 3 algoritma tersebut akan dibandingkan dan dipilih salah satu yang terbaik untuk melakukan klasifikasi jamur. saya juga menggunakan hyperparameter yang ada pada package `luwiji`. `luwiji` merupakan package yang dikembangkan oleh Wira D. K. Putra sebagai pelengkap kurikulum pembelajaran mesinnya bagi para peminat pembelajaran mesin muda. untuk memudahkan saya mendapatkan parameter terbaik, saya menggunakan fungsi `GridSearchCV()` dan `RandomSearchCV` yang terdapat pada library `sklearn`.
- Untuk evaluasi masing-masing model, saya menggunakan matriks evaluasi **accuracy**,**precision** dan **recall**.

  

## Data Understanding

Pada proyek ini saya menggunakan dataset jamur yang didapat dari UCI Machine Learning Repository. Kumpulan data ini mencakup deskripsi sampel hipotesis yang sesuai dengan 23 spesies jamur insang dalam Keluarga Agaricus dan Lepiota (hlm. 500-525). Setiap spesies diidentifikasi sebagai pasti dapat dimakan, pasti beracun, atau tidak diketahui dapat dimakan dan tidak direkomendasikan. Data ini memiliki  23 features dengan 8124 baris yang mana pada semua features tersebut bertipe object. dan pada masing-masing features tidak terdapat missing value.\
Untuk lebih lengkapnya dapat dicek pada tautan dibawah ini:\
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom).

  

### Variabel-variabel pada Mushrooms UCI dataset adalah sebagai berikut:

- class : kelas yang menentukan apakah jamur dapat dimakan atau beracun. features ini memiliki 2 kategori, yaitu:
	* e = dapat dimakan
	* p = beracun 
- cap-shape : Bentuk Kepala Jamur. Features ini memiliki 6 kategori, yaitu:
	* b = lonceng
	* c = kerucut
	* x = cembung
	* f = datar 
	* k = kenop
	* s = cekung
- cap-surface : Permukaan kepala jamur. Features ini memiliki 4 kategori, yaitu:
	* f = berserat
	* g = alur
	* y = bersisik
	* s = halus
- cap-color : warna kepala jamur. Features ini memiliki 10 kategori, yaitu: 
	* n = coklat
	* b = buff
	* c = kayu manis
	* g = abu-abu
	* r = hijau
	* p = pink
	* u = ungu
	* e = merah
	* w = putih
	* y = kuning
- bruises : memar pada jamur. Feature ini memiliki 2 kategori, yaitu:
	* t = terdapat memar
	* f = tidak terdapat memar
- odor : bau yang dimiliki oleh jamur. Features ini memiliki 9 kategori, yaitu:
	* a = almond
	* l = adas manis
	* c = kreosot
	* y = amis
	* f = busuk 
	* m = apek
	* n = tidak ada bau
	* p = menyengat
	* s = pedas
- gill-attachment: bentuk bagian bilah insang jamur. features ini memiliki 4 kategori, yaitu: 
	* a = terpasang
	* d = turun 
	* f = bebas 
	* n = berlekuk
- gill-spacing: Kerapatan bilah insang jamur. features ini memiliki 3 kategori, yaitu:
	* c = dekat
	* w = ramai
	* d = jauh
- gill-size : ukurang bilah insang jamur. features ini memiliki 2 kategori, yaitu:
	* b = luas
	* n = sempit
- gill-color : warna bilah insang jamur. features ini memiliki 12 kategori, yaitu:
	* k = hitam
	* n = coklat
	* b = buff
	* h = coklat
	* g = abu-abu
	* r = hijau
	* o = jingga
	* p = pink
	* u = ungun
	* e = merah
	* w = putih
	* y = kuning
- stalk-shape : bentuk tangkai jamur. Features ini memiliki 2 kategori, yaitu:
	* e = membesar
	* t = mengecil
- stalk-root : bentuk akar jamur. Features ini memiliki 7 kategori, yaitu:
	* b = bulat
	* c = klub
	* u = cangkir
	* e = sama
	* z = rhizomorphs
	* r = berakar
	* ? = belum diketahui
- stalk-surface-above-ring: bentuk permukaan tangkai di atas cincin jamur. Features ini memiliki 4 kategori, yaitu:
	* f = berserat
	* y = bersisik
	* k = halus sutra
	* s = halus
- stalk-surface-below-ring: bentuk permukaan tangkai di bawah cincin jamur. Features ini memiliki 4 kategori, yaitu:
	* f = berserat
	* y = bersisik
	* k = halus sutra
	* s = halus
- stalk-color-above-ring: warna tangkai di atas cincin jamur. Features ini memiliki 10 kategori, yaitu: 
	* n = coklat
	* b = buff
	* c = kayu manis
	* g = abu-abu
	* r = hijau
	* p = pink
	* u = ungu
	* e = merah
	* w = putih
	* y = kuning
- stalk-color-below-ring: warna tangkai di bawah cincin jamur. Features ini memiliki 10 kategori, yaitu: 
	* n = coklat
	* b = buff
	* c = kayu manis
	* g = abu-abu
	* r = hijau
	* p = pink
	* u = ungu
	* e = merah
	* w = putih
	* y = kuning
- veil-type : jenis veil jamur. Features ini memiliki 2 kategori, yaitu: 
	* p = parsial
	* u = universal
- veil-color : warna veil jamur. Features ini memiliki 4 kategori, yaitu: 
	* n = coklat
	* o = jingga
	* w = putih
	* y = kuning
- ring-number : ukuran cincin/annulus jamur. Features ini memiliki 3 kategori, yaitu: 
	* n = tidak ada
	* o = satu
	* t = dua
- ring-type : jenis cincin/annulus jamur. Features ini memiliki 8 kategori, yaitu: 
	* c = sarang laba-laba
	* e = evanescent
	* f = flaring
	* l = besar
	* n = tidak ada
	* p = liontin
	* s = selubung
	* z = zona
- spore-print-color : warna spora jamur. Features ini memiliki 10 kategori, yaitu: 
	* n = coklat
	* b = buff
	* c = kayu manis
	* g = abu-abu
	* r = hijau
	* p = pink
	* u = ungu
	* e = merah
	* w = putih
	* y = kuning
- population: populasi jamur yang ada di alam. Features ini memiliki 6 kategori, yaitu: 
	* a = berlimpah
	* c = berkerumun
	* n = banyak 
	* s = tersebar
	* v = beberapa
	* y = soliter
- habitat : habitat jamur. Features ini memiliki 7 kategori, yaitu: 
	* g = rumput
	* l = daun
	* m = padang rumput
	* p = jalan
	* u = perkotaan
	* w = limbah
	* d = kayu
  

**Rubrik/Kriteria Tambahan (Opsional)**:

Dalam memahami data yang akan saya gunakan untuk model, saya menggunakan library pandas, seaborn dan matplotlib. dengan pandas saya bisa mengetahui jumlah features data, type masing-masing features dan mengidentifikasi apakah terdapat missing value atau tidak. dalam mencari missing value saya juga  seaborn untuk memetakan seluruh data dengan plotting. karena keseluruhan features data merupakan categorical features sehingga saya menggunakan matplotlib untuk membantu saya dalam melihat berbagai kategori yang terdapat pada masing-masing features, disini saya juga melihat persebaran data pada masing-masing features.

  

## Data Preparation
- Sebelum menggunakan data untuk membuat model machine learning, saya terlebih dahulu melihat data dari missing value. pada proyek ini, saya menggunakan **pandas** dan **seaborn** untuk membantu saya dalam melihat missing value. pada library **pandas** terdapat fungsi `.isna().sum()` untuk mengetahui jumlah missing value pada masing-masing features. saya juga menggunakan plotting heatmap pada **seaborn** untuk memetakan seluruh data dan menandai warna yang berbeda jika terdapat missing value pada data. dan hasilnya dapat dilihat pada gambar berikut :

![missing_value](https://user-images.githubusercontent.com/87218279/182414061-7e11557a-b5f2-41ad-9c72-a03e41e86efb.png)

di tahap ini saya juga menggunakan **matplotlib** untuk melihat persebaran data untuk masing-masing features.

- Setelah data bersih dari missing value, saya membagi dataset menjadi data train dan data test. pada tahap ini saya menggunakan `train_test_split()` yang terdapat pada **sklearn**. Karena jumlah data >5000 maka saya membagi 75% untuk data train dan 25% untuk data test.

- Tahap selanjutnya adalah standarisasi. pada tahap standarisasi saya menggunakan `OneHotEncoder`. dan untuk meringkas kode, saya menggunakan `Pipelines` dan `ColumnTransformer` pada sklearn.

**Rubrik/Kriteria Tambahan (Opsional)**:

Pada proses data preparation, untuk standarisasi karena semua features berupa categorical features maka saya menggunakan method `OneHotEncoder` yang terdapat pada sklearn. method ini akan mengubah semua categorical features menjadi bilangan 0 dan 1. tujuan dari standarisasi ini adalah karena komputer sulit memproses data bertipe kategori sehingga perlu mengubah data tersebut berbentuk bilangan.\

Pada data preparation saya juga menggunakan `Pipelines` dan `ColumnTransformer` pada sklearn untuk merapikan dan meringkas code agar mudah dibaca.

  

## Modeling

- Pada tahap modeling saya menggunakan 3 algoritma machine learning, yaitu : LogisticRegression, SVM dan Random forest.

- Pada tahap modeling saya menggunakan 3 algoritma machine learning, yaitu : LogisticRegression, SVM dan Random forest.

- LogisticRegression akan menjadi baseline model machine learning. kemudian model ini nantinya akan dibandingkan dengan model machine learning yang menggunakan algoritma svm dan Random forest. dan disini saya menggunakan seluruh parameter yang terdapat pada masing-masing model. dan untuk mendapatkan best parameter, saya menggunakan `GridSearchCV` dan `RandomizedSearchCV`.

- Dari ketiga algoritma machine learning tersebut, didapatkan train score sebagai berikut:

| Nama | Train_score |
| ------ | ------ |
| LogisticRegression | 1.0 |
| SVM | 1.0 |
| RandomForest | 1.0 |

Dari tabel tersebut bisa dilihat, hanya menggunakan Logistic Regression score yang didapat bisa menyamai algoritma SVM dan RandomForest. Oleh karena itu saya memutuskan untuk memilih **Logistic Regression** sebagai model terbaik karena proses train nya yang lebih cepat dan score yang didapatkan sudah mampu menyamai algoritma machine learning yang lain.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.

- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.

- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

  

## Evaluation

- Untuk melakukan evaluasi pada masing-masing model saya menggunakan teknik accuracy, precision dan recall
	* **accuracy** merupakan matrik evaluasi yang menggambarkan seberapa akurat model dapat mengklasifikasikan dengan benar. Maka, accuracy merupakan rasio prediksi benar (positif dan negatif) dengan keseluruhan data. Dengan kata lain, accuracy merupakan tingkat kedekatan nilai prediksi dengan nilai aktual (sebenarnya).
	* **precision** merupakan matrik evaluasi yang menggambarkan tingkat keakuratan antara data yang diminta dengan hasil prediksi yang diberikan oleh model. Maka, precision merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf. Dari semua kelas positif yang telah diprediksi dengan benar, berapa banyak data yang benar-benar positif.
	* **recall** merupakan matrik evaluasi yang menggambarkan keberhasilan model dalam menemukan kembali sebuah informasi. Maka, recall merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.

- Dari evaluasi yang dilakukan, didapatkan hasil sebagai berikut:
 
| Nama | accuracy | precision | recall |
| ------ | ------ | ------ |------ |
| LogisticRegression | 1.0 | 1.0 | 1.0 |
| SVM | 1.0 | 1.0 | 1.0 |
| RandomForest | 1.0 | 1.0 | 1.0 |
  

**Rubrik/Kriteria Tambahan (Opsional)**:

- accuracy adalah proporsi dari instances pada train test yang diprediksi secara tepat.

accuracy  = $\frac{TP+TN}{TP+TN+FP+FN}$

- precision adalah proporsi dari testing set yang diprediksi oleh model yang benar-benar positif.

precision = $\frac{TP}{TP+FP}$

- recall adalah proporsi dari data pada testing set positive yang dipresidksi positif pada model.

recall = $\frac{TP}{TP+FN}$
