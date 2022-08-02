# Laporan Proyek Machine Learning - Lutfan Zainul Haq

  

## Domain Proyek

  

Jamur merupakan salah-satu jenis tumbuhan yang banyak dijumpai di alam, sehingga sejak dahulu jamur banyak dijadikan bahan konsumsi utama. di alam terbuka ada jenis jamur yang beracun dan yang dapat dikonsumsi. dan untuk membedakannya dapat dilihat berdasarkan bentuk, sifat, dan keadaanya yang mana itu sangat sulit dilakukan oleh masyarakat biasa. oleh karena itu dengan dibuatnya model machine learning ini dapat membantu masyarakat dalam membedakan antara jamur beracun dan jamur yang dapat dikonsumsi. dataset yang digunakan berupa kumpulan data jamur yang didapat dari UCI repository of machine learning. Dataset ini mencakup deskripsi sampel hipotesis yang sesuai dengan 23 spesies jamur insang di Agaricus dan Jamur Keluarga Lepiota yang diambil dari The Audubon Society Field Guide to North American Mushrooms (1981). Setiap spesies diidentifikasi sebagai pasti dapat dimakan, pasti beracun, atau tidak diketahui dapat dimakan dan tidak direkomendasikan. Kelas terakhir ini digabungkan dengan yang beracun[1].

  

**Rubrik/Kriteria Tambahan (Opsional)**:

- Permasalahan dan Solusi
dengan adanya berbagai jenis jamur yang ada di alam, akan sulit bagi masyarakat awam untuk mengidentifikasi jenis jamur yang bisa dimakan atau tidak. dengan adanya dataset jamur, bisa kita manfaatkan untuk membuat sebuah model machine learning untuk melakukan klasifikasi agar dapat membantu masyarakat dalam mengidentifikasi jamur di alam.
  

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

Pada proyek ini saya menggunakan dataset jamur yang didapat dari UCI Machine Learning Repository. Kumpulan data ini mencakup deskripsi sampel hipotesis yang sesuai dengan 23 spesies jamur insang dalam Keluarga Agaricus dan Lepiota (hlm. 500-525). Setiap spesies diidentifikasi sebagai pasti dapat dimakan, pasti beracun, atau tidak diketahui dapat dimakan dan tidak direkomendasikan. Data ini memiliki  23 _features_ dengan 8124 baris yang mana pada semua _features_ tersebut bertipe _object_. dan pada masing-masing _features_ tidak terdapat missing value.\
Untuk lebih lengkapnya dapat dicek pada tautan dibawah ini:\
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom).

  

### Variabel-variabel pada _Mushrooms_ UCI dataset adalah sebagai berikut:

- class : kelas yang menentukan apakah jamur dapat dimakan atau beracun. _Features_ ini memiliki 2 kategori, yaitu:
	* e = dapat dimakan
	* p = beracun 
- cap-shape : Bentuk Kepala Jamur. _Features_ ini memiliki 6 kategori, yaitu:
	* b = lonceng
	* c = kerucut
	* x = cembung
	* f = datar 
	* k = kenop
	* s = cekung
- cap-surface : Permukaan kepala jamur. _Features_ ini memiliki 4 kategori, yaitu:
	* f = berserat
	* g = alur
	* y = bersisik
	* s = halus
- cap-color : warna kepala jamur. _Features_ ini memiliki 10 kategori, yaitu: 
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
- bruises : memar pada jamur. _Feature_ ini memiliki 2 kategori, yaitu:
	* t = terdapat memar
	* f = tidak terdapat memar
- odor : bau yang dimiliki oleh jamur. _Features_ ini memiliki 9 kategori, yaitu:
	* a = almond
	* l = adas manis
	* c = _kreosot_
	* y = amis
	* f = busuk 
	* m = apek
	* n = tidak ada bau
	* p = menyengat
	* s = pedas
- gill-attachment: bentuk bagian bilah insang jamur. _Features_ ini memiliki 4 kategori, yaitu: 
	* a = terpasang
	* d = turun 
	* f = bebas 
	* n = berlekuk
- gill-spacing: Kerapatan bilah insang jamur. _Features_ ini memiliki 3 kategori, yaitu:
	* c = dekat
	* w = ramai
	* d = jauh
- gill-size : ukurang bilah insang jamur. _Features_ ini memiliki 2 kategori, yaitu:
	* b = luas
	* n = sempit
- gill-color : warna bilah insang jamur. _Features_ ini memiliki 12 kategori, yaitu:
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
- stalk-shape : bentuk tangkai jamur. _Features_ ini memiliki 2 kategori, yaitu:
	* e = membesar
	* t = mengecil
- stalk-root : bentuk akar jamur. _Features_ ini memiliki 7 kategori, yaitu:
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
- stalk-surface-below-ring: bentuk permukaan tangkai di bawah cincin jamur. _Features_ ini memiliki 4 kategori, yaitu:
	* f = berserat
	* y = bersisik
	* k = halus sutra
	* s = halus
- stalk-color-above-ring: warna tangkai di atas cincin jamur. _Features_ ini memiliki 10 kategori, yaitu: 
	* n = coklat
	* b = _buff_
	* c = kayu manis
	* g = abu-abu
	* r = hijau
	* p = pink
	* u = ungu
	* e = merah
	* w = putih
	* y = kuning
- stalk-color-below-ring: warna tangkai di bawah cincin jamur. _Features_ ini memiliki 10 kategori, yaitu: 
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
- veil-type : jenis veil jamur. _Features_ ini memiliki 2 kategori, yaitu: 
	* p = parsial
	* u = universal
- veil-color : warna veil jamur. _Features_ ini memiliki 4 kategori, yaitu: 
	* n = coklat
	* o = jingga
	* w = putih
	* y = kuning
- ring-number : ukuran cincin/annulus jamur. _Features_ ini memiliki 3 kategori, yaitu: 
	* n = tidak ada
	* o = satu
	* t = dua
- ring-type : jenis cincin/annulus jamur. _Features_ ini memiliki 8 kategori, yaitu: 
	* c = sarang laba-laba
	* e = _evanescent_
	* f = _flaring_
	* l = besar
	* n = tidak ada
	* p = liontin
	* s = selubung
	* z = zona
- spore-print-color : warna spora jamur. _Features_ ini memiliki 10 kategori, yaitu: 
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
- population: populasi jamur yang ada di alam. _Features_ ini memiliki 6 kategori, yaitu: 
	* a = berlimpah
	* c = berkerumun
	* n = banyak 
	* s = tersebar
	* v = beberapa
	* y = soliter
- habitat : habitat jamur. _Features_ ini memiliki 7 kategori, yaitu: 
	* g = rumput
	* l = daun
	* m = padang rumput
	* p = jalan
	* u = perkotaan
	* w = limbah
	* d = kayu
  

**Rubrik/Kriteria Tambahan (Opsional)**:

Dalam memahami data yang akan saya gunakan untuk model, saya menggunakan library pandas, seaborn dan matplotlib. dengan pandas saya bisa mengetahui jumlah _features_ data, tipe masing-masing _features_ dan mengidentifikasi apakah terdapat _missing value_ atau tidak. dalam mencari _missing value_ saya juga seaborn untuk memetakan seluruh data dengan _plotting_. karena keseluruhan _features_ data merupakan _categorical features_ sehingga saya menggunakan matplotlib untuk membantu saya dalam melihat berbagai kategori yang terdapat pada masing-masing _features_, disini saya juga melihat persebaran data pada masing-masing .

  

## Data Preparation
- Sebelum menggunakan data untuk membuat model machine learning, saya terlebih dahulu melihat data dari _missing value_. pada proyek ini, saya menggunakan **pandas** dan **seaborn** untuk membantu saya dalam melihat _missing value_. pada library **pandas** terdapat fungsi `.isna().sum()` untuk mengetahui jumlah _missing value_ pada masing-masing _features_. saya juga menggunakan _plotting heatmap_ pada **seaborn** untuk memetakan seluruh data dan menandai warna yang berbeda jika terdapat _missing value_ pada data. dan hasilnya dapat dilihat pada gambar berikut :
![missing_value](https://user-images.githubusercontent.com/87218279/182414061-7e11557a-b5f2-41ad-9c72-a03e41e86efb.png)
dapat dilihat dari gambar di atas, tidak ada warna yang berbeda pada masing-masing _features_ yang menandakan tidak adanya _missing value_. di tahap ini saya juga menggunakan **matplotlib** untuk melihat persebaran data untuk masing-masing _features_. dan didapatkan hasil sebagai berikut: 
![plot2](https://user-images.githubusercontent.com/87218279/182416063-9393895a-7316-40de-b990-06375ae6fb98.png)
![plot3](https://user-images.githubusercontent.com/87218279/182416070-0d5d526e-20df-44d6-be15-e25ad096da74.png)
![plot4](https://user-images.githubusercontent.com/87218279/182416074-576dbcce-12ec-415d-9efc-29e470ca81fe.png)
![plot5](https://user-images.githubusercontent.com/87218279/182416077-b5f21c3e-d8e7-4501-b3b6-d68fc9aefbe3.png)
![plot6](https://user-images.githubusercontent.com/87218279/182416084-2cb80462-d83a-45b8-af22-a689f3177309.png)
![plot7](https://user-images.githubusercontent.com/87218279/182416088-c7e708fa-7dee-48ce-b9b7-68f3137b00d7.png)
![plot8](https://user-images.githubusercontent.com/87218279/182416094-62e6419a-2e47-4823-8e12-e22776330119.png)
![plot9](https://user-images.githubusercontent.com/87218279/182416100-3ed29a55-af5f-4f3c-8a38-a6da3f6016bb.png)
![plot10](https://user-images.githubusercontent.com/87218279/182416107-d780dfa5-3bfd-481b-9fbb-600ca6ba29e0.png)
![plot11](https://user-images.githubusercontent.com/87218279/182416113-7d2f0b20-c181-40be-97b7-662256101bab.png)
![plot12](https://user-images.githubusercontent.com/87218279/182416117-ac08a52b-4111-4fa3-8daf-712fc0b5b424.png)
![plot13](https://user-images.githubusercontent.com/87218279/182416122-7cf95ccc-7635-46e3-a2ac-45d94d8b628a.png)
![plot14](https://user-images.githubusercontent.com/87218279/182416125-1ed5ff43-6367-4210-9548-1b2457caef54.png)
![plot15](https://user-images.githubusercontent.com/87218279/182416133-47c4c0c1-ece1-487c-b518-2d471959c4a6.png)
![plot16](https://user-images.githubusercontent.com/87218279/182416135-b3bc8f6e-80e8-4903-99e4-3adc0bf9178b.png)
![plot17](https://user-images.githubusercontent.com/87218279/182416141-7c54f5d3-877f-4610-9733-f8bc56c5dcc5.png)
![plot18](https://user-images.githubusercontent.com/87218279/182416148-a920ffee-fa2a-42b3-a8d9-e479b1520c28.png)
![plot19](https://user-images.githubusercontent.com/87218279/182416153-01e838b6-4cf8-43e6-b49f-c2dc3cfad3d3.png)
![plot21](https://user-images.githubusercontent.com/87218279/182416162-ec1a103e-c8d6-4b99-baa6-8b285a5245da.png)
![plot22](https://user-images.githubusercontent.com/87218279/182416165-cba95a51-0df1-4dcf-ad06-506e1da13295.png)
![plot1](https://user-images.githubusercontent.com/87218279/182416169-d185297d-8aed-49da-be92-f3d0bddd0776.png)\
Dapat dilihat dari hasil _plotting_ diatas ada satu _features_ yang tidak memberikan insight, yaitu _features_ **veil-type**. _features_ ini hanya terdapat satu kategori sehingga bisa dibuang untuk mengurangi dimensi pada model nanti.

- Setelah data bersih dari _missing value_, saya membagi dataset menjadi data train dan data test. pada tahap ini saya menggunakan `train_test_split()` yang terdapat pada **sklearn**. Karena jumlah data >5000 maka saya membagi 75% untuk data train dan 25% untuk data test.

- Tahap selanjutnya adalah standarisasi. pada tahap standarisasi saya menggunakan `OneHotEncoder`. dan untuk meringkas kode, saya menggunakan `Pipelines` dan `ColumnTransformer` pada sklearn.

**Rubrik/Kriteria Tambahan (Opsional)**:

Pada proses _data preparation_, untuk standarisasi karena semua _features_ berupa _categorical features_ maka saya menggunakan method `OneHotEncoder` yang terdapat pada sklearn. method ini akan mengubah semua _categorical features_ menjadi bilangan 0 dan 1. tujuan dari standarisasi ini adalah karena komputer sulit memproses data bertipe kategori sehingga perlu mengubah data tersebut berbentuk bilangan[2].\

Pada data preparation saya juga menggunakan `Pipelines` dan `ColumnTransformer` pada sklearn untuk merapikan dan meringkas code agar mudah dibaca.

  

## Modeling

- Pada tahap modeling saya menggunakan 3 algoritma machine learning, yaitu : Logistic Regression, SVM dan Random forest.

- Pada tahap modeling saya menggunakan 3 algoritma machine learning, yaitu : Logistic Regression, SVM dan Random forest.

- Logistic Regression akan menjadi baseline model machine learning. kemudian model ini nantinya akan dibandingkan dengan model machine learning yang menggunakan algoritma SVM dan Random forest. dan disini saya menggunakan seluruh parameter yang terdapat pada masing-masing model. Pada algoritma Logistic Regressiond dan SVM untuk mendapatkan best  saya menggunakan `GridSearchCV`,   untuk algoritma Random forest karena memiliki banyak parameter maka saya menggunakan `RandomizedSearchCV` agar proses _train_ jauh lebih cepat.

- Dari ketiga algoritma machine learning tersebut, didapatkan _train score_ sebagai berikut:

| Nama | Train_score |
| ------ | ------ |
| LogisticRegression | 1.0 |
| SVM | 1.0 |
| RandomForest | 1.0 |

Dari tabel tersebut bisa dilihat, hanya menggunakan Logistic Regression _score_ yang didapat bisa menyamai algoritma SVM dan RandomForest. Oleh karena itu saya memutuskan untuk memilih **Logistic Regression** sebagai model terbaik karena proses trainnya yang lebih cepat dan _score_ yang didapatkan sudah mampu menyamai algoritma machine learning yang lain.
  

## Evaluation

- Untuk melakukan evaluasi pada masing-masing model saya menggunakan teknik _accuracy, precision_ dan _recall_.
	* **accuracy** merupakan matrik evaluasi yang menggambarkan seberapa akurat model dapat mengklasifikasikan dengan benar. Maka, _accuracy_ merupakan rasio prediksi benar (positif dan negatif) dengan keseluruhan data. Dengan kata lain, _accuracy_ merupakan tingkat kedekatan nilai prediksi dengan nilai aktual (sebenarnya)[3].
	* **precision** merupakan matrik evaluasi yang menggambarkan tingkat keakuratan antara data yang diminta dengan hasil prediksi yang diberikan oleh model. Maka, _precision_ merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf. Dari semua kelas positif yang telah diprediksi dengan benar, berapa banyak data yang benar-benar positif.
	* **recall** merupakan matrik evaluasi yang menggambarkan keberhasilan model dalam menemukan kembali sebuah informasi. Maka, _recall_ merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif[4].

- Dari evaluasi yang dilakukan, didapatkan hasil sebagai berikut:
 
| Nama | accuracy | precision | recall |
| ------ | ------ | ------ |------ |
| LogisticRegression | 1.0 | 1.0 | 1.0 |
| SVM | 1.0 | 1.0 | 1.0 |
| RandomForest | 1.0 | 1.0 | 1.0 |
  

**Rubrik/Kriteria Tambahan (Opsional)**:

- accuracy adalah proporsi dari instances pada train test yang diprediksi secara tepat.

accuracy  = $\frac{TP+TN}{TP+TN+FP+FN}$

- precision adalah proporsi dari _testing set_ yang diprediksi oleh model yang benar-benar positif.

precision = $\frac{TP}{TP+FP}$

- recall adalah proporsi dari data pada _testing set_ positif yang dipresidksi positif pada model.

recall = $\frac{TP}{TP+FN}$

## Kesimpulan

Dari pembahasan diatas, dapat disimpulkan sebagai berikut:
- Logistic Regression, SVM dan Random forest dapat digunakan untuk melakukan klasifikasi.
- Dalam membangun model machine learning, ada beberapa tahapan yang harus diselesaikan yaitu: _Data Understanding_, _Data Preparation_, _Modeling_ dan _Evaluation_.
- Pada proyek ini, matrik evaluasi model yang digunakan adalah _accuracy_, _precision_ dan _recall_

## Referensi
[1]  A. Wibowo, Y. Rahayu, A. Riyanto, and T. Hidayatulloh, “Classification algorithm for edible mushroom identification,” in _2018 International Conference on Information and Communications Technology (ICOIACT)_, 2018, pp. 250–253.

[2]  I. Ul Haq, I. Gondal, P. Vamplew, and S. Brown, “Categorical features transformation with compact one-hot encoder for fraud detection in distributed environment,” in _Australasian Conference on Data Mining_, 2018, pp. 69–80.

[3]  G. H. Ayres, “Evaluation of accuracy in photometric analysis,” _Anal. Chem._, vol. 21, no. 6, pp. 652–657, 1949.

[4]  D. M. W. Powers, “Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation,” _arXiv Prepr. arXiv2010.16061_, 2020.
