# -*- coding: utf-8 -*-
"""mushroom_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TxDGDQ2HiNyK4RVzUhkpAEZhHJzEa9m3

# Mushrooms CLassifier
Pada notebook ini saya membuat model machine learning untuk melakukan klasifikasi apakah jamur bisa dimakan atau tidak. 

## Import Library
disini saya mengimport beberapa library standar yang biasa digunakan untuk membuat model machine learning dan data analisis, seperti: `sklearn`,`numpy`,`pandas` dll. saya juga menginstall package [luwiji](https://pypi.org/project/luwiji/). `luwiji` adalah package yang dikembangkan oleh Wira D. K. Putra sebagai pelengkap kurikulum pembelajaran mesinnya bagi para peminat pembelajaran mesin muda. disini saya hanya akan mengambil hyperparameter yang didapat oleh Wira D. K. Putra.
"""

!pip install luwiji

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

"""## Tentang Dataset
Dataset ini mencakup deskripsi sampel hipotetis yang sesuai dengan 23 spesies jamur insang di Agaricus dan Jamur Keluarga Lepiota yang diambil dari The Audubon Society Field Guide to North American Mushrooms (1981).
"""

mushrooms = pd.read_csv("mushrooms.csv")
mushrooms.head()

"""## Exploratory Data Analysis - Deskripsi Variabel

### Melihat missing value pada dataset
hal pertama yang harus saya lakukan adalah melihat missing value pada dataset. sehingga didapatkan sebagai berikut:
"""

mushrooms.info()

mushrooms.shape

mushrooms.describe()

mushrooms.isna().sum()

plt.figure(figsize=(15,7))
sns.heatmap(mushrooms.isnull(), yticklabels=False, cbar=False, cmap='coolwarm');

"""dari analisa dan ploting di atas, bisa kita ketahui bahwa tidak ada missing value pada dataset yang digunakan dan semua features pada data bertipe categorical features

## Categorical Analysis
karena semua features data adalah data categorical, maka saya hanya akan melakukan analisa secara categorical
"""

mushrooms.columns

categorical_features = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                        'stalk-surface-below-ring', 'stalk-color-above-ring',
                        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                        'ring-type', 'spore-print-color', 'population', 'habitat']

"""### Categorical Features
pada tahap ini saya akan melihat persebaran data pada `Categorical_Features` dan didapat sebagai berikut:
"""

feature = categorical_features[0]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[1]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[2]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[3]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[4]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[5]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[6]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[7]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[8]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[9]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[10]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[11]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[12]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[13]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[14]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[15]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[16]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[17]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[18]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[19]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[20]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[21]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[21]
count = mushrooms[feature].value_counts()
percent = 100*mushrooms[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Dari grafik persebaran data pada masing-masing feature, ada satu features yang tidak memberikan insight yaitu `veil-type`, oleh karena itu features tersebut perlu dibuang."""

categorical_features.remove('veil-type')
categorical_features.remove('class')
mushrooms.drop(columns='veil-type', inplace=True)
categorical_features

"""## Train-Test-Split Dataset
karena ukuran data lebih besar dari 5000 maka saya menggunakan 25% untuk data test dan 75% untuk data train
"""

X = mushrooms.drop(columns='class')
y = mushrooms['class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""## Data Preparation
### Standarisasi
pada tahap standarisasi, karena features data merupakan `categorical_features` maka saya menggunakan `OneHotEncode` yang terdapat pada `sklearn`
### Pipeline
untuk mempermudah dan menyingkat code pada data preparation saya menggunakan `Pipeline` pada sklearn. dengan menggunakan Pipeline, code akan lebih singkat dan mudah dibaca.

"""

from sklearn.preprocessing import OneHotEncoder

cat_pip = Pipeline([
  ('Onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer([
  ('categoric', cat_pip, categorical_features)
])

"""## Model Development
### Model Development Dengan LogisticRegression
pada tahap development pertama, saya menggunakan algoritma LogisticRegression sebagai baseline(dasar) dari model machine learning yang saya buat. dan untuk hyperparameter saya menggunakan hyperparameter yang terdapat pada package `luwiji` tepatnya pada library `jcopml.tuning`, pada library tersebut banyak properti yang berisi berbagai hyperparameter untuk berbagai algoritma machine learning. untuk mendapatkan best parameter saya menggunakan method `GridSearchCV` pada sklearn. 
"""

from jcopml.tuning import grid_search_params as gsp
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
  ('prep', preprocessor),
  ('algo', LogisticRegression())
])

LogisticR = GridSearchCV(pipeline, gsp.linreg_params, cv=3, n_jobs=-1, verbose=1)
LogisticR.fit(X_train, y_train)

"""
### Model Development dengan SVM
untuk model machine learning menggunakan algoritma `SVM Classifier`. saya juga menggunakan hyperparameter yang terdapat pada pakcage `luwiji` dan untuk mendapatkan best parameter saya menggunakan `GridSearchCV`. """

from sklearn.svm import SVC

pipeline = Pipeline([
  ('prep', preprocessor),
  ('algo', SVC(max_iter=500))
])

svm = GridSearchCV(pipeline, gsp.svm_params, cv=3, n_jobs=-1, verbose=1)
svm.fit(X_train, y_train)

"""### Model Development Dengan RandomForest
pada random forest ini untuk mendapatkan best parameter saya menggunakan method `RandomizedSearchCV`. alasannya karena menggunakan `GridsearchCV` proses train terlalu lama, dan untuk mempercepat proses train maka saya menggunakan method lain yaitu `RandomizedSearchCV`.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

pipeline = Pipeline([
  ('prep', preprocessor),
  ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
])

RF = RandomizedSearchCV(pipeline, rsp.rf_params,n_iter=50, verbose=1, cv=10, n_jobs=-1)
RF.fit(X_train, y_train)

"""Setelah melakukan train, selanjutnya menampilkan hasil score yang diperoleh oleh masing-masing model"""

model_score = pd.DataFrame(columns=['Train_score'], index=['SVM','LogisticRegression','RandomForest'])
 
# membuat dictionary untuk setiap algoritma yang digunakan
model_dict = {'SVM': svm,'LogisticRegression': LogisticR,'RandomForest':RF}


for name, model in model_dict.items():
    model_score.loc[name, 'Train_score'] = model.score(X_train,y_train)
 
model_score

"""## Evaluasi Model

Untuk evaluasi model, saya menggunakan `accuracy_score`, `precision_score` dan `recall_score` yang terdapat pada `sklearn`
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score

model_evaluate = pd.DataFrame(columns=['acc', 'Precision', 'Recall'], index=['SVM','LogisticRegression','RandomForest'])
 
# membuat dictionary untuk setiap algoritma yang digunakan
model_dict = {'SVM': svm,'LogisticRegression': LogisticR,'RandomForest':RF}
 
# menghitung accuracy, precision dan recall masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    y_pred = model.predict(X_test)
    model_evaluate.loc[name, 'acc'] = accuracy_score(y_test, y_pred)
    model_evaluate.loc[name, 'Precision'] = precision_score(y_test, y_pred, average="binary", pos_label="e")
    model_evaluate.loc[name, 'Recall'] = recall_score(y_test, y_pred, average="binary", pos_label="e")
 
model_evaluate

"""dapat dilihat hasilnya cukup baik walaupun hanya menggunakan model LogisticRegression. selanjutnya saya akan menguji model dan didapat hasil sebagai berikut:"""

prediksi = X_test.iloc[:10].copy()
pred_dict = {'y_true':y_test[:10]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi)
 
pd.DataFrame(pred_dict)