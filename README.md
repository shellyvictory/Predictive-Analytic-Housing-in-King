# Laporan Proyek Machine Learning - [Shelly Victory](https://www.linkedin.com/in/shellyvictory/)

## Domain Proyek
Peningkatan harga properti terutama di kota metropolitan mengakibatkan banyaknya masyarakat dengan pendapatan rendah mengalami kesulitan untuk memiliki tempat tinggal yang layak. Hal ini membuat bisnis properti merupakan hal yang amat menjanjikan namun juga rentan mengalami kerugian apabila tidak mampu memahami pasar. Pada model algoritma *machine learning* yang dikembangkan oleh [Hu et al.](https://www.sciencedirect.com/science/article/abs/pii/S0264837718316429) (2019), didapat fakta bahwa lingkungan perumahan mengakibatkan harga sewa yang lebih tinggi dibandingkan dengan lokasi. *Random forest regression* dan *extra-trees regression* merupakan 2 model terbaik dengan tingkat akurasi tinggi.

## *Business Understanding*
Pemahaman pasar properti merupakan kunci untuk menjalankan bisnis penjualan properti seperti rumah dan aparatemen yang untung. Agar tujuan tersebut tercapai, analisis yang mumpuni harus dilakukan agar tidak merugikan pihak-pihak yang terlibat. Walaupun terjadi peningkatan kebutuhan akan tempat tinggal, pengusaha properti harus tetap memiliki daya saing tinggi sehingga usahanya dapat terus berjalan. Harga rumah yang terlalu mahal rentan dengan resiko penjualan tidak mencapai target dan begitu pula sebaliknya, sehingga kualitas fitur rumah harus sebanding dengan harga jual sekaligus memenuhi permintaan pasar.

### *Problem Statements*
Pada analisis ini, faktor yang harus diperhatikan meliputi: 
1. Fitur apa saja yang terdapat di dalam *dataset* penjualan rumah di Kota King dan fitur manakah yang paling mempengaruhi harga rumah?
2. Metode apakah yang menghasilkan performa terbaik dan bagaimana cara memperolehnya?

### *Goals*
*Goals* yang ingin dicapai pada *predictive analysis* ini yaitu:
1. Mengetahui fitur-fitur yang terdapat di dalam *dataset* dan fitur apakah yang paling mempengaruhi harga penjualan rumah.
2. Mengidentifikasi model algoritma *machine learning* dengan tingkat performa terbaik.

### *Solution statements*
Pada permasalahan yang telah dijelaskan sebelumnya, akan digunakan tiga jenis algoritma *machine learning* sebagai solusi yaitu K-Nearest Neighborhood, Random Forest, dan Boosting Algorithm. Penggunaan ketiga jenis model algoritma adalah untuk membandingkan performanya untuk mendapat model yang terbaik. Adapun 3 model algoritma yang digunakan adalah:
1. K-Nearest Neighbor: 
KNN merupakan algoritma yang memanfaatkan kesamaan fitur antar sampel pelatihan untuk prediksi nilai dari tiap data yang baru. Metode ini memiliki kelebihan yaitu lebih sederhana serta dapat digunakan untuk kasus klasifikasi dan regresi. Akan tetapi, KNN tidak dapat digunakan untuk menganalisis dimensi yang besar.
2. Random forest:
Merupakan model prediksi yang memiliki lebih dari satu model dan bekerja pada saat yang bersamaan. Model ini memiliki kesamaan dengan KNN yaitu bersifat sederhana dan dapat diaplikasikan untuk klasifikasi serta regresi. Perbedaannya yaitu model ini memiliki stabilitas yang lebih mumpuni.
3. Boosting algorithm
Model prediksi ini merupakan *model ensemble* learning teknik *boosting*. Perbedaannya dengan teknik RF yaitu pelatihan model dilakukan secara berurutan. Algoritma ini cukup populer dikarenakan memiliki kemampuan meningkatkan akurasi yang lebih baik.

## Data Understanding
Pada analisis ini, dataset yang digunakan merupakan [Penjualan Rumah di Kota King, USA](https://www.kaggle.com/harlfoxem/housesalesprediction). *Dataset* memiliki beberapa fitur seperti:
1. id: Id unik penjualan masing-masing rumah
2. date: tanggal terjualnya rumah
3. price: harga rumah
4. bedrooms: jumlah kamar tidur
5. bathrooms: jumlah kamar mandi
6. sqft_living: luas ruang tamu dalam satuan kaki persegi
7. sqft_lot: luas tanah dalam satuan kaki persegi
8. floors: jumlah lantai
9. waterfront: Apakah apartemen menghadap kawasan tepi air
10. view: tingkat kebagusan pemandangan dalam skala 1-4
11. condition: kualitas kondisi rumah dalam skala 1-5
12. grade: kualitas konstruksi dan desain rumah dalam skala 1-13.
13. sqft_above: luas kaki kuadrat rumah di atas permukaan tanah
14. sqft_basement: luas kaki kuadrat rumah di bawah permukaan tanah
15. yr_built: tahun rumah mulai dibangun
16. yr_renovated: tahun terakhir rumah direnovasi
17. zipcode: kode pos
18. lat: garis lintang
19. long: garis bujur
20. sqft_living15: luas kaki kuadrat area perumahan 15 tetangga terdekat
21. sqft_lot15: luas kaki kuadrat tanah 15 tetangga terdekat

Pada saat dilakukan pemeriksaan kelengkapan *dataset*, tidak terdapat data yang hilang dan penamaan fitur telah sesuai dengan keterangannya. Tahap selanjutnya yang dilakukan adalah menjabarkan informasi statistik *dataset*. Informasi statistik *dataset* meliputi count, mean, std, min, 25%, 50%, 75% dan max.

### Identifikasi *Missing Values*
Agar *dataset* dapat menghasilkan nilai performa optimal, identifikasi nilai yang eror dilakukan dan diperoleh *dataset* tidak memiliki nilai yang eror maupun hilang. Selanjutnya dilakukan penghapusan fitur id karena memiliki nilai yang unik dan tidak mempengaruhi harga penjualan rumah. Fitur tanggal yang pada awalnya memiliki tipe data object dibuah diurai menjadi tahun, bulan, dan tanggal sehingga menjadi tipe data numerik.

### Menangani *Outliers*
*Outliers* merupakan nilai yang sangat jauh dari rentang atau pola data lainnya sehingga perlu diatasi. Pada kasus ini, tiap fitur numerik data divusualisasikan *outlier*nya lalu dibersihkan dengan teknik *Inter Quartile Range*.

### *Univariate* dan *Multivariate Analysis*
Analisis fitur tunggal numerik pada *dataset* dilakukan dengan menggunakan histogram.
Sementara itu, analisis antar 2 fitur atau lebih dilakukan dengan menggunakan *pairplot* dan visualisasi skor evaluasinya. Dari hasil analisis tersebut, diperoleh bahwa fitur 'sqft_living', 'grade', 'sqft_above', 'lat', dan 'sqft_living15' memiliki korelasi yang lebih tinggi dengan harga dibandingkan fitur lainnya. Beberapa fitur yang hampir tidak memiliki korelasi dengan harga (skor < 0.1) dihilangkan.

## Data Preparation
Persiapan data dilakukan dengan beberapa teknik yaitu 
1. Reduksi dengan Metode *Principal Component Analysis* Mengurangi reduksi, mengekstraksi fitur, dan melakukan transformasi data ke dalam dimensi baru yang lebih kecil. 
2. Pembagian *Dataset*
Sebelum membuat model, pembagian *dataset* merupakan hal yang wajib dilakukan. Pada proyek ini digunakan teknik train-test-split dengan pembagian 90:10.
3. Standarisasi Data
Merupakan persiapan data dengan mentransformasinya menjadi data dengan skala yang relatif sama. Teknik yang digunakan yaitu StandarScaler dari *library* Scikitlearn.

## Modeling
Guna memperoleh model *machine learning* dengan hasil prediksi terbaik, digunakan 3 jenis algoritma, yaitu:
1. K-Nearest Neighbor
Pada kasus ini, prediksi dibuat dengan nilai k = 10. Pemilihan nilai k yang besar dilakukan untuk menghindari *overfit*. Jarak antar titik menggunakan metrik Euclidean. Selanjutnya data *training* dilatih dan pada tahap evaluasi menggunakan data *testing*.

2. Random Forest
Random Forest merupakan algoritma supervised learning yang termasuk ke dalam model ensemble. Pada kasus regresi prediksi akhir RF merupakan rata-rata prediksi seluruh pohon dalam model ensemble. Pada kasus ini, algoritma ditetapkan pada *dataset* dengan library scikit-learn. Parameter yang digunakan antara lain 
a. n_estimator : jumlah *trees* di dalam *forest*. Pada pemodelan digunakan sebanyak 50.
b. max_depth: kedalaman pohon maksmimum. Digunakan sebanyak 16.
c. random_state: mengendalikan tingkat keacakan dan *bootstrapping* pada sampel ketika membangun pohon. Jumlah *random state* yang diterapkan adalah 55.
d. n_jobs: jumlah *jobs* yang di *run* dalam paralel. Digunakan sebanyak 1 atau none.


3. Boosting Algorithm.
Pada *boosting algorithm*, model dilatih secara berurutan dengan cara kerja membangun model dari data latih. Pada kasus ini, digunakan motode *adapative boosting* yaitu AdaBoost. Pada model ini, bobot dengan nilai lebih tinggi akan diberikan pada model yang salah sehingga akan masuk ke tahap selanjutnya hingga proses model mencapai akurasi yang diinginkan. Parameter n_estimator dan random_state merupakan parameter yang sama dengan model random forest, yang membedakan adalah *learning _rate*. learning_rate merupakan nilai koreksi *weight* saat proses *training*.

## Evaluasi
Pada analisis dilakukan evaluasi model dengan metrik *Mean Squared Error* (MSE). MSE merupakan metrik yang menghitung perbedaan nilai sebenarnya dengan nilai selisih. Semakin kecil nilai MSE, maka eror yang dihasilkan model akan semakin kecil pula. ![Image of MSE Formula](https://user-images.githubusercontent.com/89523435/175800241-379a912e-e6bf-45b0-84e4-3bd5a786a608.png). Metrik ini memiliki kelebihan yaitu lebih sederhana karena berfokus dengan nilai selsisih dan umum digunakan untuk model regresi. Akan tetapi, metrik ini akan menghasilkan nilai eror yang tinggi apabila memiliki *outliers*.
Pada tahap evaluasi, model telah dilatih dengan algoritma KNN, RF, dan Boosting Algorithm. Proses *scaling* fitur numerik pada data uji dilakukan sebelum menghitung nilai MSE pada model unuk menghindari kebocoran data. 
 Model | train | Test|
------------ | ------------- | -----------|
KNN | 1.5734e+07 | 1.91459e+07
RF | 5.56765e+06 | 1.8288e+07
Boosting | 2.12881e+07 | 2.14908e+07

Setelah dievaluasi dengan MSE, diperoleh hasil bahwa algorima KNN memiliki nilai eror yang paling kecil pada data latih dan algoritma RF menghasilkan nilai eror terkecil pada data uji. 

y_true |	prediksi_KNN |	prediksi_RF	| prediksi_Boosting
------------ | ------------- | -----------| -----------|
472500.0 |	338937.0|	311840.2|	500534.4|

Selanjutnya, buat prediksi dengan beberapa harga dari data uji. Berdasarkan hasil yang diperoleh, algoritma KNN menghasilkan nilai yang paling mendekati nlai aslinya.  

**---Ini adalah bagian akhir laporan---**
