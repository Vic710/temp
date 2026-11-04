# Deep Learning Practical

Lab notebooks for five practical sections (20 exercises). Each notebook starts with the problem statement, then the implementation split into the exercise steps. Code follows the original write-up. Notebooks are not pre-executed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most exercises call `pd.read_csv('some_file.csv')` with a bare filename. Put those files in the working directory you launch Jupyter from (the kernel's current directory), or change the filename in the notebook to a path you prefer.

Image exercises in section 5 use a folder path (`keras.utils.image_dataset_from_directory`) instead of a CSV. Exercise 3.b expects an MVTec AD category folder. Exercise 4.c reads `document3.txt`. Exercise 3.e reads the NSL-KDD text files.

Saved Word2Vec models (`twitter_cbow.model`, `document3_cbow.model`) and dataset files are gitignored.

## Exercises

### Section 1 — Feedforward neural networks

| Exercise | Notebook | Data files |
| --- | --- | --- |
| 1.a Feedforward Neural Network for MNIST | [1a_mnist_fnn.ipynb](01_FNN/1a_mnist_fnn.ipynb) | `mnist_train.csv`<br>`mnist_test.csv` |
| 1.b Feedforward Neural Network for Fashion MNIST | [1b_fashion_mnist_fnn.ipynb](01_FNN/1b_fashion_mnist_fnn.ipynb) | `fashion-mnist_train.csv`<br>`fashion-mnist_test.csv` |
| 1.c Feedforward Neural Network for CIFAR-10 | [1c_cifar10_fnn.ipynb](01_FNN/1c_cifar10_fnn.ipynb) | `cifar-10.csv` |
| 1.d Feedforward Neural Network for Sign Language MNIST | [1d_sign_mnist_fnn.ipynb](01_FNN/1d_sign_mnist_fnn.ipynb) | `sign_mnist_train.csv`<br>`sign_mnist_test.csv` |

### Section 2 — Convolutional neural networks

| Exercise | Notebook | Data files |
| --- | --- | --- |
| 2.a CNN Image Classification | [2a_cifar10_cnn.ipynb](02_CNN/2a_cifar10_cnn.ipynb) | `cifar-10.csv` |
| 2.b CNN Image Classification for MNIST | [2b_mnist_cnn.ipynb](02_CNN/2b_mnist_cnn.ipynb) | `mnist_train.csv`<br>`mnist_test.csv` |
| 2.c CNN for Sign Language MNIST | [2c_sign_mnist_cnn.ipynb](02_CNN/2c_sign_mnist_cnn.ipynb) | `sign_mnist_train.csv`<br>`sign_mnist_test.csv` |

### Section 3 — Autoencoders and anomaly detection

| Exercise | Notebook | Data files |
| --- | --- | --- |
| 3.a Credit Card Fraud Detection with an Autoencoder | [3a_creditcard_autoencoder.ipynb](03_Autoencoders/3a_creditcard_autoencoder.ipynb) | `creditcard.csv` |
| 3.b Industrial Anomaly Detection on MVTec AD | [3b_mvtec_conv_autoencoder.ipynb](03_Autoencoders/3b_mvtec_conv_autoencoder.ipynb) | `mvtec_ad/` image folder (`train/good`, `test/good`, and other `test` subfolders; not a CSV) |
| 3.c Energy Consumption Anomaly Detection with an LSTM Autoencoder | [3c_energy_lstm_autoencoder.ipynb](03_Autoencoders/3c_energy_lstm_autoencoder.ipynb) | `energy_consumption.csv` |
| 3.d NASA Turbofan Fault Detection with an LSTM Autoencoder | [3d_turbofan_lstm_autoencoder.ipynb](03_Autoencoders/3d_turbofan_lstm_autoencoder.ipynb) | `nasa_turbofan.csv` |
| 3.e Network Intrusion Detection on NSL-KDD | [3e_nslkdd_autoencoder.ipynb](03_Autoencoders/3e_nslkdd_autoencoder.ipynb) | `KDDTrain+.txt`<br>`KDDTest+.txt` |
| 3.f Sign Language MNIST Anomaly Detection with an Autoencoder | [3f_sign_mnist_autoencoder.ipynb](03_Autoencoders/3f_sign_mnist_autoencoder.ipynb) | `sign_mnist_train.csv`<br>`sign_mnist_test.csv` |

### Section 4 — NLP and CBOW

| Exercise | Notebook | Data files |
| --- | --- | --- |
| 4.a CBOW Embeddings for IMDB Document Classification | [4a_imdb_cbow.ipynb](04_NLP_CBOW/4a_imdb_cbow.ipynb) | `IMDB Dataset.csv` |
| 4.b CBOW Word2Vec on a Twitter Dataset | [4b_twitter_cbow.ipynb](04_NLP_CBOW/4b_twitter_cbow.ipynb) | `twitter_dataset.csv` |
| 4.c CBOW Word2Vec on Textual Document 3 | [4c_document3_cbow.ipynb](04_NLP_CBOW/4c_document3_cbow.ipynb) | `document3.txt` |
| 4.d Content-based Recommendation with CBOW Embeddings | [4d_cbow_recommender.ipynb](04_NLP_CBOW/4d_cbow_recommender.ipynb) | `twitter_dataset.csv` |

### Section 5 — Transfer learning

| Exercise | Notebook | Data files |
| --- | --- | --- |
| 5.a Transfer Learning with VGG16 on Caltech-101 | [5a_caltech101_vgg16.ipynb](05_Transfer_Learning/5a_caltech101_vgg16.ipynb) | `C:\Users\Prajwal\PICT\TE SEM 7\LP3\LP IV Updated Datasets\Object Detection\caltech-101` (image folder; original Windows path) |
| 5.b Transfer Learning with VGG16 on Intel Image Classification | [5b_intel_vgg16.ipynb](05_Transfer_Learning/5b_intel_vgg16.ipynb) | `C:\Users\Prajwal\PICT\TE SEM 7\LP3\LP IV Updated Datasets\Object Detection\Intel Image Classification` (image folder; original Windows path) |
| 5.c Transfer Learning with VGG16 on Fashion MNIST | [5c_fashion_mnist_vgg16.ipynb](05_Transfer_Learning/5c_fashion_mnist_vgg16.ipynb) | `C:\Users\Prajwal\PICT\TE SEM 7\LP3\LP IV Updated Datasets\Object Detection\fashion mnist` (image folder; original Windows path) |

## Notes on the source write-up

- **2.a** asks for handwritten MNIST. The code trains on `cifar-10.csv` at 32×32×3. The notebook keeps that code and records the mismatch.
- **1.b** is Fashion MNIST. Step (b) of the statement mentions CIFAR-10; the code loads the Fashion MNIST CSVs. Left as written.
- **1.c** uses one `cifar-10.csv` (label column, rows with missing labels dropped) rather than separate train and test files.
- **3.b** had no code on the source page. The notebook is a new convolutional autoencoder in the same style as the other anomaly exercises, and it is marked as newly written.
- **5.a, 5.b, and 5.c** keep the original absolute Windows `dataset_path` values. Update those strings before running them.
