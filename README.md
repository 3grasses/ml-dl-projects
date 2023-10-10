# Machine Learning/Deep Learning Projects

This repository is a collection of the machine learning and deep learning projects I had completed during my studies for Master of Data Science at the University of Sydney. Domains that these projects involved include:

- **Fashion-MNIST image classification**
- **Time series forecasting for customer price index (CPI)**
- **Image reconstruction with non-negative matrix factorization**
- **Label-noise learning with transition matrix for image classification**

All the code were written and presented in `.ipynb` file, and the tools I used to build the models are mostly Keras and PyTorch.


## Fashion-MNIST Image Classification

*Collaborator: Yuxuan Mu*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/215a58fd-e7b9-4599-a100-cadbc3b1e6ab" width="700">
 <br>
 <em> Fashion-MNIST dataset </em>
</p>

Image classification is one of the main branches in computer vision. In this project, three different models, including **random forest, feed-forward neural network (FFNN), and convolutional neural network (CNN)**, were built to classify the Fashion-MNIST dataset [[1]](#1). A step-by-step procedure for data preprocessing, hyperparameter tuning, model training and evaluation were presented with brief explanation. The accuracy and the runtime of the training and evaluation process for each model was also recorded for comparison.


## Time Series Forecasting for Customer Price Index (CPI)

*Collaborator: Yuxuan Mu, Diogo Melo Paes, Rafiul Nakib, William Miao*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/a53a0955-6421-446f-b6fd-654bb650bb92" width="650">
 <br>
 <em> Forecasting result </em>
</p>

This project constructed a predictive model to forecast the Customer Price Index (CPI) based on the historical data. Methods examined in this project include **time series decomposition, linear regression, exponential smoothing, and ARIMA.** Long short-term memory (LSTM) network was also considered due to the sequential nature of time series data. All the parameters were selected and fine-tuned based on the characteristics presented in the data, such as the appearance of trend component, seasonality, and its frequency. The final model was determined by considering both the accuracy in terms of mean square error, mean absolute percentage error, and AIC, as well as the stability of the forecast.


## Image Reconstruction with Non-negative Matrix Factorization

*Collaborator: Ke Wang, Zijie Zhao*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/3b615b44-20d7-41d7-ab69-63d379d2703d" width="400">
  <br>
 <em> Reconstructed images </em>
</p>

In this project, two Non-negative Matrix Factorization (NMF) algorithms, **$L_2$-norm based NMF and $L_{2, 1}$-norm based NMF** [[2]](#2), were implemented to reconstruct face images. The datasets used for training are ORL [[3]](#3) and Extended YaleB dataset [[4]](#4). In addition, to test the robustness of the proposed algorithms, two different types of noise, random additional noise and block-occlusion noises, were added to the images to simulate data corruption. The performance of the algorithms was evaluated and compared in terms of various metrics, including relative reconstruction error, average accuracy, and normalized mutual information (NMI). Definition of each of these metrics can be found in the corresponding code file.


## Label-noise Learning with Transition Matrix for Image Classification

*Collaborator: Ke Wang, Zijie Zhao*

This project aims to construct models robust to the appearance of label noise for image classification task. Three corrupted datasets with various flip rates of class-conditional label noise were used. The flip rate of the first two datasets were known, but the one for the last dataset was not. Hence, two methods, which are **anchor point assumption and dual-T estimator** [[5]](#5), were applied to estimate the transition matrix between the correct and noisy labels. The robustness of the two estimation methods were evaluated and compared in terms of mean square error by applying them to the first two datasets with known matrices, respectively. After that, three neural-based models, including FFNN, CNN, and ResNet, were then trained to perform classification task by utilizing the known or estimated transition matrix to infer the clean class posterior.


## Reference

<a id="1">[1]</a> Han Xiao, Kashif Rasul, Roland Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.

<a id="2">[2]</a> Deguang Kong, Chris Ding, and Heng Huang. Robust nonnegative matrix factorization using L21-norm. *In Proceedings of the 20th ACM international conference on Information and knowledge management*, pages 673-682, 2011.

<a id="3">[3]</a> ATT Laboratories Cambridge. The database of faces. https://cam-orl.co.uk/facedatabase.html

<a id="4">[4]</a> UCSD Computer Vision. Extended yale face database b. https://paperswithcode.com/dataset/extended-yale-b-1 

<a id="5">[5]</a> Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Jiankang Deng, Gang Niu, and Masashi Sugiyama. Dual T: Reducing estimation error for transition matrix in label-noise learning. *Advances in neural information processing systems*, 33:7260-7271, 2020.
