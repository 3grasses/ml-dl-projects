# Machine Learning/Deep Learning Projects

This repository is a collection of the machine learning and deep learning projects I had completed during my studies for Master of Data Science at the University of Sydney. Data used in some of these projects are not provided due to the non-disclosure agreement, but I still provid the code and ideas/reasoning I used to solve theses problems in case someone has similar problems and would like to have an example for reference.

The domains that these projects involved include:

- **Fashion-MNIST image classification**
- **Time series forecasting for customer price index (CPI)**
- **Image reconstruction with non-negative matrix factorization**
- **Label-noise learning with transition matrix for image classification**

All of the code were provided in `.ipynb` file, and the tools used to build the models are mostly scikit-learn, Keras, and PyTorch.


## Fashion-MNIST Image Classification

*Collaborator: Yuxuan Mu*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/e9fc5bfb-d5b1-4cf5-8a53-121e684f7a39" width="750">
 <br>
 <em> Fashion-MNIST dataset </em>
</p>

Image classification is one of the main branches in computer vision. In this project, three different models, including **random forest, feed-forward neural network (FFNN), and convolutional neural network (CNN)**, were built to classify the Fashion-MNIST dataset [[1]](#1). A step-by-step procedure for data preprocessing, hyperparameter tuning, model training and evaluation were presented with brief explanation. The runtime of the training and evaluation process for each model was also recorded for comparison.

The result shows that the highest accuracy is 90.72% achieved by CNN. However, the runtime is almost 4.5 times longer than FFNN, which shows a trade-off between the accuracy and training efficiency in this case.


## Time Series Forecasting for Customer Price Index (CPI)

*Collaborator: Yuxuan Mu, Diogo Melo Paes, Rafiul Nakib, William Miao*

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/95a13176-838e-44e8-93c0-08aead0b47ab" width="700">
 <br>
 <em> Forecasting result </em>
</p>

This project constructed a predictive model to forecast the Customer Price Index (CPI) based on historical data. Forecasting methods examined in this project include **time series decomposition, linear regression, exponential smoothing, and ARIMA.** Long short-term memory (LSTM) network was also considered due to the sequential nature of time series data. ***Parameters were also selected and fine-tuned based on the characteristics presented in the data, such as the appearance of trend, seasonality, and its frequency.*** The final model was constructed by combining the forecasts of exponential smoothing and ARIMA in order to reach a balance between accuracy and stability.


## Image Reconstruction with Non-negative Matrix Factorization

*Collaborator: Ke Wang, Zijie Zhao*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/0ce68668-0ffc-41b2-a386-d09efda88334" width="450">
  <br>
 <em> Reconstructed images </em>
</p>

In this project, two Non-negative Matrix Factorization (NMF) algorithms, **$L_2$-norm based NMF and $L_{2, 1}$-norm based NMF**, were implemented to reconstruct face images. The image datasets used are ORL [[2]](#2) and Extended YaleB dataset [[3]](#3). In addition, to test the robustness of the proposed algorithms, two different types of noise, random additional noise and block-occlusion noises, were added to the images to simulate possible corruption of data. There are three metrics used to evaluate the performance of the algorithms: relative rconstruction error, average accuracy, and normalized mutual information (NMI). More infromation of the definition of each of these metrics can be found in the corresponding code file.

The final result indicates that $L_{2, 1}$-norm based NMF has a better robustness on average due to the way it defines the loss function.

## Label-noise Learning with Transition Matrix for Image Classification

*Collaborator: Ke Wang, Zijie Zhao*

The goal of this project is to construct a model robust to label noise for image classification tasks, i.e., the classifier can still perform well even when there are some of the examples mislabelled. Three corrupted datasets with various flip rate of class-conditional label noise were used. The flip rate of the first two datasets were known, but the one for the last dataset was not. To solve this problem, transition matrix estimator was used. The transition matrix was estimated by two different methods, which are anchor point assumption and dual-T estimator [5], and the estimated results were compared for their robustness.

## Reference

<a id="1">[1]</a> Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747

<a id="2">[2]</a> ORL

<a id="3">[3]</a> Extended YaleB dataset.

[4] Deguang Kong, Chris Ding, and Heng Huang. Robust nonnegative matrix factorization using L21-norm. *In Proceedings of the 20th ACM international conference on Information and knowledge management*, pages 673–682, 2011.

[5] Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Jiankang Deng, Gang Niu, and Masashi Sugiyama. Dual T: Reducing estimation error for transition matrix in label-noise learning. *Advances in neural information processing systems*, 33:7260–7271, 2020.
