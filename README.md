# Machine Learning/Deep Learning Projects

This repository is a collection of the machine learning and deep learning projects I had completed during my studies for Master of Data Science at the University of Sydney. Data used in some of these projects are not provided due to the non-disclosure agreement, but the code and ideas/reasoning I used to solve these problems are still provided for someone who has similar problems and would like to have an example for reference.


The domains that these projects involved include:

- **Fashion-MNIST image classification**
- **Time series forecasting for customer price index (CPI)**
- **Image reconstruction with non-negative matrix factorization**
- **Label-noise learning with transition matrix for image classification**

All of them were presented in `.ipynb` file, and the tools used for model construction are mostly scikit-learn, Keras, and PyTorch.


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

In this project, two Non-negative Matrix Factorization (NMF) algorithms, **$L_2$-norm based NMF and $L_{2, 1}$-norm based NMF** [[2]](#2), were implemented to reconstruct face images. The datasets used for training are ORL [[3]](#3) and Extended YaleB dataset [[4]](#4). In addition, to test the robustness of the proposed algorithms, two different types of noise, random additional noise and block-occlusion noises, were added to the images to simulate data corruption. The performance of the algorithms was evaluated and compared in terms of various metrics, including relative rconstruction error, average accuracy, and normalized mutual information (NMI). Definition of each of these metrics can be found in the corresponding code file.

The result shows that both NMF algorithms performed well on image reconstruction except for fixing the noise pattern. Overall, $L_2$-norm NMF has slightly better performance regardless of the metrics used, while $L_{2, 1}$-norm based NMF is more robust when block-occlusion noise was applied. However, the computational time of the $L_{2, 1}$-norm NMF is significantly longer than $L_2$-norm NMF.


## Label-noise Learning with Transition Matrix for Image Classification

*Collaborator: Ke Wang, Zijie Zhao*

This project aims to consturct a image classifier under label noise. Three corrupted datasets with various flip rate of class-conditional label noise were used. The flip rate of the first two datasets were known, but the one for the last dataset was not. Hence, two methods, which are anchor point assumption and dual-T estimator [[5]](#5), were applied to estimate the transition matrix between correct and noisy labels. The robustness of the two estimation methods were evaluate and compared by mean squared error by apply them to the known transtion matrics in the first two datasets. Three neural models, including FFNN, CNN, and ResNet, were then trained to handle classification tasks by applying known or estimated transition matirx. 

The result shows that both methods could reach a high accuracy on transition matrix estimation. However, the model perforamnce decreses as the flip rate increases or the complexity of the images increases, e.g., transefer from gray scale to color scale images.

## Reference

<a id="1">[1]</a> Han Xiao, Kashif Rasul, Roland Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.

<a id="2">[2]</a> Deguang Kong, Chris Ding, and Heng Huang. Robust nonnegative matrix factorization using L21-norm. *In Proceedings of the 20th ACM international conference on Information and knowledge management*, pages 673-682, 2011.

<a id="3">[3]</a> ATT Laboratories Cambridge. The database of faces. https://cam-orl.co.uk/facedatabase.html

<a id="4">[4]</a> UCSD Computer Vision. Extended yale face database b. https://paperswithcode.com/dataset/extended-yale-b-1 

<a id="5">[5]</a> Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Jiankang Deng, Gang Niu, and Masashi Sugiyama. Dual T: Reducing estimation error for transition matrix in label-noise learning. *Advances in neural information processing systems*, 33:7260-7271, 2020.
