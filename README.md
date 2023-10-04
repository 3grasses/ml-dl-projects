# Machine Learning/Deep Learning Projects

**NOTE: This repository is only for SHARING. Please DO NOT copy my code in case of PLARGARISM.**

This repository is a collection of the machine learning and deep learning projects I had completed during my studies for Master of Data Science at the University of Sydney. Data used in some of these projects are not provided due to non-disclosure agreement, but I provided my code with some explanations in case someone would like to have a reference to solve similar problems.

The domains that these projects involved include:

- **Fashion-MNIST Image Classification**
- **Time Series Forecasting for Customer Price Index (CPI)**
- **Image Reconstruction with Non-negative Matrix Factorization**
- **Label-noise Learning with Transition Matrix for Image Classification**

All of them were done by jupyter notebook, and the tools I used are mostly scikit-learn, Keras, and PyTorch.



## Fashion-MNIST Image Classification

*Collaborator: Yuxuan Mu*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/e9fc5bfb-d5b1-4cf5-8a53-121e684f7a39" width="750">
 <br>
 <em> Fashion-MNIST dataset </em>
</p>

In this project, three different models, including Random Forest, Feed-forward Neural Network (FFNN), and Convolutional Neural Netowrk (CNN), were constructed to classify the Fashion-MNIST dataset. The entire procedure including data preprocessing, hyperparameter tuning, training and evaluation were presented in details. The runtime for training and evaluation was also recorded for comparison. The hieghest accuracy was 91% achieved by CNN.


## Time Series Forecasting for Customer Price Index (CPI)

*Collaborator: Yuxun Mu, Diogo Melo Paes, Rafiul Nakib, William Miao*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/95a13176-838e-44e8-93c0-08aead0b47ab" width="700">
 <br>
 <em> Forecasting result </em>
</p>

This project used various forecasting methods to forecast the Customer Price Index in the future based on historical data. Methods used in this project include time series decomposition, linear regression, exponential smoothing, ARIMA, and Long Short-term Memory (LSTM). Parameters were fine tuned based on the characteristics presented in the data, such as trend and seasonality. The final model was selected with a combination of exponential smoothing and ARIMA in order to reach a balance between accuracy and stability.

## Image Reconstruction with Non-negative Matrix Factorization

*Collaborator: Ke Wang, Zijie Zhao*

<br>

<p align="center">
 <img src="https://github.com/3grasses/ml-dl-projects/assets/146526540/0ce68668-0ffc-41b2-a386-d09efda88334" width="450">
  <br>
 <em> Reconstructed images </em>
</p>

In this project, two Non-negative Matrix Factorization (NMF) algorithms, which is $L_2$-norm based NMF and $L_{2, 1}$-norm based NMF, were implemented to reconstruct face images corrupted with noise. The image dataset used in this prject are ORL and Extended YaleB dataset. The images were first processed to add two different types of noise, random additional noise and block-occlusion noises, and then fed into the algorithms. Moreover, three metrics, including relative rconstruction errors (RRE), average accuracy, and normalized mutual information (NMI), were used to evaluate the performance and the robustness of the algorithms on the two datasets. More infromation of the definition of these metrics can be found in the corresponding code file.

## Label-noise Learning with Transition Matrix for Image Classification

*Collaborator: Ke Wang, Zijie Zhao*



## Reference

[1] Fashion-MNIST
[2] ORL
[3] Extended YaleB dataset2.
