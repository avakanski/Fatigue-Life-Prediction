# Predictive Modeling and Uncertainty Quantification of Fatigue Life in Metal Alloys using Machine Learning

Authors: Jiang Chang, Deekshith Basvoju, Aleksandar Vakanski, Indrajit Charit, Min Xian

Recent advancements in machine learning-based methods have demonstrated great potential for improved property prediction in material science. However, reliable estimation of the confidence intervals for the predicted values remains a challenge, due to the inherent complexities in material modeling. This study introduces a novel approach for uncertainty quantification in fatigue life prediction of metal materials based on integrating knowledge from physics-based fatigue life models and machine learning models. The proposed approach employs physics-based input features estimated using the Basquin fatigue model to augment the experimentally collected data of fatigue life. Furthermore, a physics-informed loss function that enforces boundary constraints for the estimated fatigue life of considered materials is introduced for the neural network models. 

Experimental validation on datasets comprising collected data from fatigue life tests for Titanium alloys and Carbon steel alloys demonstrates the effectiveness of the proposed approach. Among the evaluated models, Bayesian Neural Networks with Markov Chain Monte Carlo (BNN-MCMC) provide the most accurate and reliable estimates. Gaussian Process Regression (GPR) also shows competitive performance. The findings highlight the potential of physics-informed machine learning methods for improved performance and uncertainty quantification in material science applications.  

## 📁 Repository Organization
This repository is organized as follows:
- **Uncertainty Quantification ML Models directory**: contains codes related to the application of machine learning methods for predicting the fatigue life of Titanium alloys and Carbon steel alloys. The objective is to calculate both single-point estimates and uncertainty estimates for the predicted fatigue life. The codes provide implementation of 8 Machine Learning methods, which include conventional Machine Learning approaches (Quantile Regression, Natural Gradient Boosting Regression, Gaussian Process Regression), Neural Networks-based approaches with deterministic network parameters (Deep Ensemble, Monte Carlo Dropout), and Neural Network-based approaches with probabilistic network parameters (Variational Inference BNNs, Markov Chain Monte Carlo BNNs). For comparison, the implementation codes for standard Neural Networks with deterministic network parameters that output single-point predictions are also provided.
- **Physics-Informed ML Models directory**: contains code for predicting fatigue life via physics-informed machine learning methods based on integrating knowledge from governing laws for fatigue modeling into data-driven approaches. Implementation codes are provided for the 8 methods listed abovea used for uncertainty quantification. 

## 📊 Data and Evaluation Metrics
The implemented methods for predicting fatigue life are evaluated on four fatigue datasets: 
- Titanium alloys dataset (from <a href="https://pubs.aip.org/aip/aml/article/1/1/016102/2878729/Machine-learning-assisted-interpretation-of-creep">Swetlana et al., 2023</a>), containing 222 samples with 24 features per sample.
- Carbon steel alloys datasets: the first dataset consists of 378 test samples from uniaxial tension-compression fatigue tests, the second dataset consists of 611 test samples from bending fatigue tests, and the third dataset has 208 test samples from torsion fatigue tests. Each dataset have 18 features per sample.

The set of used performance metrics for evaluating the implemented methods include predictive accuracy metrics (Pearson Correlation Coefficient, $R^2$ Coefficient of Determination, Root-Mean-Square Deviation, Mean Absolute Error) and uncertainty quantification metrics (Coverage, Mean Interval Width, Composite Metric).

## ▶️ Use
The codes are provided as Jupyter Notebook files. To reproduce the results, run the .ipynb files. 

## 🔨 Requirements
keras  2.6.0  
tensorflow 2.6.0  
pyro-ppl 1.8.5  
torchbnn 1.2  
torch 2.0.1  
pandas 1.5.3  
numpy 1.23.5  
scikit-learn 1.2.2  

## ✉️ Contact or Questions
<a href="https://www.webpages.uidaho.edu/vakanski/">A. Vakanski</a>, e-mail: vakanski at uidaho.edu
