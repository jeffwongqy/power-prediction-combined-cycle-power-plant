# Power Prediction in Combined Cycle Power Plant

<img width="796" height="400" alt="dataset-cover" src="https://github.com/user-attachments/assets/55806426-2fe7-4f21-9363-7f6aaf6ed0c3" />

# Introduction 
Maximizing the output of a Combined Cycle Power Plant (CCPP) is critical for meeting grid demands efficiently and minimizing environmental impact. Plant performance - specifically the Hourly Net Electrical Energy Output (PE) - is highly dependent on ambient environmental factors. Because these thermodynamic variables interact in complex, non-linear ways, traditional static physics equations can be challenging to deploy in real-time. 

This micro-project leverages data-driven predictive modeling to address this challenge. Utilizing the Kaggle CCPP dataset, the project establishes a robust end-to-end machine learning pipeline. It implements rigorous data preprocessing—including duplicate handling, structural verification, categorical label encoding, and Interquartile Range (IQR) outlier removal. It then evaluates a diverse suite of predictive architectures, ranging from baseline regularized linear models (Ridge, Lasso, ElasticNet) and non-parametric tree ensembles (Decision Trees, Random Forests) to an optimized Feed-Forward Neural Network (FFNN).

# Objectives
