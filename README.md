# Power Prediction in Combined Cycle Power Plant

<img width="796" height="400" alt="dataset-cover" src="https://github.com/user-attachments/assets/55806426-2fe7-4f21-9363-7f6aaf6ed0c3" />

## 1. Introduction 
Maximizing the output of a Combined Cycle Power Plant (CCPP) is critical for meeting grid demands efficiently and minimizing environmental impact. Plant performance - specifically the Hourly Net Electrical Energy Output (PE) - is highly dependent on ambient environmental factors. Because these thermodynamic variables interact in complex, non-<img width="1157" height="475" alt="outlier" src="https://github.com/user-attachments/assets/2037551e-3a11-4246-b240-bc37b5080ba0" />
linear ways, traditional static physics equations can be challenging to deploy in real-time. 

This micro-project leverages data-driven predictive modeling to address this challenge. Utilizing the Kaggle CCPP dataset, the project establishes a robust end-to-end machine learning pipeline. It implements rigorous data preprocessing—including duplicate handling, structural verification, categorical label encoding, and Interquartile Range (IQR) outlier removal. It then evaluates a diverse suite of predictive architectures, ranging from baseline regularized linear models (Ridge, Lasso, ElasticNet) and non-parametric tree ensembles (Decision Trees, Random Forests) to an optimized Feed-Forward Neural Network (FFNN).

## 2. Objectives
- Clean the CCPP dataset by systematically identifying duplicates, ensuring data integrity, isolating numerical features via boxplots, and filtering statistical anomalies using the Interquartile Range (IQR) method.
- Conduct a Pearson correlation analysis to calculate exact coefficients and p-values, statistically proving the linear relationships and significance between environmental variables ($AT$, $V$, $AP$, $RH$) and the energy target ($PE$).
- Systematically tune and optimize hyperparameter grids across 5-fold cross-validation to prevent overfitting and ensure robust generalization.
- Evaluate the optimized models on a dedicated testing holdout set using comprehensive regression metrics ($R^2$, MAE, MSE, and RMSE) and visualize the final performance hierarchy using a comparative bar plot.

## 3. Data Loading and Structural Verification 
- `power_df.head()`: Displays the first five rows of the dataset to verify the alignment of columns, inspect data shapes, and confirm that features like Ambient Temperature ($AT$), Exhaust Vacuum ($V$), Ambient Pressure ($AP$), Relative Humidity ($RH$), and Net Electrical Output ($PE$) are formatted properly.
- `power_df.info()`: Provides a high-level summary of the dataframe's schema, including the total row count, column data types (e.g., float64, object), and memory usage. This step highlights whether any categorical variables exist that will require downstream structural conversion or encoding.
- Missing Value & Duplicate Audit (`isnull().sum() / duplicated().sum()`): Computes the exact frequency of null values and duplicate rows across the matrix. This acts as a diagnostic gatekeeper; identifying these anomalies early ensures that subsequent preprocessing steps—such as row elimination or data imputation—are precisely targeted to prevent downstream model bias or data leakage.

## 4. Statistical Analysis 
### 4.1 Outlier Detection 
- Looking at both the first and second images, the distributions for $AT$ and $V$ remain identical. No data points sit outside the lower or upper whiskers in either plot, confirming that these two features were completely free of statistical outliers from the start.
- Ambient Pressure ($AP$): In the first image, the $AP$ plot shows a heavy cluster of black flier points (outliers) sitting both above the upper whisker (around $> 1030$ mbar) and below the lower whisker (around $< 998$ mbar). In the second image, after the IQR filter is applied, the bottom anomalies are eliminated, and the top cluster is reduced to just a couple of borderline points, significantly tightening the feature's variance.
- Relative Humidity ($RH$): The initial $RH$ plot shows a few minor outlier points trailing below the lower whisker (around $< 30\%$). In the post-IQR plot, these trailing minimum anomalies are successfully purged, extending the lower whisker cleanly to the newly calculated bound.

<img width="1157" height="475" alt="outlier" src="https://github.com/user-attachments/assets/924656e9-0442-4ccb-bbf5-4f912ae0d30a" />



### 4.2 Linear Association 

