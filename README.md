# Power Prediction in Combined Cycle Power Plant

<img width="796" height="400" alt="dataset-cover" src="https://github.com/user-attachments/assets/55806426-2fe7-4f21-9363-7f6aaf6ed0c3" />

## 1. Introduction 
Maximizing the output of a Combined Cycle Power Plant (CCPP) is critical for meeting grid demands efficiently and minimizing environmental impact. Plant performance - specifically the Hourly Net Electrical Energy Output (PE) - is highly dependent on ambient environmental factors. Because these thermodynamic variables interact in complex, non-linear ways, traditional static physics equations can be challenging to deploy in real-time. 

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



### 4.2 Feature Association via Pearson Correlation and P-Values

<img width="552" height="82" alt="pearson_correlation" src="https://github.com/user-attachments/assets/0676490d-c6f1-4a1e-aae7-20005faaf24f" />

- The results reveal that $AT$ ($r = -0.9459$) and $V$ ($r = -0.8667$) share an extremely strong negative linear relationship with energy output, meaning power production drops sharply as temperature and vacuum levels rise. Conversely, $AP$ ($r = 0.5229$) and $RH$ ($r = 0.3868$) display moderate positive linear associations.
- P-values: Testing against the null hypothesis ($H_0$: no linear correlation exists), all features yield a p-value of $0.0000$ ($\alpha < 0.05$). This confirms that these observed linear relationships are highly statistically significant and completely non-random.
- Pairplot visually solidifies the strong linear paths discovered during the Pearson calculations, proving that the dataset is highly structured and well-suited for linear, ensemble, and neural network regression modeling.

<img width="1231" height="1231" alt="pairplot" src="https://github.com/user-attachments/assets/582473ea-4bc5-4d58-adaf-e421f5ac4c26" />

## 5. Data Splitting
The dataset is partitioned by isolating the target variable ($PE$) from the environmental features ($AT$, $V$, $AP$, $RH$). Using train_test_split, the data is divided into an $80\%$ training set to fit the models and a $20\%$ testing holdout set, ensuring a clean, unbiased evaluation of generalization performance.

## 6. Data Scaling
Because the features operate on entirely different physical scales (e.g., Temperature vs. Ambient Pressure around $1000+$ mbar), StandardScaler is applied. It normalizes features to a mean of $0$ and a standard deviation of $1$, preventing scale-dominant features from distorting regularized regressions and optimizing neural network gradient descent.

## 7. Model Comparison 
Models undergo automated hyperparameter grid search with 5-fold cross-validation to maximize negative MSE. Optimized estimators are then tested on a holdout split. Performance is rigorously benchmarked using $R^2$, MAE, MSE, and RMSE metrics, and the results are visualized via comparative bar plots to identify the most accurate architecture.

<img width="972" height="240" alt="Screenshot 2026-06-10 213725" src="https://github.com/user-attachments/assets/01b75343-8903-4a2e-9c78-54ec34f0fe89" />

<img width="846" height="557" alt="r2" src="https://github.com/user-attachments/assets/e4652ce8-f2ec-4044-8321-707c442b5e1f" />

<img width="846" height="557" alt="mae" src="https://github.com/user-attachments/assets/0857d643-1193-46a0-972e-6de50d1f6925" />

<img width="841" height="557" alt="mse" src="https://github.com/user-attachments/assets/7bef1687-fc02-4f8b-8fb9-7c54bddcab6f" />

<img width="833" height="557" alt="rmse" src="https://github.com/user-attachments/assets/22e2241e-8d50-42d2-9245-8aa33272f930" />




## 8. Future Work
Future iterations of this project can expand predictive performance by incorporating advanced gradient-boosting tree architectures, such as XGBoost, LightGBM, or CatBoost, which often outperform standard decision trees on tabular datasets. Additionally, exploring complex feature engineering—such as interaction terms between ambient temperature and relative humidity ($AT \times RH$) or polynomial features—could help expose underlying thermodynamic relationships. Finally, expanding the hyperparameter grid search for the Feed-Forward Neural Network (FFNN) to include dropout regularization, batch normalization, and alternative learning rate schedulers could unlock deeper pattern recognition and further minimize prediction errors.


## 9. Conclusion 
This micro-project successfully established a robust end-to-end machine learning regression pipeline to predict the Hourly Net Electrical Energy Output ($PE$) of a Combined Cycle Power Plant. Rigorous preprocessing—including duplicate handling and Interquartile Range (IQR) filtering—effectively stabilized feature variance by removing atmospheric anomalies. Statistical evaluation via Pearson correlation confirmed that Ambient Temperature ($AT$) and Exhaust Vacuum ($V$) exert dominant negative linear constraints on energy production, while Ambient Pressure ($AP$) and Relative Humidity ($RH$) exhibit positive linear trends. Among the optimized models, the non-parametric Decision Tree Regressor achieved the highest predictive performance, yielding an $R^2$ of $0.9490$ and a sharply minimized Mean Absolute Error (MAE) of $2.6597$. This outperformance over regularized linear models (Ridge, Lasso, and Elastic Net) highlights the presence of complex, non-linear thermodynamic interactions between environmental inputs and plant efficiency.

## 20. References
[1] Rao, S. U. M., Kumar, A. V. S. P., Neelima, S., Krishna, C. V. M., & Lakshmanarao, A. (2024). Power Prediction in Combined Cycle Power Plant through ML and DL Regression Techniques. 2024 International Conference on Cognitive Robotics and Intelligent Systems (ICC - ROBINS), 58–62. https://doi.org/10.1109/icc-robins60238.2024.10533904

‌

