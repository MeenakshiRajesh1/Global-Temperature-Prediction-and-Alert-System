# Global-Temperature anomaly-Prediction-and-Alert-System

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#objectives">Objectives</a>
    </li>
    <li>
      <a href="#data">Data</a>
    </li>
    <li>
      <a href="#exploratory-data-analysis">Exploratory Data Analysis</a>
    </li>
    <li>
      <a href="#pre-processing">Pre-Processing</a>
    </li>
    <li>
      <a href="#models">Models</a>
    </li>
    <li>
      <a href="#Result">Result</a>
    </li>
    <li>
      <a href="#conclusion">Conclusion</a>
    </li>
  </ol>
</details>


## Objectives

- **Data Exploration and Visualization**: Explore and visualize the dataset.
- **Data Pre-processing**: Build a pipeline to pre-process the dataset.
- **Model Training**: Train and customize a viable model.
- **Collaboration**: Use GitHub and Google Colab for a collaborative research environment.



## Built With
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
* ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
* ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)



## Data
<!-- content -->
The dataset used for this project is the GISS Surface Temperature Analysis (GISTEMP v4), which provides an estimate of global surface temperature changes, including monthly and annual temperature anomalies. This data integrates reports from NOAA GHCN v4 (meteorological stations) and ERSST v5 (ocean areas) and is updated monthly by NASA's Goddard Institute for Space Studies (GISS). Spanning from 1880 to the present, it serves as a reliable and comprehensive resource for analysing long-term temperature trends and climate change impacts globally. The dataset was collected to monitor and understand global temperature variations, contributing significantly to climate change research. By combining land and ocean surface temperature readings corrected for seasonal and geographical anomalies, this dataset is essential for climate modelling and policy decision-making. It was chosen for its reliability, extensive temporal coverage, and relevance to our research question, which focuses on global temperature trends and anomalies.



## Exploratory Data Analysis
<!-- content -->
To begin, I used the .info() method to examine the structure of the data, checking for missing values and data types. Through the describe() function, I reviewed summary statistics for the numerical columns, which gave me insights into the spread and distribution of the data.
I also checked for irregularities such as symbols like ?, #, or other non-numeric characters in the dataset, which were replaced with NaN. This helped clean up the dataset before further analysis. Using a heatmap and bar plots, I visualized the distribution of missing data across the columns. I found that several columns had missing values, particularly in temperature data. To address this, I chose to impute missing numerical values with the median of each column, ensuring that the data remained consistent while minimizing bias from the imputation.
To explore the distribution of numerical features, I used histograms, which highlighted the spread of variables such as temperature anomalies, and boxplots to detect outliers. A correlation heatmap helped reveal any strong relationships between variables. For example, I noticed that certain temperature anomalies exhibited a clear trend over the years, which I then visualized with a line plot. I calculated a 5-year rolling mean of global temperature anomalies to smooth out short-term fluctuations and reveal long-term trends. Additionally, I transformed the monthly temperature anomalies into a format suitable for analysis, and plotted seasonal variations using a boxplot, which helped to identify seasonal patterns in temperature fluctuations.




## Pre-Processing
<!-- content -->
Preprocessing involves converting raw data into a clean format, which is essential for enhancing the performance and accuracy of machine learning models. Three preprocessing methods were used in this analysis: Encoding, Scaling, and Imputation 

The data preprocessing involved several steps. Categorical variables were converted into numerical values using label encoding, transforming categories such as 'Nov' and 'Dec' into numerical representations. 

StandardScaler was applied to normalize the data, enhancing model efficiency. This standardization process ensures that each feature has a mean of zero and a standard deviation of one, allowing them to contribute equally to the model’s performance. For numerical values, median imputation was used because it is less affected by outliers compared to the mean. 




## Models
<!-- content -->
**LSTM**
The optimization of the LSTM and Prophet models focused on enhancing their ability to predict annual temperature anomalies accurately and efficiently. For the LSTM model, a deep architecture was designed with four stacked layers, where the first three layers used return_sequences=True to retain the sequential data structure, and the final layer used return_sequences=False for flattened predictions. Each layer comprised 50 units, balancing complexity and performance, followed by a dense output layer. Regularization techniques, including dropout layers (0.2–0.4 rates), L2 regularization, and batch normalization, were employed to mitigate overfitting and stabilize training. A ReduceLROnPlateau scheduler and early stopping ensured efficient convergence by adjusting the learning rate dynamically and halting training when validation loss plateaued. Hyperparameter tuning with Keras Tuner explored configurations such as LSTM units, dropout rates, and L2 values, using the Hyperband algorithm to identify optimal parameters. Performance was evaluated using metrics like MSE, MAE, and R², with residual analysis confirming no systematic bias.


**Prophet**
For the Prophet model, preprocessing involved renaming columns (ds and y), converting the Year column to datetime format, and handling missing values. The model was tuned with growth='linear' to capture steady trends, a changepoint_prior_scale of 0.05 to balance sensitivity to trend shifts, and a seasonality_prior_scale of 10 for robust seasonal pattern detection. Custom decadal seasonality (10-year period, Fourier order of 5) was added to capture long-term cycles. Evaluation metrics like MAE and R² validated the model's accuracy, while uncertainty intervals (yhat_lower and yhat_upper) provided a range for forecasted values, reflecting inherent variability. Visualizations combined historical and forecasted data, illustrating the contributions of trend and seasonal components.

Both models exhibited strong predictive capabilities, with the Prophet model excelling in interpretability and reliability, while the LSTM model effectively captured complex temporal patterns, showcasing their complementary strengths.

**Anomaly Detection**
This project uses a hybrid approach for global temperature anomaly detection, combining Prophet and Isolation Forest. Prophet detects anomalies by comparing actual values to forecasted predictions with confidence intervals, categorizing them as Above Expected, Below Expected, or Normal. It excels in interpretability and trend-based anomaly detection.

Isolation Forest complements Prophet by analyzing feature distributions, using Prophet forecasts as inputs to identify complex anomalies through recursive partitioning. Performance is evaluated with precision, recall, and F1-score metrics, ensuring accurate detection. This combination provides a robust system for detecting anomalies in applications like climate analysis, finance, and IoT.

**Alert system**
I developed an automated alert system for detecting anomalies in temperature anomaly forecasts. The system compares simulated actual values with Prophet model predictions and their confidence intervals. Anomalies are flagged when actual values fall outside the forecasted bounds and are categorized as "Above Expected," "Below Expected," or "Normal."

When an anomaly is detected, an email alert is automatically triggered, containing essential information such as the anomaly date, forecasted value, actual value, and anomaly type. The SMTP protocol is used to send these alerts. Additionally, a visual plot highlights anomalies, forecasted values, and confidence intervals, providing a clear overview of data trends and deviations. This system is designed to support decision-making in fields like climate studies, finance, and IoT monitoring by quickly notifying stakeholders of significant changes.


## Result
<!-- content -->
This project focused on anomaly detection using the LSTM and Prophet models for temperature anomaly forecasting. The LSTM model was first evaluated, showing a Test Loss of 0.3387, a MAE of 0.3764, and an R² of 0.7037. However, after hyperparameter tuning, the model improved with a Test Loss of 0.2872, but MAE increased slightly to 0.4222, and R² decreased to 0.6869, highlighting a trade-off between bias and variance.

The Prophet model outperformed the LSTM model with a MAE of 0.2957 and an R² of 0.8687, demonstrating its strength in handling seasonality and trends without extensive parameter tuning.

For anomaly detection, the Prophet model achieved perfect scores in precision, recall, and F1-score (1.0), while the Isolation Forest model performed poorly, with a precision of 0.2 and recall of 0.0118, indicating its limitations in this task. The Prophet model’s superior performance makes it the better choice for reliable anomaly detection.

Additionally, the email alert system successfully notified stakeholders when anomalies were detected by the Prophet model, ensuring prompt action could be taken.



## Conclusion
<!-- content -->
This project compared the LSTM and Prophet models for anomaly detection in temperature data. The Prophet model performed better, achieving a MAE of 0.2957 and R² of 0.8687, with perfect precision, recall, and F1-scores, making it highly effective for identifying anomalies. The LSTM model showed improvement after hyperparameter tuning, but it still faced challenges like overfitting, with slight increases in MAE and decreases in R². The Isolation Forest model underperformed in comparison. In conclusion, the Prophet model was the most reliable, while the LSTM model showed potential for further refinement.




I hope you find our project insightful and useful!
