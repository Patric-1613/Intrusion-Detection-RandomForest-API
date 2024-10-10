# Intrusion Detection System Using Random Forest

## Overview

This project implements an **Intrusion Detection System (IDS)** using a **Random Forest classifier**. Initially, the dataset size (494020 entries, 42 features) indicated the potential use of deep learning algorithms. However, through effective feature extraction and engineering, we reduced the dataset significantly, enabling a simpler machine learning algorithm like Random Forest to perform exceptionally well. The model was trained to detect network intrusions and deployed via an API, allowing users to predict whether a connection is normal or an attack based on critical network features.

## Motivation

Initially, the dataset's large size and complexity suggested that a deep learning approach would be necessary. However, by reducing the dataset and removing irrelevant or low-variance features, we demonstrated that a simpler model like **Random Forest** can still deliver **high accuracy** and precise results, making it suitable for production in an API environment.

## Dataset

The dataset used is from the **KDD Cup 1999** dataset for network intrusion detection. The dataset contains both normal and malicious connections. We transformed the dataset into a binary classification problem where:
- `0` represents normal connections
- `1` represents attack connections

## Project Steps

1. **Loading the Dataset**  
   The dataset was loaded, and checks were made for any null or missing values. No missing values were found.

2. **Removing Duplicates**  
   Duplicate entries were identified and removed to avoid model bias and to ensure cleaner data, reducing redundancy and preventing overfitting.

3. **Exploring the Dataset**  
   The data was explored to get the exact count of numeric and categorical features.

4. **Handling Low-Variance Features**  
   We identified two columns with just **one unique value** and deleted them as they added no predictive value.  
   Then, we visualized the **variance graph** to filter out features with variance lower than 0.01. This process reduced the features from 113 to **47** after one-hot encoding and variance analysis.

5. **Multicollinearity Analysis**  
   We used a **correlation matrix** and visualized it through a heatmap to identify multicollinearity among features.  
   To avoid removing features based on a single criterion, we applied **Variance Inflation Factor (VIF)** values. Features with VIF values greater than **10** were identified and removed.  
   The common features from both correlation analysis and VIF were removed:
   - `'lnum_compromised'`, `'srv_serror_rate'`, `'srv_rerror_rate'`
   - `'dst_host_serror_rate'`, `'dst_host_srv_serror_rate'`
   - `'flag_S0'`, `'flag_SF'`

6. **Binary Classification Problem**  
   The labels were converted into a **binary classification** problem, where **normal** connections are labeled as `0`, and **attack** connections are labeled as `1`.

7. **Label Distribution**  
   The label distribution was checked, and the proportion of normal to attack connections was found to be **60:40**, which was acceptable for our classification task, so no balancing was performed.

8. **Feature Selection for API**  
   Since the aim was to deploy this model via an API, we aimed to minimize the number of features users would need to input. To achieve this:
   - We applied **PCA (Principal Component Analysis)** but found it led to 22 important features.
   - We then turned to a **Random Forest Classifier** to rank feature importance and selected the top **10** features based on importance scores.

9. **Top 10 Selected Features**
   - `same_srv_rate`
   - `count`
   - `src_bytes`
   - `diff_srv_rate`
   - `dst_host_srv_count`
   - `service_private`
   - `dst_host_same_srv_rate`
   - `serror_rate`
   - `dst_bytes`
   - `dst_host_diff_srv_rate`

10. **Model Training**  
    Multiple machine learning algorithms were evaluated, including Gradient Descent, Distance-based, and Tree-based methods.  
    The **Random Forest Classifier** was chosen for its high accuracy, particularly its performance on the **confusion matrix**, where it showed very few false negatives.

11. **Model Deployment**  
    The trained Random Forest model was saved as a `.pkl` file using **joblib**.  
    An API was developed using **Flask** to predict whether a connection is normal or an attack based on user inputs for the 10 selected features.

## API Structure

The API takes the following inputs:
1. `same_srv_rate`
2. `count`
3. `src_bytes`
4. `diff_srv_rate`
5. `dst_host_srv_count`
6. `service_private`
7. `dst_host_same_srv_rate`
8. `serror_rate`
9. `dst_bytes`
10. `dst_host_diff_srv_rate`

Upon submitting these inputs, the API predicts whether the connection is **"Normal"** or an **"Attack Detected"**.

## Results

- The **Random Forest classifier** performed exceptionally well with minimal features, showcasing high accuracy on test data.
- The streamlined model and reduced feature set demonstrated that even with smaller datasets and fewer features, machine learning models can deliver robust predictions.

## Technologies Used

- **Python**: For data preprocessing, model building, and deployment.
- **Pandas, NumPy**: For data manipulation.
- **Scikit-learn**: For model training, evaluation, and feature selection.
- **Flask**: For building the API.
- **HTML/CSS**: For building the frontend of the API.

## How to Use This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Patric-1613/Intrusion-Detection-RandomForest-API.git
2. Install dependencies:
   '''bash
   pip install -r requirements.txt
3. Run the Flask app:
   '''bash
   python app.py
4. Access the API at http://127.0.0.1:5000/ and input the required features to get a prediction.

## Conclusion:
By applying thoughtful feature engineering and selection methods, we were able to reduce the complexity of the intrusion detection dataset, allowing a simple Random Forest model to perform exceptionally well. This project demonstrates that with the right preprocessing techniques, even a dataset initially suited for deep learning can be efficiently handled using machine learning models.
   
