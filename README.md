# Stroke Prediction
Predict whether a patient is likely to get stroke using machine learning.

## Objectives
Analyze healthcare data to find features related to stroke and predict whether a patient is likely to get stroke.

## Factors
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
12) stroke: 1 if the patient had a stroke or 0 if not

- Note: "Unknown" in smoking_status means that the information is unavailable for this patient.

## Data clean
### Delete missing values in 'bmi'
201 values are missing in 'bmi' column. It's only 4% of data and there's no pattern in other columns where 'bmi' is missing, so I delete all NaNs.

### Delete 'Other' in 'gender
There's only one record is 'Other' so I drop the row.

## EDA
### 'age' distribution for people with stroke
People over 50 are more likely to get stroke, especially female.
![stroke-Figure_1](https://user-images.githubusercontent.com/82603737/125129766-cb088380-e0b4-11eb-834a-8f03121d4f3a.png)
### Compare 'hypertension' by gender for people with stroke
People who don't have hypertension are likely to get stroke, especially for female.
![stroke-Figure_2](https://user-images.githubusercontent.com/82603737/125129978-13c03c80-e0b5-11eb-9471-10108215d960.png)
### Compare 'heart_disease' by gender for people with stroke
People who don't have heart disease are likely to get stroke, especially for female.
![stroke-Figure_3](https://user-images.githubusercontent.com/82603737/125130112-48cc8f00-e0b5-11eb-9b7e-da825a9226f5.png)
### Hypertension
In people with stroke, more people have hypertension
![stroke-Figure_4](https://user-images.githubusercontent.com/82603737/125130171-67328a80-e0b5-11eb-989f-c03aca48951d.png)
### Average glucose level distribution for people with stroke
Most people are lower than 120.

![stroke-Figure_5](https://user-images.githubusercontent.com/82603737/125130232-829d9580-e0b5-11eb-8ae1-3188425f3b33.png)
### Average glucose level
On average, people with stroke have higher average glucose level, especially male.
![stroke-Figure_6](https://user-images.githubusercontent.com/82603737/125130288-99dc8300-e0b5-11eb-9966-59babc0d0d1c.png)
### Average bmi distribution for people with stroke
Most people are between 25-33.

![stroke-Figure_7](https://user-images.githubusercontent.com/82603737/125131242-423f1700-e0b7-11eb-83cd-e1556ae7c7aa.png)
### BMI
On average, people with stroke have slightly higher BMI than people who don't have stroke.
![stroke-Figure_8](https://user-images.githubusercontent.com/82603737/125130421-cdb7a880-e0b5-11eb-8413-318dc147e755.png)
### Relationship between average glucose level and BMI in people with stroke
No pattern

![stroke-Figure_9](https://user-images.githubusercontent.com/82603737/125131255-4a975200-e0b7-11eb-97a5-f0a08fe113be.png)
### Smoke
More never smoked people have stroke. But if combine 'formerly smoked' and 'smokes', the total is higher than 'never smoked'.

![stroke-Figure_10](https://user-images.githubusercontent.com/82603737/125130585-1a9b7f00-e0b6-11eb-94f6-73966e8388e1.png)
### Correlation between features and target
'age', 'hypertension', 'heart_disease', 'avg_glucose_level' are more related with 'stroke'.
![stroke-Figure_11](https://user-images.githubusercontent.com/82603737/125130816-81209d00-e0b6-11eb-8135-fbbb809e5524.png)

## Modeling
Use train test split on Random Forest, KNN wiht Pipeline, Logistic Regression with Pipeline, and Gradient Boosting.
### Random Forest
- GridSearchCV() to determine best parameters
- 'max_depth': None, 'min_samples_leaf: 1, 'min_samples_split': 4, 'n_estimators': 100
- training score: 0.98
- testing score: 0.96
### KNN with Pipeline
- 'n_neighbors': 14
- Pipeline to keep code simple
- training score: 0.96
- testing score: 0.96
### Logistic Regression with Pipeline
- Pipeline to keep code simple
- training score: 0.96
- testing score: 0.96
### Gradient Boosting
- training score: 0.97
- testing score: 0.95
## Conclusions
The final model I choose is Random Forest because it has the best score for both training and testing data.

## Data Source
[kaggle.com - Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
