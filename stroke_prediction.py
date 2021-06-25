import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

df = pd.read_csv('/Users/jiali/Documents/Python_CodingDojo/stroke-prediction/healthcare-dataset-stroke-data.csv')
print(df.head())
print(df.info())

df.drop(columns='id', inplace=True)

# There's no pattern in data where 'bmi' is N/A, so I decided to dropna.
print(df.isna().sum())
print(df[df['bmi'].isna()]['gender'].value_counts())
print(df[df['bmi'].isna()]['age'].value_counts())
print(df[df['bmi'].isna()]['hypertension'].value_counts())
print(df[df['bmi'].isna()]['heart_disease'].value_counts())
print(df[df['bmi'].isna()]['ever_married'].value_counts())
print(df[df['bmi'].isna()]['work_type'].value_counts())
print(df[df['bmi'].isna()]['Residence_type'].value_counts())
print(df[df['bmi'].isna()]['avg_glucose_level'].value_counts())
print(df[df['bmi'].isna()]['smoking_status'].value_counts())
print(df[df['bmi'].isna()]['stroke'].value_counts())
df.dropna(inplace=True)

# There's only one record where 'gender' is 'Other', so I dropped the row.
print(df['gender'].value_counts())
print(df.loc[df['gender']=='Other'])
df = df.loc[df['gender'] != 'Other']

print(df['age'].value_counts())
print(df['hypertension'].value_counts())
print(df['heart_disease'].value_counts())
print(df['ever_married'].value_counts())
print(df['work_type'].value_counts())
print(df['Residence_type'].value_counts())
print(df['avg_glucose_level'].value_counts())
print(df['bmi'].value_counts())
print(df['smoking_status'].value_counts())
print(df['stroke'].value_counts())

stroke_filter = df['stroke']==1

# 'Age' distribution for people who got stroke
# People over 50 are more likely to get stroke, especially female.
plt.figure(1)
df.loc[stroke_filter]['age'].hist(edgecolor='black', alpha=0.7, label='All gender')
df.loc[stroke_filter & (df['gender']=='Female')]['age'].hist(edgecolor='black', alpha=0.7, label='Female')
df.loc[stroke_filter & (df['gender']=='Male')]['age'].hist(edgecolor='black', alpha=0.7, label='Male')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Counts')
plt.title('Age distribution for people who got stroke')

# Compare 'Hypertension' by gender for people who got stroke
# People who don't have hypertension are likely to get stroke, especially for female.
plt.figure(2)
# reference(value_counts() as dataframe): https://re-thought.com/pandas-value_counts/
df_hyper0 = df.loc[stroke_filter & (df['hypertension']==0)]['gender'].value_counts().sort_index(ascending=True).to_frame()
df_hyper0 = df_hyper0.reset_index()
df_hyper0.columns = ['gender', 'counts by gender']
print(df_hyper0)
plt.bar(df_hyper0['gender'].index-0.2, df_hyper0['counts by gender'], 0.4)
df_hyper = df.loc[stroke_filter & (df['hypertension']==1)]['gender'].value_counts().sort_index(ascending=True).to_frame()
df_hyper = df_hyper.reset_index()
df_hyper.columns = ['gender', 'counts by gender']
print(df_hyper)
plt.bar(df_hyper['gender'].index+0.2, df_hyper['counts by gender'], 0.4)
plt.xticks([0,1], ['Female', 'Male'])
plt.legend(['No Hypertension', 'Hypertension'])
plt.title('Hypertension by gender for people who got stroke')

# Compare 'heart_disease' by gender for people who got stroke
# People who don't have heart disease are likely to get stroke, especially for female.
plt.figure(3)
# # reference(value_counts() as dataframe): https://re-thought.com/pandas-value_counts/
df_heart0 = df.loc[stroke_filter & (df['heart_disease']==0)]['gender'].value_counts().sort_index(ascending=True).to_frame()
df_heart0 = df_heart0.reset_index()
df_heart0.columns = ['gender', 'counts by gender']
print(df_heart0)
plt.bar(df_heart0['gender'].index-0.2, df_heart0['counts by gender'], 0.4)
df_heart = df.loc[stroke_filter & (df['heart_disease']==1)]['gender'].value_counts().sort_index(ascending=True).to_frame()
df_heart = df_heart.reset_index()
df_heart.columns = ['gender', 'counts by gender']
print(df_heart)
plt.bar(df_heart['gender'].index+0.2, df_heart['counts by gender'], 0.4)
plt.xticks([0,1], ['Female', 'Male'])
plt.legend(['No heart disease', 'Heart disease'])
plt.title('Heart disease by gender for people who got stroke')


# Average glucose level distribution for people who got stroke, most people are lower than 120
plt.figure(4)
df.loc[stroke_filter]['avg_glucose_level'].hist(edgecolor='black', alpha=0.7, label='All gender')
df.loc[stroke_filter & (df['gender']=='Female')]['avg_glucose_level'].hist(edgecolor='black', alpha=0.7, label='Female')
df.loc[stroke_filter & (df['gender']=='Male')]['avg_glucose_level'].hist(edgecolor='black', alpha=0.7, label='Male')
plt.legend()
plt.xlabel('Average glucose level')
plt.ylabel('Counts')
plt.title('Average glucose level distribution for people who got stroke')


# Average bmi distribution for people who got stroke, most people are between 25-33.
plt.figure(5)
df.loc[stroke_filter]['bmi'].hist(edgecolor='black', alpha=0.7, label='All gender')
df.loc[stroke_filter & (df['gender']=='Female')]['bmi'].hist(edgecolor='black', alpha=0.7, label='Female')
df.loc[stroke_filter & (df['gender']=='Male')]['bmi'].hist(edgecolor='black', alpha=0.7, label='Male')
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Counts')
plt.title('BMI distribution for people who got stroke')

# relationship between average clucose level and BMI in people who got stroke
# No pattern
plt.figure(6)
sns.scatterplot(x='avg_glucose_level', y='bmi', data=df.loc[stroke_filter], hue='gender', alpha=0.7)
plt.xlabel('Average glucose level')
plt.ylabel('BMI')
plt.title('BMI distribution for people who got stroke')

df['gender'] = df['gender'].map({'Male':1, 'Female':0})
df['ever_married'] = df['ever_married'].replace({'Yes':1, 'No':0})
df['Residence_type'] = df['Residence_type'].map({'Urban':1, 'Rural':0})


df_dummies = pd.get_dummies(df, columns = ['work_type', 'smoking_status'], drop_first=True)

# Modeling
X = df_dummies.drop(columns='stroke')
y = df_dummies['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, stratify=y)

# Random Forest
params = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 4, 6],'min_samples_leaf': [1, 2, 3]}
gs = GridSearchCV(RandomForestClassifier(), params) 
gs.fit(X_train, y_train) 
print(gs.best_params_)
print('Training score for Random Forest is:', gs.score(X_train, y_train))
print('Testing score for Random Forest is:', gs.score(X_test, y_test))

# KNN with pipeline
estimator_range = list(range(1,20))
scores = {}
for estimator in estimator_range:
    knn = KNeighborsClassifier(n_neighbors=estimator)
    knn.fit(X_train, y_train)
    scores[estimator] = knn.score(X_test, y_test)
print(max(scores, key=scores.get))
knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=14))
knn_pipe.fit(X_train, y_train)
print('Training accuracy for KNN is:', knn_pipe.score(X_train, y_train))
print('Testing accuracy for KNN is:', knn_pipe.score(X_test, y_test))

# Logistic Regression with pipeline
log_pipe = make_pipeline(StandardScaler(), LogisticRegression())
log_pipe.fit(X_train, y_train)
print('Training accuracy for Logistic Regression is:', log_pipe.score(X_train, y_train))
print('Testing accuracy for Logistic Regression is:', log_pipe.score(X_test, y_test))


plt.show()

