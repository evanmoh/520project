# Evan Moh 
# Duke AIPI520 Semester Project
# Video that explains: https://drive.google.com/file/d/1VaPbxRvD187XqYS3et5hS0Ilo9lm3jDO/view?usp=sharing

'''
Problem statement:
Using UCLA California Health Interview Survey (CHIS) data, the goal is to build a regression model that predicts the number of emergency room visits that a person will make in a 1 year period.
This model aims to find out key demographic and healthcare related factors associated with frequent emergency rom use and support potential interventions in public health planning.

Data:
https://healthpolicy.ucla.edu/our-work/california-health-interview-survey-chis/access-chis-data

More information about One-Year Public Use Files (PUFs) 2023
https://healthpolicy.ucla.edu/our-work/public-use-files/one-year-public-use-files-pufs

Examples of data features (530 total variables/topics); interviewed more than 21,671 adults
health status, health conditions, mental health, health behaviors, gun violence, women's health, dental health, neighborhood and housing, adverse childhood experiences, 
access to and use of health care, food environment, etc
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

'''
Data and Data Prep Pipeline

Loaded data, replaced special codes with NaN, 
Dropped high-missing and high-cardinality features, 
Performed one hot encoding, 
Aligned and filtered features and target,
Imputed any missing values or NaNs that was changed from not applicable values. 
'''

#Importing data - UCLA California Health Interview Survey (CHIS) Data Dictionary

df = pd.read_sas("adult.sas7bdat", format='sas7bdat')

#Replacing all the values that are N/A's.
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].replace([-1,-2,-9],np.nan)

#dropping values that has no value at all
df = df.dropna(axis=1,how='all')
df=df.loc[:,df.nunique(dropna=True)>1]

#target
df['AH95_P1'] = df['AH95_P1'].replace(-1, 0)
y = df['AH95_P1'] # TIMES VISITED ER IN PAST 12 MOS


X = df.drop(columns=['ER','AH12V2','AH95_P1']) #drop ER related column

# Drop features with >50% missing values
X = X.loc[:, X.isnull().mean() < 0.5]

# Drop features with too many categories (>50)
X = X.loc[:, X.nunique(dropna=True) < 50]

#using dummies to convert categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

mask = y.notna()
X_encoded = X_encoded.loc[mask]
y = y.loc[mask]


# Impute missing values in features
imputer = SimpleImputer(strategy='mean')
X_encoded_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

# Align index for compatibility
X_encoded_imputed.reset_index(drop=True, inplace=True)
y = y.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded_imputed, y, test_size=0.2, random_state=42)

'''
Feature engineering and selection

Selected the top 100 features using random forest regressor by using feature importance. The reason is that I did not want to all the features as this might bring down accuracy.
Applied standard scaler for Adeline which I will be using later
Looked at Correlation analysis for top predictive variables.

'''
#instead of selecting all the features, choose top 100 features - Feature Engineering
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)

importances = rf_temp.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

top_100_features = feature_importance_df.head(100)['feature'].tolist()
X_train_top100 = X_train[top_100_features]
X_test_top100 = X_test[top_100_features]



# Let's scale our data to help the algorithm converge faster
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_top100)
X_test_scaled = scaler.transform(X_test_top100)


'''
Model Optimization

Selecting the top features
Tuned the learning rate and epochs for Adaline
Used max_depth = 10 for RF and default for XGBoost after trying few times.
Added gradient clipping to stabilize training
'''

# Random forest and XGBoost models for non Deep learning models

#Use random forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_top100, y_train)
y_pred_rf = rf.predict(X_test_top100)


scale_ratio = (y == 0).sum() / (y == 1).sum()

#Use XGBoost
xgb = XGBRegressor(
    objective='reg:squarederror',
    base_score=0.5,
    random_state=42
)
xgb.fit(X_train_top100, y_train)
y_pred_xgb = xgb.predict(X_test_top100)

'''
Choice of evaluation metric / justification
I used MAE to measure avg error in predicting number of ER visits
Used MSE for penalizing larger error
Used R squared to explain the variance.
'''

print("Random Forest")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2:", r2_score(y_test, y_pred_rf))


print("XGBoost Regressor")
print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("MSE:", mean_squared_error(y_test, y_pred_xgb))
print("R2:", r2_score(y_test, y_pred_xgb))




correlation_with_target = pd.DataFrame(X_encoded.corrwith(y), columns=['correlation'])
correlation_with_target['abs_correlation'] = correlation_with_target['correlation'].abs()
correlation_with_target = correlation_with_target.sort_values('abs_correlation', ascending=False)



# Print top 20 correlated features
print("Top 20 features by correlation with target:")
print(correlation_with_target.head(20))

'''
Top 20 features by correlation with target:
            correlation  abs_correlation
INSMC          0.115998         0.115998.  Covered by Medicare Yes/No
AC116_P1      -0.113856         0.113856.  How long since last Marijuana/Hashish use
AI1            0.106906         0.106906.  Covered by Medicare Yes/No
INS9TP         0.101613         0.101613.  Type of current health coverage source for all ages
AC81C         -0.097306         0.097306.  Ever smoked electronic cigarettes
MARCUR        -0.093596         0.093596   Current MARIJUANA user
INSTYPE        0.093167         0.093167   type of current health coverage source
AI25          -0.089583         0.089583   covered for prescription drugs
AG3           -0.089073         0.089073   Have any kind of dental insurance
AQ3           -0.085555         0.085555   Lived with anyone who used illegal drugs/abused prescription meds
AK1           -0.084784         0.084784   Work status last week
AF112B_4       0.084303         0.084303   Mental health harmed by event: extreme heat wave
UR_OMB        -0.083455         0.083455   Rural and urban OMB
AK22_P1        0.080060         0.080060   HH Total Ann. inc before taxes in previous year
TRANSGEND2     0.076094         0.076094   current self reported gender and sex at birth (cisgender, transgender)
SRAGE_P1      -0.075572         0.075572   Self-reported age
CV7V3_10      -0.074673         0.074673
UR_RHP        -0.074460         0.074460
AK139_8        0.073373         0.073373
AB154         -0.073237         0.073237
'''

# Neural Network model, Adaline for Deep learning model 
class Adaline:
    
    def __init__(self,eta=0.01,n_iter=100,random_state=0):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
        self.cost_path=[]
    
    def fit(self,X,y):
        # Initialize the weights and bias (weights[0]) to small random numbers
        rgen=np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0,scale=0.01,size=(1+X.shape[1]))

        # Train adaline using batch gradient descent
        for i in range(self.n_iter):
            yhat = self.predict(X)
            error = y - yhat


            # Calculate the cost function
            cost = np.sum(0.5 * (y-yhat)**2)
            # Gradient of cost with respect to weights
            gradient_weights = error.T.dot(-X)
            # Gradient of cost with respect to bias
            gradient_bias = -np.sum(error)
            # Update weights and bias
            max_grad = 1.0
            delta_weights = np.clip(self.eta * gradient_weights, -max_grad, max_grad)
            delta_bias = np.clip(self.eta * gradient_bias, -max_grad, max_grad)
            self.weights[1:] -= delta_weights
            self.weights[0] -= delta_bias
            # Add cost to total cost counter
            self.cost_path.append(cost)
        return self
    
    def predict(self,X):
        z = np.dot(X,self.weights[1:]) + self.weights[0]
       
        return z
    
adaline_model = Adaline(eta=0.000005, n_iter=200)
adaline_model.fit(X_train_scaled,y_train)

test_preds = adaline_model.predict(X_test_scaled)

print("Adaline Regressor")
print("MAE:", mean_absolute_error(y_test, test_preds))
print("MSE:", mean_squared_error(y_test, test_preds))
print("R2:", r2_score(y_test, test_preds))

plt.plot(adaline_model.cost_path)
plt.title("Adaline Training Cost Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

print("Top 20 Features from Random Forest (by importance):")
print(feature_importance_df.head(20))

'''
Top 20 Features from Random Forest (by importance):
         feature  importance
196      ACMDNUM    0.078621    # OF DOCTOR VISITS PAST YEAR
293      HGHTI_P    0.018698    HEIGHT: INCHES
285     AJ115_P1    0.016190    # OF DAYS MISS WORK DUE TO ILLNESS, INJURY, OR DISABILITY
286      AK22_P1    0.016131    Income BEFORE TAXES IN PREVIOUS YEAR; income bracket
304     SRAGE_P1    0.013546    self reported age
281       AH3_P1    0.013306    KIND OF PLACE FOR USUAL SOURCE OF HEALTH CARE
193    AADATE_MM    0.012762    DATE ADULT SECTION A DONE
310     OCCMAIN2    0.011684    Main Occupation
163      AI22A_P    0.011572    Name of health plan
125        POVLL    0.010935    Poverty level
103        AJ247    0.010837    Dental service in past 12 months
284     AHEDC_P1    0.010669    Educational attainment
309     INDMAIN2    0.009464    Main industry
1            AB1    0.008961
258      DSTRSYR    0.008275
305  WORKSICK_P1    0.008057
36          AH14    0.007983
294    HHSIZE_P1    0.007924
287     AK3_P1V2    0.007569
289     AM186_P1    0.007245
'''

'''
Model performance and comment:

Random Forest
MAE: 0.6827522901509302
MSE: 0.859232628283613
R2: 0.1488155198610639

XGBoost Regressor
MAE: 0.7045217554018597
MSE: 0.9620149747029847
R2: 0.046995901722033495

Adaline Regressor
MAE: 0.6617259223859119
MSE: 0.8767267687070994
R2: 0.1314852412709826

Random forest had the best overall R squared and MSE. XGBoost might be underfitting. Adaline had the best MAE.
It seems like RF achieved the best performance in terms of explained variance and MSE which means it made fewer mistakes than other models.
XGBoost underperformed especially with R^2. This might be due to lack of hyperparameter tuning which can be the next step.
Adaline got the lowest MAE, which means it made the smallest average prediction errors. Adaline is linear based NN and this seems pretty competitive.
In the end, RF is the best performing model.
Could use more advanced neural net like Keras MLP and could have done GridSearchCV for better hyperparameters for RF and XGBoost.

I expected that neural net based model would perform better and XGBoost model to perform better than random forest model to begin with. It was interesting RF model
was better performing model compared to the others.
'''