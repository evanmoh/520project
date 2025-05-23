# Duke AIPI520 Semester Project
## Evan Moh
## Video that explains the project: 

https://drive.google.com/file/d/1VaPbxRvD187XqYS3et5hS0Ilo9lm3jDO/view?usp=sharing

## Problem statement:
Using UCLA California Health Interview Survey (CHIS) data, the goal is to build a regression model that predicts the number of emergency room visits that a person will make in a 1 year period.
This model aims to find out key demographic and healthcare related factors associated with frequent emergency room use and support potential interventions in public health planning.

## Data:
https://drive.google.com/file/d/1mXFo4WdzvG_3oh0zeHnKgmnVMZmVKqNX/view?usp=drive_link

https://healthpolicy.ucla.edu/our-work/california-health-interview-survey-chis/access-chis-data

More information about One-Year Public Use Files (PUFs) 2023
https://healthpolicy.ucla.edu/our-work/public-use-files/one-year-public-use-files-pufs

## Examples of data features (530 total variables/topics); interviewed more than 21,671 adults
health status, health conditions, mental health, health behaviors, gun violence, women's health, dental health, neighborhood and housing, adverse childhood experiences, 
access to and use of health care, food environment, etc

## Models compared 
Random forest regressor (non deep learning) <br />
XGBoost Regressor (non deep learning) <br />
Adaline (deep learning) <br />

## Processes
Cleaned special codes, one hot encoded categorical variables, imputed missing values, selected top 100 features using RF

## Evaluation Metrics
MAE, MSE, R^2

## Top 20 features by correlation with target:
feature   correlation    abs_correlation <br />
INSMC          0.115998         0.115998.  Covered by Medicare Yes/No<br />
AC116_P1      -0.113856         0.113856.  How long since last Marijuana/Hashish use<br />
AI1            0.106906         0.106906.  Covered by Medicare Yes/No<br />
INS9TP         0.101613         0.101613.  Type of current health coverage source for all ages<br />
AC81C         -0.097306         0.097306.  Ever smoked electronic cigarettes<br />
MARCUR        -0.093596         0.093596   Current MARIJUANA user<br />
INSTYPE        0.093167         0.093167   type of current health coverage source<br />
AI25          -0.089583         0.089583   covered for prescription drugs<br />
AG3           -0.089073         0.089073   Have any kind of dental insurance<br />
AQ3           -0.085555         0.085555   Lived with anyone who used illegal drugs/abused prescription meds<br />
AK1           -0.084784         0.084784   Work status last week<br />
AF112B_4       0.084303         0.084303   Mental health harmed by event: extreme heat wave<br />
UR_OMB        -0.083455         0.083455   Rural and urban OMB<br />
AK22_P1        0.080060         0.080060   HH Total Ann. inc before taxes in previous year<br />
TRANSGEND2     0.076094         0.076094   current self reported gender and sex at birth (cisgender, transgender)<br />
SRAGE_P1      -0.075572         0.075572   Self-reported age<br />


## Top 20 Features from Random Forest (by importance):
feature  importance <br />
ACMDNUM    0.078621    # OF DOCTOR VISITS PAST YEAR <br />
HGHTI_P    0.018698    HEIGHT: INCHES<br />
AJ115_P1    0.016190    # OF DAYS MISS WORK DUE TO ILLNESS, INJURY, OR DISABILITY<br />
AK22_P1    0.016131    Income BEFORE TAXES IN PREVIOUS YEAR; income bracket<br />
SRAGE_P1    0.013546    self reported age<br />
AH3_P1    0.013306    KIND OF PLACE FOR USUAL SOURCE OF HEALTH CARE<br />
AADATE_MM    0.012762    DATE ADULT SECTION A DONE<br />
OCCMAIN2    0.011684    Main Occupation<br />
AI22A_P    0.011572    Name of health plan<br />
POVLL    0.010935    Poverty level<br />
AJ247    0.010837    Dental service in past 12 months<br />
AHEDC_P1    0.010669    Educational attainment<br />
INDMAIN2    0.009464    Main industry<br />


## Model performance and comment:

Random Forest
MAE: 0.6827522901509302
MSE: 0.859232628283613
R2: 0.1488155198610639<br />

XGBoost Regressor
MAE: 0.7045217554018597
MSE: 0.9620149747029847
R2: 0.046995901722033495<br />

Adaline Regressor
MAE: 0.6617259223859119
MSE: 0.8767267687070994
R2: 0.1314852412709826<br />

Random forest had the best overall R squared and MSE. XGBoost might be underfitting. Adaline had the best MAE.
It seems like RF achieved the best performance in terms of explained variance and MSE which means it made fewer mistakes than other models.
XGBoost underperformed especially with R^2. This might be due to lack of hyperparameter tuning which can be the next step.
Adaline got the lowest MAE, which means it made the smallest average prediction errors. Adaline is linear based NN and this seems pretty competitive.
In the end, RF is the best performing model.
Could use more advanced neural net like Keras MLP and could have done GridSearchCV for better hyperparameters for RF and XGBoost.

I expected that neural net based model would perform better and XGBoost model to perform better than random forest model to begin with. It was interesting RF model
was better performing model compared to the others.
