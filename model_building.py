""""
Created on Mon, July 17, 2023
'
@author: cipher499
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

#read the data into a dataframe
df = pd.read_csv("data_eda.csv")

# chooose relevant columns
df.columns

df_model = df[['Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'num_comp', 'hourly', 
            'employer_provided', 'state', 'age', 'python', 'spark', 'aws', 'excel', 'tableau', 'job_simp', 'seniority', 'desc_len', 'avg_salary']]

# get dummy data/ one-hot encode: for each type of categorical variable
df_dum = pd.get_dummies(df_model)
df_dum.head()
df_dum.columns

# create a train-test split; train -> cross-validate -> test
X = df_dum.drop('avg_salary', axis = 1)
# convert the df into an array which is the suitable data structure for ML modeling
y = df_dum.avg_salary.values          
print(X.shape, y.shape)

#null values in y
pd.isnull(y).sum()
mean_value = np.nanmean(y)
y[np.isnan(y)] = mean_value

#---------------------------------------------------#
from sklearn.model_selection import train_test_split
#---------------------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
#---------------------------------------------------#
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
#---------------------------------------------------#
lr = LinearRegression()
lr.fit(X_train, y_train)

np.mean(cross_val_score(lr, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=10))

# Lasso Regression
lm_l = Lasso()
np.mean(cross_val_score(lm_l, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=10))

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/100)
    lm_l = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lm_l, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3)))

plt.plot(alpha, error)
plt.show()

tup = tuple(zip(alpha, error))
df_err = pd.DataFrame(tup, columns=['alpha', 'error'])
df_err[df_err.error==max(df_err.error)]

# error is lowest for alpha = 0.5
lm_l = Lasso(alpha=0.5)
lm_l.fit(X_train, y_train)

# Random Forest
#-------------------------------- ----------------#
from sklearn.ensemble import RandomForestRegressor
#-------------------------------- ----------------#
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=10))

# Tune the model using GridSearchCV
#-------------------------------- ----------------#
from sklearn.model_selection import GridSearchCV
#-------------------------------- ----------------#
parameters = {'n_estimators':range(10, 300, 10), 'criterion':['squared_error', 'absolute_error'], 'max_features':['auto', 'sqrt', 'log2']}

#scoring = make_scorer(mean_absolute_error, greater_is_better=False)

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=10)
gs.fit(X_train, y_train)

best_params = gs.best_params_
best_score = -gs.best_score_

print("Best Parameters:", best_params)
print("Best Score (mean absolute error):", best_score)

# Test ensembles

#-------------------------------- ----------------#
from sklearn.metrics import mean_absolute_error
#-------------------------------- ----------------#

tpred_lr = lr.predict(X_test)
tpred_lm_l = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

mean_absolute_error(y_test, tpred_lr)
mean_absolute_error(y_test, tpred_lm_l)
mean_absolute_error(y_test, tpred_rf)


# productionise the model by converting it into an API endpoint using Flask
#-------------------------------- ----------------#
import pickle
#-------------------------------- ----------------#
pickl = {'model': lm_l}
pickle.dump(pickl, open('model_file' + ".p", "wb"))

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']


model.predict(X_test.iloc[1,:].values.reshape(1,-1))
y_test[1]

list(X_test.iloc[1,:])
