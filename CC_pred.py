#Building Logistic Regression model to predict credit card approval 
#Best LR model has 0.85 test score

import pandas as pd
import numpy as np
cc_apps = pd.read_csv('datasets/cc_approvals.data', header=None)
cc_apps.head()

# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)
print("\n")

cc_apps_info = cc_apps.info()
print(cc_apps_info)
print("\n")

# Replace the '?'s with NaN
cc_apps = cc_apps.replace('?', np.NaN)

# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)
print(cc_apps.isnull().sum())

# Iterate over each column
for col in cc_apps:
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])
        
# Count the number of NaNs in the dataset and print
print(cc_apps.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps:
    if cc_apps[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col]=le.fit_transform(cc_apps[col])

from sklearn.model_selection import train_test_split

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X,y = cc_apps[:,0:12] , cc_apps[:,13]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size= 0.33,
                                random_state=42)

from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)

from sklearn.metrics import confusion_matrix

# Use logreg to predict
y_pred = logreg.predict(rescaledX_test)

print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test, y_pred)

from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100,150,200]
param_grid = dict(tol=tol, max_iter=max_iter)

# Instantiate GridSearchCV
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


