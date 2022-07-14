import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Load cars data
data = pd.read_csv(r'C:\3rd_year\second term\pattern recognition\labs\lab4\assignment2_dataset_cars.csv')

# preprocessing phase
# Drop the rows that contain missing values
data.dropna(how='any', inplace=True)

# Features
X = data.iloc[:, 0:3]
# Label
Y = data['price']


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

cols = {'car_maker'}
X = Feature_Encoder(X, cols)


scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)


# Feature Selection phase _ Get the correlation between the features
corr = data.corr()
# Top 50% Correlation training features with the price
top_feature = corr.index[abs(corr['price']) > 0.01]

# Correlation plotting
plt.subplots(figsize=(8, 5))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

cls = linear_model.LinearRegression()
cls.fit(X_train, y_train)
prediction = cls.predict(X_test)


poly_features = PolynomialFeatures(degree=4)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred = poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
