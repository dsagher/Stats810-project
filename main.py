import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("smoking.csv") # load our dataset

# some data cleaning
df['gender'] = df['gender'].map({'M': 0, 'F': 1}) # encode male to 0 and female to 1
df['tartar'] = df['tartar'].map({'N': 0, 'Y': 1}) # encode "No" to 0 and "Yes" to 1
df = df.drop(columns=['oral']) # useless column which has the same value for everyone, just drop it

y = df['smoking'] # separate our binary target variable
X = df.drop(columns=['smoking']) # all features

# Split the set for training set and testing set with 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)

# subtract mean from each column and divide by standard deviation (data standardization is required for PCA)
X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std


# Perform standard singular value decomposition (required for PCA)
U, S, Vt = np.linalg.svd(X_train, full_matrices=False)

number_of_features = 3 # how many features we will use
principal_components = Vt[:number_of_features].T # rows of V^T will be the principal components

# Project the data onto principal components
X_train_pca = X_train @ principal_components
X_test_pca  = X_test  @ principal_components


""" 
Now our data is prepared for regression/model prediction by Dan
We have X_train_pca and y_train for model training,
and X_test_pca with y_test for evaluation of the model
"""

"""=============LogReg on Original Data=================="""

logreg = LogisticRegression(penalty='l2') 

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) 


odds_ratio = np.exp(logreg.coef_)

def zip_cols(df, weights):

    cols = np.array(df.columns)
    coef = weights.reshape(-1,1)
    return list(zip(cols, coef))

zipped_odds = zip_cols(df, odds_ratio)
odds_sorted = sorted(zipped_odds, key=lambda dct:dct[1], reverse=True)

rows = []
for i in odds_sorted:
    feature = i[0]
    odds = i[1]
    pct = round((odds.tolist()[0] - 1)* 100,2)
    if pct < 10 and pct > -10:
        continue
    row = {"feature": feature, "odds": pct}
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("smoking_odds.csv", index=True)

"""=============LogReg on Principal Components=================="""

logreg = LogisticRegression(penalty='l2')

logreg.fit(X_train_pca, y_train)

y_pred_pca = logreg.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred_pca)
