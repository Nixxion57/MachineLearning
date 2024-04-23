import keras_tuner
import pandas as pd
from keras_tuner.src.backend import keras
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

dataset = pd.read_csv('smartphone_cleaned_new.csv')

print("-------------------DATA PREPROCESSING---------------------")


# print(dataset.info())
# Feature Creation (adding the missing values instead of dropping them)
def remove_missing_values(set):
    for column in set.columns:
        if set[column].dtype == object:
            set[column] = set[column].fillna(set[column].mode().iloc[0])
        else:
            set[column] = set[column].fillna(set[column].mean())
    return set


cleaned_dataset = remove_missing_values(dataset)
print(cleaned_dataset.isnull().sum())
print(cleaned_dataset.dtypes)
cleaned_dataset_copy = cleaned_dataset.copy()

# Convert to categorical data
cleaned_dataset_copy[['brand_name', 'model', 'processor_brand', 'os']] = cleaned_dataset_copy[
    ['brand_name', 'model', 'processor_brand', 'os']].astype('category')

print(cleaned_dataset_copy.dtypes)

# Encode the target variable 'brand_name' to numerical labels
label_encoder = LabelEncoder()
cleaned_dataset_copy['brand_name'] = label_encoder.fit_transform(cleaned_dataset_copy['brand_name'])

# Remove outliers
z_scores = np.abs(stats.zscore(cleaned_dataset_copy['price']))
cleaned_dataset_copy = cleaned_dataset_copy[(z_scores < 2)]
print(cleaned_dataset_copy.shape)

# Exclude the target variable from input features
X = cleaned_dataset_copy.select_dtypes(include=np.number).drop(columns=['brand_name'])
y = cleaned_dataset_copy['brand_name']

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=3)
X_new = selector.fit_transform(X, y)
best_features = selector.get_support(indices=True)
print(X.columns[best_features].tolist())

print("-------------------DATA SEPARATION---------------------")
X = cleaned_dataset_copy[['price', 'num_cores', 'battery_capacity']]
y = cleaned_dataset_copy['brand_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("-------------------DECISION TREE---------------------")
# DECISION TREE CLASSIFIER MODEL

# Create, train model and make some predictions
model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(X_train, y_train)

# Visualize the tree
# plt.figure(figsize=(12, 6))
# tree.plot_tree(model_decision_tree, feature_names=X.columns, class_names=y.unique(), filled=True, fontsize=10)
# plt.show()

predictions = model_decision_tree.predict(X_test)
print(label_encoder.inverse_transform(predictions))

# Evaluate the model
score = accuracy_score(y_test, predictions)
print(f'Accuracy Score Decision Tree: {score}')

print("-------------------RANDOM FOREST---------------------")
# RANDOM FOREST CLASSIFIER MODEL
model_random_forest = RandomForestClassifier()
model_random_forest.fit(X_train, y_train)

predictions_random_forest = model_random_forest.predict(X_test)
print(label_encoder.inverse_transform(predictions_random_forest))

# Evaluate the model
score_random_forest = accuracy_score(y_test, predictions_random_forest)
print(f'Accuracy Score Random Forest: {score_random_forest}')

# Cross validation
# Cross-validation for Decision Tree
scores_decision_tree = cross_val_score(model_decision_tree, X_train, y_train, cv=5, scoring='accuracy')
print("Decision Tree Cross-Validation Scores:", scores_decision_tree)
print("Mean Accuracy (Decision Tree):", scores_decision_tree.mean())

# Cross-validation for Random Forest
scores_random_forest = cross_val_score(model_random_forest, X_train, y_train, cv=5, scoring='accuracy')
print("Random Forest Cross-Validation Scores:", scores_random_forest)
print("Mean Accuracy (Random Forest):", scores_random_forest.mean())

