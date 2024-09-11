import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree

# Загрузка данных
data = pd.read_csv('diamonds.csv', encoding='utf-8')
data = data.drop(['cut','color', 'clarity'], axis='columns')

X = data.drop(['cut_encoded'], axis='columns')
y = data['cut_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


def get_dataset_information(data: pd.DataFrame):
    print(data.info)
    print(data.describe())


class DiamondClassifier:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(f"Точность: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(classification_report(y_test, y_pred))


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

param_grid = [
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': list(range(2, 11))
    },
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': list(range(1, 6))
    },
    {
        'classifier': [LogisticRegression(max_iter=1000)],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['lbfgs', 'liblinear']
    }
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Лучшие параметры: {grid_search.best_params_}")

classifier = DiamondClassifier(best_model)
classifier.evaluate(X_test, y_test)

if isinstance(best_model.named_steps['classifier'], DecisionTreeClassifier):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10), dpi=300)
    plot_tree(best_model.named_steps['classifier'],
              feature_names=X.columns,
              class_names=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
              filled=True)
    plt.show()


