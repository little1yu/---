import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

lung_cancer_data = pd.read_csv('survey lung cancer.csv')

le_gender = LabelEncoder()
le_cancer = LabelEncoder()
lung_cancer_data['GENDER'] = le_gender.fit_transform(lung_cancer_data['GENDER'])
lung_cancer_data['LUNG_CANCER'] = le_cancer.fit_transform(lung_cancer_data['LUNG_CANCER'])

X = lung_cancer_data.drop('LUNG_CANCER', axis=1)
y = lung_cancer_data['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure(figsize=(40,20))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Cancer', 'Cancer'])
plt.title('可视化的决策树图-程唐琛',fontsize='xx-large')
plt.show()

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(train_accuracy, test_accuracy)

labels = ['Training Set', 'Testing Set']
accuracies = [train_accuracy, test_accuracy]
plt.figure(figsize=(8, 4))
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('在训练集和测试集上的模型准确率图-程唐琛')
plt.ylim([0.85, 1])
plt.show()

feature_importances = clf.feature_importances_
plt.figure(figsize=(12, 8))
plt.barh(X.columns, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('决策树模型中的特征重要性图-程唐琛')
plt.show()



