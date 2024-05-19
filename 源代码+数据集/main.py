import pandas as pd
# 加载数据集
data = pd.read_csv('survey lung cancer.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# 处理缺失值
data = data.dropna()
# 编码分类变量
labelencoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = labelencoder.fit_transform(data[column])
# 分离特征和标签
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']
# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# 定义分类器
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB(),
    "Multi-layer Perceptron": MLPClassifier(),
}
# 存储结果
results = {}
# 训练和评估模型
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}:\n{classification_report(y_test, y_pred)}\n")

# 比较结果
plt.figure(figsize=(12, 8))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xticks(rotation=45)
plt.ylabel('准确率')
plt.title('模型正确率比较-于康')
plt.show()

# 训练模型并计算预测概率
fpr = dict()
tpr = dict()
roc_auc = dict()
# 训练和计算ROC曲线
plt.figure(figsize=(12, 8))
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr[name], tpr[name], _ = roc_curve(y_test, y_pred_proba)
    roc_auc[name] = auc(fpr[name], tpr[name])
    plt.plot(fpr[name], tpr[name], lw=2, label=f'{name} (AUC = {roc_auc[name]:.2f})')
# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('不同模型的ROC曲线-于康')
plt.legend(loc='lower right')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# 定义分类器
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB(),
    "Multi-layer Perceptron": MLPClassifier(max_iter=1000),
}
# 存储每个分类器的交叉验证得分
results = []
# 进行交叉验证并存储结果
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    results.append(scores)
    print(f"{name}:\nMean Accuracy: {scores.mean():.2f}, Standard Deviation: {scores.std():.2f}\n")
# 绘制箱型图
plt.figure(figsize=(15, 10))
sns.boxplot(data=results)
plt.xticks(ticks=np.arange(len(classifiers) + 1), labels=list(classifiers.keys()) + ['Extreme Learning Machine'], rotation=45)
plt.ylabel('准确率')
plt.title('所有模型的箱型图-于康')
plt.show()




