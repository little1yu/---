import pandas as pd
# 加载数据集
data_path = r'C:\Users\DN\Desktop\机器学习\肺癌预测/survey lung cancer.csv'
lung_cancer_data = pd.read_csv(data_path)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
# 对分类变量进行编码
le_gender = LabelEncoder()
le_cancer = LabelEncoder()
lung_cancer_data['GENDER'] = le_gender.fit_transform(lung_cancer_data['GENDER'])
lung_cancer_data['LUNG_CANCER'] = le_cancer.fit_transform(lung_cancer_data['LUNG_CANCER'])
# 定义特征和目标变量
X = lung_cancer_data.drop('LUNG_CANCER', axis=1)
y = lung_cancer_data['LUNG_CANCER']
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
# 训练模型
mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=200, random_state=42)
mlp_clf.fit(X_train, y_train)
# 使用MLP预测训练和测试数据
y_train_pred_mlp = mlp_clf.predict(X_train)
y_test_pred_mlp = mlp_clf.predict(X_test)
# 预测MLP的概率
y_train_proba_mlp = mlp_clf.predict_proba(X_train)[:, 1]
y_test_proba_mlp = mlp_clf.predict_proba(X_test)[:, 1]
# 计算MLP的准确率
train_accuracy_mlp = accuracy_score(y_train, y_train_pred_mlp)
test_accuracy_mlp = accuracy_score(y_test, y_test_pred_mlp)
# 训练集和测试集的ROC曲线和AUC
fpr_train_mlp, tpr_train_mlp, _ = roc_curve(y_train, y_train_proba_mlp)
auc_train_mlp = auc(fpr_train_mlp, tpr_train_mlp)
fpr_test_mlp, tpr_test_mlp, _ = roc_curve(y_test, y_test_proba_mlp)
auc_test_mlp = auc(fpr_test_mlp, tpr_test_mlp)

# 可视化权重矩阵
weights = mlp_clf.coefs_
# 绘制权重矩阵
fig, axes = plt.subplots(1, len(weights), figsize=(15, 5))
for i, (ax, weight) in enumerate(zip(axes, weights)):
    cax = ax.matshow(weight, cmap='viridis')
    fig.colorbar(cax, ax=ax)
    ax.set_title(f'第 {i} 权重')
plt.title('MLP权重矩阵图-于康')
plt.show()

# 绘制训练集和测试集的准确率图像
plt.figure(figsize=(8, 4))
plt.bar(['训练集', '测试集'], [train_accuracy_mlp, test_accuracy_mlp], color=['blue', 'green'])
plt.xlabel('数据集')
plt.ylabel('准确率')
plt.title('训练集和测试集上的MLP模型准确率对比图-于康')
plt.ylim([0.85, 1])
plt.show()


# 绘制ROC曲线
plt.figure(figsize=(12, 6))
plt.plot(fpr_train_mlp, tpr_train_mlp, label=f'Train AUC = {auc_train_mlp:.2f}')
plt.plot(fpr_test_mlp, tpr_test_mlp, label=f'Test AUC = {auc_test_mlp:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('MLP的ROC曲线-于康')
plt.legend()
plt.show()


# 绘制损失的变化曲线
plt.figure(figsize=(8, 4))
plt.plot(mlp_clf.loss_curve_)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('MLP的损失曲线-于康')
plt.show()

# 绘制模型在测试集的混淆矩阵
cm = confusion_matrix(y_test, y_test_pred_mlp)
plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(cm, display_labels=['没有癌症', '有癌症']).plot(values_format='d')
plt.title('测试集上MLP的混淆矩阵-于康')
plt.show()




