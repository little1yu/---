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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

# 训练梯度提升分类器
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_clf.fit(X_train, y_train)

# 使用梯度增强预测训练和测试数据
y_train_pred_gb = gb_clf.predict(X_train)
y_test_pred_gb = gb_clf.predict(X_test)

# 预测梯度增强的概率
y_train_proba_gb = gb_clf.predict_proba(X_train)[:, 1]
y_test_proba_gb = gb_clf.predict_proba(X_test)[:, 1]

# 计算准确度
train_accuracy_gb = accuracy_score(y_train, y_train_pred_gb)
test_accuracy_gb = accuracy_score(y_test, y_test_pred_gb)

# 训练集和测试集的ROC曲线和AUC
fpr_train_gb, tpr_train_gb, _ = roc_curve(y_train, y_train_proba_gb)
auc_train_gb = auc(fpr_train_gb, tpr_train_gb)
fpr_test_gb, tpr_test_gb, _ = roc_curve(y_test, y_test_proba_gb)
auc_test_gb = auc(fpr_test_gb, tpr_test_gb)

# 训练集和测试集的准确度对比
plt.figure(figsize=(8, 4))
plt.bar(['训练集', '测试集'], [train_accuracy_gb, test_accuracy_gb], color=['blue', 'green'])
plt.xlabel('数据集')
plt.ylabel('准确率')
plt.title('训练集和测试集上的梯度提升模型的准确率对比图-于康')
plt.ylim([0.85, 1])
plt.show()

# 绘制ROC曲线
plt.figure(figsize=(12, 6))
plt.plot(fpr_train_gb, tpr_train_gb, label=f'Train AUC = {auc_train_gb:.2f}')
plt.plot(fpr_test_gb, tpr_test_gb, label=f'Test AUC = {auc_test_gb:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('梯度提升模型的ROC曲线-于康')
plt.legend()
plt.show()

# 绘制准确度随迭代而变化
train_score = np.empty(len(gb_clf.estimators_))
test_score = np.empty(len(gb_clf.estimators_))

for i, pred in enumerate(gb_clf.staged_predict(X_train)):
    train_score[i] = accuracy_score(y_train, pred)

for i, pred in enumerate(gb_clf.staged_predict(X_test)):
    test_score[i] = accuracy_score(y_test, pred)

plt.figure(figsize=(12, 6))
plt.plot(train_score, label='Training Score')
plt.plot(test_score, label='Test Score')
plt.xlabel('升压级数')
plt.ylabel('准确率')
plt.title('梯度助推训练迭代的准确性变化图-于康')
plt.legend()
plt.show()