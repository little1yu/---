import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, cohen_kappa_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

lung_cancer_data = pd.read_csv('survey lung cancer.csv')

numerical_columns = lung_cancer_data.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(20, 10))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, len(numerical_columns)//3 + 1, i)
    sns.histplot(data=lung_cancer_data, x=column, bins=30, kde=True)
    plt.title(column)

plt.tight_layout()
plt.show()

# 数据预处理函数
def encode_and_fill_missing(data, columns):
    le = LabelEncoder()
    for column in columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column].fillna(data[column].mode()[0]))
    return data
categorical_columns = lung_cancer_data.select_dtypes(include=['object']).columns
lung_cancer_data = encode_and_fill_missing(lung_cancer_data, categorical_columns)

# 特征和目标变量
X = lung_cancer_data.drop('LUNG_CANCER', axis=1)
y = lung_cancer_data['LUNG_CANCER']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 特征选择
estimator = DecisionTreeClassifier(random_state=42)
selector = RFE(estimator, n_features_to_select=5, step=1)
X_selected = selector.fit_transform(X_scaled, y)

test_size = 0.3
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=random_state)

log_reg = LogisticRegression(random_state=42)
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores.mean())

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    # 使用网格搜索进行超参数调优
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag']  # 只保留支持l2的求解器
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=1)
    grid_search.fit(X_train, y_train)
    
    # 使用网格搜索找到的最佳参数来训练模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 返回最佳模型和预测结果
    return best_model, y_pred, grid_search

# 调用函数进行模型训练和评估
best_model, y_pred, grid_search = train_and_evaluate_model(X_train, X_test, y_train, y_test, LogisticRegression(random_state=42))

# 打印分类报告
print(classification_report(y_test, y_pred))

# 打印最佳参数
print("Best parameters found by GridSearchCV:", grid_search.best_params_)

y_proba = best_model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
auc_score = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线图--李鑫煜')
plt.legend(loc="lower right")
plt.show()

# 假设 y_test 和 y_pred 是测试集真实标签和模型预测的标签
conf_matrix = confusion_matrix(y_test, y_pred)

# 从y_test中提取所有类别
classes = np.unique(y_test)

# 绘制混淆矩阵
plt.figure(figsize=(10, 7))  # 可以根据需要调整图像大小
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=classes, yticklabels=classes)
plt.title('逻辑回归模型的混淆矩阵图--李鑫煜')

# 添加轴标签
plt.xlabel('预测的类别')
plt.ylabel('真正的类别')

# 显示完整标签
plt.xticks(rotation=45)  # 旋转x轴标签，以便于阅读

# 显示图表
plt.show()

# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 精确率、召回率和F1分数（针对每个类别）
precision = precision_score(y_test, y_pred, average=None)  # None 返回每个类别的精确率
recall = recall_score(y_test, y_pred, average=None)        # None 返回每个类别的召回率
f1 = f1_score(y_test, y_pred, average=None)               # None 返回每个类别的F1分数

print(f"Precision (per class): {precision}")
print(f"Recall (per class): {recall}")
print(f"F1 Score (per class): {f1}")

# 支持度（每个类别在测试集中的样本数量）
support = np.bincount(y_test)
print(f"Support: {support}")
















