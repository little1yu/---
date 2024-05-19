import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
import warnings
from sklearn.metrics import precision_score, recall_score
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

lung_cancer_data = pd.read_csv('survey lung cancer.csv')

numerical_columns = lung_cancer_data.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(20, 10))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, len(numerical_columns)//3 + 1, i)
    sns.histplot(data=lung_cancer_data, x=column, bins=30, kde=True)
    plt.title(column)

plt.tight_layout()
plt.show()

le_gender = LabelEncoder()
le_cancer = LabelEncoder()
lung_cancer_data['GENDER'] = le_gender.fit_transform(lung_cancer_data['GENDER'])  # 性别编码
lung_cancer_data['LUNG_CANCER'] = le_cancer.fit_transform(lung_cancer_data['LUNG_CANCER'])  # 肺癌状态编码

X = lung_cancer_data.drop('LUNG_CANCER', axis=1)  # 特征数据，去掉LUNG_CANCER列
y = lung_cancer_data['LUNG_CANCER']  # 目标数据，即肺癌状态

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_clf = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
svm_clf.fit(X_train, y_train)  # 训练模型

y_train_pred_svm = svm_clf.predict(X_train)  # 训练集预测
y_test_pred_svm = svm_clf.predict(X_test)  # 测试集预

train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)  # 训练集准确率
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)  # 测试集准确率

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure(figsize=(8, 4))  # 设置图形的大小
bar = plt.bar(['Training Set', 'Testing Set'], [train_accuracy_svm, test_accuracy_svm], color=['blue', 'green'])  # 绘制柱状图

# 为每个柱状添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, round(height, 5), ha='center', va='bottom')

add_value_labels(bar)

plt.xlabel('Dataset')  # 设置x轴标签
plt.ylabel('Accuracy')  # 设置y轴标签
plt.title('训练集和测试集上的模型准确率图-李鑫煜')  # 设置图表标题
plt.ylim([0.85, 1])  # 设置y轴的范围
plt.show()      # 显示图表

y_train_proba_svm = svm_clf.predict_proba(X_train)[:, 1]  # 获取训练集的预测概率
y_test_proba_svm = svm_clf.predict_proba(X_test)[:, 1]  # 获取测试集的预测概率

fpr_train_svm, tpr_train_svm, _ = roc_curve(y_train, y_train_proba_svm)  # 训练集的ROC曲线
auc_train_svm = auc(fpr_train_svm, tpr_train_svm)  # 训练集的AUC值
fpr_test_svm, tpr_test_svm, _ = roc_curve(y_test, y_test_proba_svm)  # 测试集的ROC曲线
auc_test_svm = auc(fpr_test_svm, tpr_test_svm)  # 测试集的AUC值

plt.figure(figsize=(12, 6))  # 设置图形的大小
plt.plot(fpr_train_svm, tpr_train_svm, label=f'Train AUC = {auc_train_svm:.2f}')  # 绘制训练集的ROC曲线
plt.plot(fpr_test_svm, tpr_test_svm, label=f'Test AUC = {auc_test_svm:.2f}')  # 绘制测试集的ROC曲线
plt.plot([0, 1], [0, 1], 'k--')  # 绘制参考线
plt.xlabel('False Positive Rate')  # 设置x轴标签
plt.ylabel('True Positive Rate')  # 设置y轴标签
plt.title('ROC曲线-李鑫煜')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()

X_train_2d = X_train.iloc[:, :2]  # 取前两个特征进行二维可视化
svm_clf_2d = make_pipeline(StandardScaler(), SVC(random_state=42))  # 创建SVM分类器的管道
svm_clf_2d.fit(X_train_2d, y_train)  # 训练模型

plt.figure(figsize=(12, 8))  # 设置图形的大小
plot_decision_regions(X_train_2d.values, y_train.values, clf=svm_clf_2d, legend=2)  # 绘制决策边界
plt.xlabel(X_train_2d.columns[0])  # 设置x轴标签
plt.ylabel(X_train_2d.columns[1])  # 设置y轴标签
plt.title('决策边界和数据点可视化--李鑫煜')  
plt.show()

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_test_pred_svm)

# 计算召回率和精确率
# 由于这是二分类问题，我们指定pos_label为1，即肺癌状态为正类
recall = recall_score(y_test, y_test_pred_svm, pos_label=1)
precision = precision_score(y_test, y_test_pred_svm, pos_label=1)
# 打印召回率和精确率
print(f"Recall (True Positive Rate): {recall:.2f}")

print(f"Precision: {precision:.2f}")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 使用seaborn绘制热图
plt.title('SVM 模型的混淆矩阵--李鑫煜')  # 设置图表标题
plt.xlabel('Predicted Label')  # 设置x轴标签
plt.ylabel('True Label')  # 设置y轴标签
plt.show()
