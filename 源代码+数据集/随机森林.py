import pandas as pd
lung_cancer_data = pd.read_csv('survey lung cancer.csv')

# Display the first few rows of the dataset and its summary
lung_cancer_data.head(), lung_cancer_data.info(), lung_cancer_data.describe()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Encode categorical variables
le_gender = LabelEncoder()  # 创建标签编码器
le_cancer = LabelEncoder()
lung_cancer_data['GENDER'] = le_gender.fit_transform(lung_cancer_data['GENDER'])  # 对'GENDER'列进行编码
lung_cancer_data['LUNG_CANCER'] = le_cancer.fit_transform(lung_cancer_data['LUNG_CANCER'])  # 对'LUNG_CANCER'列进行编码

X = lung_cancer_data.drop('LUNG_CANCER', axis=1)  # 定义特征变量
y = lung_cancer_data['LUNG_CANCER']  # 定义目标变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 将数据集分为训练集和测试集

from sklearn.ensemble import RandomForestClassifier

# Create and train the random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # 创建随机森林分类器
rf_clf.fit(X_train, y_train)  # 使用训练数据训练模型

# Predict on training and testing data
y_train_pred_rf = rf_clf.predict(X_train)  # 对训练数据进行预测
y_test_pred_rf = rf_clf.predict(X_test)  # 对测试数据进行预测

# Calculate accuracy for random forest
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)  # 计算训练集的准确率
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)  # 计算测试集的准确率

# Plotting accuracy
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure(figsize=(8, 4))  # 创建一个新的图形，设置其大小为8x4
plt.bar(['Training Set', 'Testing Set'], [train_accuracy_rf, test_accuracy_rf], color=['blue', 'green'])  # 创建一个条形图，显示训练集和测试集的准确率
plt.xlabel('Dataset')  # 设置x轴的标签为'Dataset'
plt.ylabel('Accuracy')  # 设置y轴的标签为'Accuracy'
plt.title('训练集和测试集上的随机森林模型精度-黄锦涛')  # 设置图形的标题
plt.ylim([0.85, 1])  # 设置y轴的范围为0.85到1
plt.show()  # 显示图形

# Extract feature importances
feature_importances_rf = rf_clf.feature_importances_  # 提取特征的重要性

# 创建一个新的图形，设置大小为12x8
plt.figure(figsize=(12, 8))

# 创建一个水平条形图，条形的长度由特征的重要性决定
# X.columns 是特征的名称，feature_importances_rf 是随机森林模型中每个特征的重要性
plt.barh(X.columns, feature_importances_rf, color='skyblue')

# 设置x轴的标签为 'Importance'
plt.xlabel('Importance')

# 设置y轴的标签为 'Features'
plt.ylabel('Features')

# 设置图形的标题为 '随机森林模型中的特征重要性-黄锦涛'
plt.title('随机森林模型中的特征重要性-黄锦涛')

# 显示图形
plt.show()

# 导入所需的库和函数
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns

# 使用随机森林分类器预测测试数据的概率，取第二列（即正类的概率）
y_test_probs_rf = rf_clf.predict_proba(X_test)[:, 1]

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_test_pred_rf)

# 创建一个新的图形，设置大小为6x6
plt.figure(figsize=(6, 6))

# 使用seaborn库的heatmap函数绘制混淆矩阵的热力图
# annot=True表示在每个单元格中添加注释，fmt='d'表示将数字格式化为整数，cmap='Blues'表示使用蓝色调的颜色映射
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# 设置图形的标题为 '混淆矩阵-黄锦涛'
plt.title('混淆矩阵-黄锦涛')

# 设置x轴的标签为 'Predicted Label'
plt.xlabel('Predicted Label')

# 设置y轴的标签为 'True Label'
plt.ylabel('True Label')

# 显示图形
plt.show()

# 使用roc_curve函数计算真正率（True Positive Rate）和假正率（False Positive Rate）
fpr, tpr, _ = roc_curve(y_test, y_test_probs_rf)

# 计算ROC曲线下的面积（AUC）
roc_auc = auc(fpr, tpr)

# 创建一个新的图形，设置大小为6x6
plt.figure(figsize=(6, 6))

# 绘制ROC曲线，颜色为深橙色，线宽为2
# 并添加标签，显示ROC曲线下的面积（AUC）
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

# 绘制对角线，颜色为海军蓝，线宽为2，线型为虚线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 设置x轴的范围为[0.0, 1.0]
plt.xlim([0.0, 1.0])

# 设置y轴的范围为[0.0, 1.05]
plt.ylim([0.0, 1.05])

# 设置x轴的标签为 'False Positive Rate'
plt.xlabel('False Positive Rate')

# 设置y轴的标签为 'True Positive Rate'
plt.ylabel('True Positive Rate')

# 设置图形的标题为 '随机森林的ROC曲线-黄锦涛'
plt.title('随机森林的ROC曲线-黄锦涛')

# 添加图例，位置在右下角
plt.legend(loc="lower right")

# 显示图形
plt.show()
