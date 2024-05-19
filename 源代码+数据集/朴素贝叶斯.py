import pandas as pd

#加载数据集
lung_cancer_data = pd.read_csv('survey lung cancer.csv')

# 显示数据集的前5行信息
lung_cancer_data.head()

#获取数据集的简单描述，如总行数，每个属性的类型和非空值的数量
lung_cancer_data.info()

#显示属性摘要
lung_cancer_data.describe()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB

#对分类变量进行编码
le_gender = LabelEncoder()
le_cancer = LabelEncoder()

# 使用LabelEncoder对'GENDER'列和'LUNG_CANCER'列进行编码，将性别文本和肺癌状态文本转换为数值
lung_cancer_data['GENDER'] = le_gender.fit_transform(lung_cancer_data['GENDER'])
lung_cancer_data['LUNG_CANCER'] = le_cancer.fit_transform(lung_cancer_data['LUNG_CANCER'])

#定义特征和目标变量
X = lung_cancer_data.drop('LUNG_CANCER', axis=1)# 特征X是除了'LUNG_CANCER'列的所有列 
y = lung_cancer_data['LUNG_CANCER']# 目标变量y是'LUNG_CANCER'列 

#将数据集分割为训练集和测试集，测试集占30%，随机种子设为42 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#训练朴素贝叶斯模型
nb_clf = GaussianNB()# 创建一个高斯朴素贝叶斯分类器实例
nb_clf.fit(X_train, y_train)# 使用训练数据拟合（训练）朴素贝叶斯模型 

#创建一个条件概率表
#这里我们为每一个特征在'没有癌症'和'有癌症'两个类别下的平均值创建一个表格来进行演示
prob_table = pd.DataFrame({
    
    # 'Feature'列存储特征名称，直接从X_train的列名中获取
    'Feature': X_train.columns,
    
    # 计算每个特征在'没有癌症'（y_train == 0）类别下的平均值
    'Mean in No Cancer': [X_train.loc[y_train == 0, f].mean() for f in X_train.columns],
    
    # 计算每个特征在'有癌症'（y_train == 1）类别下的平均值
    'Mean in Cancer': [X_train.loc[y_train == 1, f].mean() for f in X_train.columns]
})

# 根据类别标签分组计算每个特征的平均值和标准差  
# 创建一个新的DataFrame来存储每个特征在不同类别下的平均值和标准差
prob_stats = pd.DataFrame({
    
    # 'Feature'列存储特征名称，直接从X_train的列名中获取
    'Feature': X_train.columns,
    
    # 计算每个特征在'没有癌症'类别下的平均值，注意这里使用了整体计算而非列表推导
    'Mean in No Cancer': X_train[y_train == 0].mean(),
    
    # 计算每个特征在'没有癌症'类别下的标准差
    'Std in No Cancer': X_train[y_train == 0].std(),
    
    # 计算每个特征在'有癌症'类别下的平均值
    'Mean in Cancer': X_train[y_train == 1].mean(),
    
    # 计算每个特征在'有癌症'类别下的标准差 
    'Std in Cancer': X_train[y_train == 1].std()
}).reset_index(drop=True)

#显示条件概率统计表
print(prob_stats)

#使用条形图绘制条件概率统计的平均值，并使用误差条表示标准差
#创建一个新的matplotlib图形，用于绘制条形图，并设置图形的大小为20x12单位  
plt.figure(figsize=(20, 12))  

# 为整个图形设置一个大标题  
plt.suptitle('癌症与非癌症条件下的特征平均值比较-刘佳伦', fontsize=20) 

# 遍历prob_stats DataFrame中的每一行（即每个特征）  
# 使用iterrows()方法来迭代DataFrame，该方法返回索引和行数据  
for i, row in prob_stats.iterrows():  
    # 为每个特征创建一个子图，在4x4的网格布局中  
    # i + 1 是因为subplot的索引从1开始，而不是0  
    # 这里的i是当前迭代的行数（从0开始），因此i + 1将用作subplot的索引  
    plt.subplot(4, 4, i + 1)  
      
    # 准备要在条形图中展示的数据  
    # means列表包含'没有癌症'和'有癌症'两个类别下的平均值  
    means = [row['Mean in No Cancer'], row['Mean in Cancer']]  
    # errors列表包含'没有癌症'和'有癌症'两个类别下的标准差，用于绘制误差条  
    errors = [row['Std in No Cancer'], row['Std in Cancer']]  
      
    # 绘制条形图，其中'No Cancer'和'Cancer'是条形图的x轴标签  
    # means是条形的高度，yerr=errors表示误差条的高度（标准差）  
    # capsize=5设置误差条帽子的大小，color设置条形的颜色  
    plt.bar(['No Cancer', 'Cancer'], means, yerr=errors, capsize=5, color=['blue', 'green'])  
      
    # 设置当前子图的标题为当前特征的名称  
    plt.title(row['Feature'])  
      
    # 设置y轴的标签  
    plt.ylabel('Value')  
      
    # 显示网格线  
    plt.grid(True)  
  
# 调整子图之间的间距，使之更加紧凑，避免重叠  
plt.tight_layout()  
  
# 显示整个图形  
plt.show()

import seaborn as sns  
  
# 设置matplotlib的配置参数，以确保负号能够正常显示
plt.rcParams['axes.unicode_minus'] = False  
  
# 创建一个新的matplotlib图形，并设置其大小为20x12
plt.figure(figsize=(20, 12))  
  
# 遍历X_train中的每一个特征（列）  
# enumerate函数会同时返回特征的索引i和特征名feature  
for i, feature in enumerate(X_train.columns):  
    
    # 为每个特征创建一个小提琴图，并放置在4x4的网格布局中  
    # i + 1用于指定子图的位置，因为subplot的索引是从1开始的  
    plt.subplot(4, 4, i + 1)  
      
    # 使用seaborn的violinplot函数绘制小提琴图  
    # x轴表示类别（这里是y_train，即目标变量的值），y轴表示特征的值  
    # palette="muted"设置了图形的调色板为柔和的色调  
    sns.violinplot(x=y_train, y=X_train[feature], palette="muted")  
      
    # 设置当前子图的标题为当前特征的名称  
    plt.title(feature)  

# 设置一个大标题  
plt.suptitle('特征与目标变量关系的小提琴图分析-刘佳伦', fontsize=20)  # 设置大标题内容和字体大小

# 调整子图布局，确保所有子图之间有足够的空间，并且不会被裁剪或重叠  
plt.tight_layout()  
  
# 显示整个图形  
plt.show()

#使用朴素贝叶斯对训练数据和测试数据进行预测
y_train_pred_nb = nb_clf.predict(X_train)
y_test_pred_nb = nb_clf.predict(X_test)

#使用朴素贝叶斯模型预测训练数据和测试数据中为正类的概率
y_train_proba_nb = nb_clf.predict_proba(X_train)[:, 1]
y_test_proba_nb = nb_clf.predict_proba(X_test)[:, 1]

#计算朴素贝叶斯模型在训练集和测试集上的准确率
train_accuracy_nb = accuracy_score(y_train, y_train_pred_nb)
test_accuracy_nb = accuracy_score(y_test, y_test_pred_nb)

#ROC曲线和AUC在训练集和测试集上的应用
fpr_train_nb, tpr_train_nb, _ = roc_curve(y_train, y_train_proba_nb)# 计算训练集上的ROC曲线相关数据
auc_train_nb = auc(fpr_train_nb, tpr_train_nb)# 计算训练集上的AUC值
fpr_test_nb, tpr_test_nb, _ = roc_curve(y_test, y_test_proba_nb)# 计算测试集上的ROC曲线相关数据
auc_test_nb = auc(fpr_test_nb, tpr_test_nb)# 计算测试集上的AUC值

#为每个类别绘制概率分布
plt.figure(figsize=(12, 6))# 创建一个新的matplotlib图形，用于绘制概率分布，并设置图形大小为12x6单位

# 绘制'没有癌症'类别的预测概率直方图  
# 选择训练集中真实标签为'没有癌症'（即y_train == 0）的样本，提取它们的预测概率  
# 使用30个柱子（bins）来绘制直方图，设置透明度（alpha）为1.0，并标记为'No Cancer' 
plt.hist(y_train_proba_nb[y_train == 0], bins=30, alpha=1.0, label='No Cancer')

# 同样地，绘制'有癌症'类别的预测概率直方图
plt.hist(y_train_proba_nb[y_train == 1], bins=30, alpha=0.5, label='Cancer')

plt.xlabel('Predicted Probability')# 设置x轴标签为'预测概率'
plt.ylabel('Frequency')# 设置y轴标签为'频数'
plt.title('两个类别的概率分布-刘佳伦') # 设置图形的标题
plt.legend()# 显示图例，即上面设置的'No Cancer'和'Cancer'标签
plt.show()# 显示图形

#绘制准确率图表
plt.rcParams['font.sans-serif'] = ['SimHei']# 设置matplotlib的字体为'SimHei'，以便正确显示中文
plt.figure(figsize=(10, 5))# 创建一个新的matplotlib图形，并设置其大小为10x5单位
bars=plt.bar(['Training Set', 'Testing Set'], [train_accuracy_nb, test_accuracy_nb], color=['blue', 'green'])# 绘制一个条形图，展示训练集和测试集上的准确率 

# 为每个条形图添加数值标签  
def add_labels(bars):  
    for bar in bars:  
        height = bar.get_height()  
        plt.text(bar.get_x() + bar.get_width() / 2, height,  
                 f'{height:.2f}',  # 保留两位小数  
                 ha='center', va='bottom', fontsize=10)  # 设置标签的水平对齐方式、垂直对齐方式和字体大小  
  
add_labels(bars)  

plt.xlabel('Dataset')# 设置x轴标签为'数据集' 
plt.ylabel('Accuracy') # 设置y轴标签为'准确率'
plt.title('训练集和测试集上的朴素贝叶斯模型精度-刘佳伦')#设置图形标题
plt.ylim([0.8, 1])# 设置y轴的范围为[0.8, 1]  
plt.show()#显示图形

import matplotlib.pyplot as plt  
from sklearn.metrics import precision_score, recall_score, f1_score  
precision = precision_score(y_test, y_test_pred_nb)  #计算精度
recall = recall_score(y_test,y_test_pred_nb)  #计算召回率
f1 = f1_score(y_test, y_test_pred_nb)  #计算F1分数
labels = ['Precision', 'Recall', 'F1 Score']  
values = [precision, recall, f1]  
  
fig, ax = plt.subplots()  
  
# 创建条形图  
ax.barh(labels, values, color='skyblue')  
  
# 设置图表标题和坐标轴标签  
ax.set_title('模型评估指标-刘佳伦')  
ax.set_xlabel('Score')  
ax.set_ylabel('Metric')  
  
# 显示每个条形图上的数值标签  
for index, value in enumerate(values):  
    plt.text(value, index, str(round(value, 2)), va='center')  
  
# 显示图表  
plt.show()

#绘制ROC曲线
plt.figure(figsize=(12, 6)) # 创建一个新的matplotlib图形，并设置其大小为12x6单位 

# 绘制训练集的ROC曲线，其中fpr_train_nb和tpr_train_nb分别是训练集上的假正率和真正率  
# 同时，在图上显示训练集的AUC（Area Under the Curve，曲线下面积）值，保留两位小数  
# label参数用于设置图例中的标签  
plt.plot(fpr_train_nb, tpr_train_nb, label=f'Train AUC = {auc_train_nb:.2f}')

# 绘制测试集的ROC曲线，其中fpr_test_nb和tpr_test_nb分别是测试集上的假正率和真正率  
# 同时，在图上显示测试集的AUC值，保留两位小数  
plt.plot(fpr_test_nb, tpr_test_nb, label=f'Test AUC = {auc_test_nb:.2f}')

plt.plot([0, 1], [0, 1], 'k--')# 绘制一条从(0,0)到(1,1)的对角线，表示一个无信息的分类器（即随机猜测）的性能 
plt.xlabel('False Positive Rate')# 设置x轴的标签为'False Positive Rate'（假正率）
plt.ylabel('True Positive Rate')# 设置y轴的标签为'True Positive Rate'（真正率）  
plt.title('朴素贝叶斯的ROC曲线-刘佳伦') # 设置图形的标题
plt.legend()# 显示图例，其中包含了训练集和测试集的AUC值 
plt.show()#显示图形

