import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import learning_curve

lung_cancer_data = pd.read_csv('survey lung cancer.csv')
lung_cancer_data.head()

lung_cancer_data.info()

lung_cancer_data.describe()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
le_gender = LabelEncoder()
le_cancer = LabelEncoder()
lung_cancer_data['GENDER'] = le_gender.fit_transform(lung_cancer_data['GENDER'])
lung_cancer_data['LUNG_CANCER'] = le_cancer.fit_transform(lung_cancer_data['LUNG_CANCER'])

X = lung_cancer_data.drop('LUNG_CANCER', axis=1)
y = lung_cancer_data['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

y_train_pred_knn = knn_clf.predict(X_train)
y_test_pred_knn = knn_clf.predict(X_test)

y_train_proba_knn = knn_clf.predict_proba(X_train)[:, 1]
y_test_proba_knn = knn_clf.predict_proba(X_test)[:, 1]

train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
fpr_train_knn, tpr_train_knn, _ = roc_curve(y_train, y_train_proba_knn)
auc_train_knn = auc(fpr_train_knn, tpr_train_knn)
fpr_test_knn, tpr_test_knn, _ = roc_curve(y_test, y_test_proba_knn)
auc_test_knn = auc(fpr_test_knn, tpr_test_knn)

plt.figure(figsize=(8, 4))
plt.bar(['Training Set', 'Testing Set'], [train_accuracy_knn, test_accuracy_knn], color=['blue', 'green'])
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('KNN Model Accuracy on Training and Testing Sets-林钰鑫')
plt.ylim([0.85, 1])
plt.show()

# 将数据点在二维空间中使用PCA进行降维可视化。
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA-Reduced Data Points for KNN-林钰鑫')
plt.colorbar(scatter)
plt.show()

# Calculate confusion matrix for training and testing sets
cm_train_knn = confusion_matrix(y_train, y_train_pred_knn)
cm_test_knn = confusion_matrix(y_test, y_test_pred_knn)

# Plot confusion matrix for training set
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Training Set--林钰鑫')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot confusion matrix for testing set
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Testing Set-林钰鑫')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import precision_recall_curve

# Calculate precision and recall for KNN
precision_train_knn, recall_train_knn, _ = precision_recall_curve(y_train, y_train_proba_knn)
precision_test_knn, recall_test_knn, _ = precision_recall_curve(y_test, y_test_proba_knn)

# Plotting PR curve
plt.figure(figsize=(12, 6))
plt.plot(recall_train_knn, precision_train_knn, label='Train PR curve', color='navy')
plt.plot(recall_test_knn, precision_test_knn, label='Test PR curve', color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for KNN-林钰鑫')
plt.legend()
plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves (KNN)-林钰鑫"
plot_learning_curve(knn_clf, title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=-1)

plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score for KNN
precision_knn = precision_score(y_test, y_test_pred_knn)
recall_knn = recall_score(y_test, y_test_pred_knn)
f1_knn = f1_score(y_test, y_test_pred_knn)

# Plotting precision, recall, and F1 score with numerical values
plt.figure(figsize=(8, 6))
metrics = ['Precision', 'Recall', 'F1 Score']
scores = [precision_knn, recall_knn, f1_knn]
bars = plt.bar(metrics, scores, color=['purple', 'orange', 'red'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score for KNN on Testing Set')
plt.ylim([0, 1])

# Add numerical values to the bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{score:.2f}', ha='center', va='center')

plt.show()