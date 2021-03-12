# Partially retrived from https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def print_results(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale data
    sc = StandardScaler()
    sc.fit(X_train.astype(float))
    X_train = sc.transform(X_train.astype(float))
    X_test = sc.transform(X_test.astype(float))

    # Decision tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict_proba(X_test)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
    roc_auc_dt = auc(fpr_dt, tpr_dt)

    # Random forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Support vector machine 
    svm = SVC(gamma='auto', kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict_proba(X_test)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    # plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_dt, tpr_dt, label='Decision Tree')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot(fpr_svm, tpr_svm, label='SVC')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_dt, tpr_dt, label='Decision Tree')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot(fpr_svm, tpr_svm, label='SVC')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()

    # print accuracy 
    print("Accuracy scores")
    print("Decision Tree: " + str(roc_auc_dt))
    print("Random Forest: " + str(roc_auc_rf))
    print("SVC: " + str(roc_auc_svm))

    # print importances
    importances = dt.tree_.compute_feature_importances(normalize=False)
    indices = np.argsort(importances)[::-1]
    names = [X.columns[i] for i in indices]
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90)
    plt.show()