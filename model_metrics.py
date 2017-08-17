from feature_engineering import *
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_curve, auc, matthews_corrcoef
from pickle_io import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def metric_threshold_plot(test_prob, y_true, model_name, thresh_length = 100, save_path = None):
    thresh = np.linspace(np.min(test_prob), np.max(test_prob), thresh_length)
    y_pred = np.zeros(test_prob.shape)
    accuracy = np.zeros(thresh.shape)
    precision = np.zeros(thresh.shape)
    f1 = np.zeros(thresh.shape)
    recall = np.zeros(thresh.shape)
    mcc = np.zeros(thresh.shape)
    for n, p in enumerate(thresh):
        y_pred[:] = 0
        y_pred[test_prob >= p] = 1
        accuracy[n] = accuracy_score(y_true, y_pred)
        precision[n] = precision_score(y_true, y_pred)
        f1[n] = f1_score(y_true, y_pred)
        recall[n] = recall_score(y_true, y_pred)
        mcc[n] = matthews_corrcoef(y_true, y_pred)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    ax.plot(thresh, accuracy, label="Accuracy")
    ax.plot(thresh, precision, label="Precision")
    ax.plot(thresh, recall, label="Recall")
    ax.plot(thresh, f1, label="F1")
    ax.plot(thresh, mcc, label="MCC")

    ax.set_xlabel("Probability Threshold", fontsize = 20)
    ax.set_xticks(np.arange(0,1.1,0.1))

    ax.set_ylabel("Metric", fontsize = 20)
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylim(0, 1.1)

    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.set_title("{0} Model Metrics".format(model_name), fontsize=25)

    ax.legend(fontsize=20, ncol=5, loc="upper center")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    fig.show()

def create_roc_curve():
    lr = pickle_load("models/logistic_regression.pkl")
    rf = pickle_load("models/random_forest_30.pkl")
    gb = pickle_load("models/gradient_boost.pkl")
    lda = pickle_load("models/linear_discriminant_analysis.pkl")

    lr_prob = lr.predict_proba(X_test_transform)[:,1]
    lr_fpr, lr_tpr, lr_thresh = roc_curve(y_test_transform, lr_prob, pos_label = 1)
    lr_auc = auc(lr_fpr, lr_tpr)

    rf_prob = rf.predict_proba(X_test_transform)[:,1]
    rf_fpr, rf_tpr, rf_thresh = roc_curve(y_test_transform, rf_prob, pos_label = 1)
    rf_auc = auc(rf_fpr, rf_tpr)

    gb_prob = gb.predict_proba(X_test_transform)[:,1]
    gb_fpr, gb_tpr, gb_thresh = roc_curve(y_test_transform, gb_prob, pos_label = 1)
    gb_auc = auc(gb_fpr, gb_tpr)

    lda_prob = lda.predict_proba(X_test_transform)[:,1]
    lda_fpr, lda_tpr, lda_thresh = roc_curve(y_test_transform, lda_prob, pos_label = 1)
    lda_auc = auc(lda_fpr, lda_tpr)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    ax.plot(lr_fpr, lr_tpr, label="Logistic Regression (AUC = %0.2f)" % lr_auc)
    ax.plot(rf_fpr, rf_tpr, label="Random Forest (AUC = %0.2f)" % rf_auc)
    ax.plot(gb_fpr, gb_tpr, label="Gradient Boost (AUC = %0.2f)" % gb_auc)
    ax.plot(lda_fpr, lda_tpr, label="Linear Discriminant Analysis (AUC = %0.2f)" % lda_auc)
    ax.plot([0,1], [0,1], "k--")

    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_xticks(np.arange(0,1.1,0.1))
    ax.set_ylabel("True Positive Rate", fontsize=20)
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title("ROC Curve", fontsize=25)

    ax.legend(fontsize=20)

    plt.tight_layout()
    plt.savefig("images/model_metrics/roc_curve.png")
    fig.show()

if __name__ == '__main__':
    # lr = pickle_load("models/logistic_regression.pkl")
    # metric_threshold_plot(test_prob = lr.predict_proba(X_test_transform)[:,1],
    #                       y_true = y_test_transform,
    #                       model_name = "Logistic Regression",
    #                       thresh_length = 200,
    #                       save_path = "images/model_metrics/logistic_regression_model_metrics.png")
    #
    # rf = pickle_load("models/random_forest_30.pkl")
    # metric_threshold_plot(test_prob = rf.predict_proba(X_test_transform)[:,1],
    #                       y_true = y_test_transform,
    #                       model_name ="Random Forest",
    #                       thresh_length = 200,
    #                       save_path = "images/model_metrics/random_forest_model_metrics.png")
    #
    # gb = pickle_load("models/gradient_boost.pkl")
    # metric_threshold_plot(test_prob = gb.predict_proba(X_test_transform)[:,1],
    #                       y_true = y_test_transform,
    #                       model_name ="Gradient Boost",
    #                       thresh_length = 200,
    #                       save_path = "images/model_metrics/gradient_boost_model_metrics.png")
    #
    # lda = pickle_load("models/linear_discriminant_analysis.pkl")
    # metric_threshold_plot(test_prob = lda.predict_proba(X_test_transform)[:,1],
    #                       y_true = y_test_transform,
    #                       model_name = "Linear Discriminant Analysis",
    #                       thresh_length = 200,
    #                       save_path = "images/model_metrics/lda_model_metrics.png")

    create_roc_curve()
