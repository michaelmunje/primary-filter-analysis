from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def print_vif(df):
    X = add_constant(df)
    for i, vif in enumerate([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]):
        if vif > 10:
            print(df.columns[i] + ' VIF: ' + str(round(vif, 3)))


def print_condition_number(df):
    X = np.insert(np.array(df.values), 0, 1, axis=1)
    xpx = np.matmul(np.transpose(X), X)
    eigvals = [np.real(eig) for eig in np.linalg.eigvals(xpx)]
    print('Condition Number:' + str(abs(max(eigvals) / min(eigvals))))


def print_strong_correlations(df):
    corr = df.corr().values

    for i in range(len(df.columns) - 1):
        for j in range(i, len(df.columns) - 1):
            if i != j:
                if abs(corr[i, j]) > 0.9:
                    print('Strong correlation between ' + df.columns[i] + ' and ' + df.columns[
                        j] + ': ' + str(round(corr[i, j], 3)))


def print_log_regression_metrics(X, Y, cv=False):
    if cv:
        log_reg = LogisticRegressionCV(multi_class='multinomial', solver='newton-cg', max_iter=500)
    else:
        log_reg = LogisticRegression(multi_class='multinomial', penalty='none', solver='newton-cg', max_iter=500)
    log_reg.fit(X, Y)

    print('Accuracy: ' + str(round(accuracy_score(Y, log_reg.predict(X)),  3)))
    print(confusion_matrix(Y, log_reg.predict(X)))


def plot_roc_precision_recall(classifier, name, X, y, groups):
    fig = plt.figure(1, figsize=(16, 6))
    axes = fig.add_subplot(121), fig.add_subplot(122)
    plot_precision_recall_ci(classifier, name, X, y, axes[1], groups)
    plot_roc_ci(classifier, name, X, y, axes[0], groups)


def plot_precision_recall_ci(classifier, name, X, y, ax, groups):
    num_splits = 10

    cv = GroupShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=1337)

    pre = []
    aucs = []
    mean_rc = np.linspace(0, 1, 1000)

    splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)

    for i, (train, test) in enumerate(splits):
        classifier.fit(X[train], y[train])
        predict = classifier.predict_proba(X[test])[:, 1]
        precision, recall, thresholds = precision_recall_curve(y[test], predict)
        pr_auc = auc(recall, precision)
        interp_pre = interp(mean_rc, precision, recall)
        interp_pre[0] = 1.0
        pre.append(interp_pre)
        aucs.append(pr_auc)

    chance = len(y[y == 1]) / len(y)

    mean_pre = np.mean(pre, axis=0)
    mean_pre[-1] = 0.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs) * 2
    ax.plot(mean_rc, mean_pre, color='b',
            label=r'AUC = %0.3f $\pm$ %0.3f' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_pre = np.std(pre, axis=0)
    pre_upper = np.minimum(mean_pre + std_pre * 2, 1)
    pre_lower = np.maximum(mean_pre - std_pre * 2, 0)
    ax.fill_between(mean_rc, pre_lower, pre_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 2 std. devs.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Precision-Recall Curve: " + name)
    ax.legend(loc="lower left")
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')


def plot_roc_ci(classifier, name, X, y, ax, groups=None, trained=False):
    num_splits = 10

    if groups is not None:
        cv = GroupShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=1337)
    else:
        cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1337)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)

    tp_rate = []
    tn_rate = []

    splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for i, (train, test) in enumerate(splits):
        if not trained:
            classifier.fit(X[train], y[train])
        predict = classifier.predict_proba(X[test])[:, 1]
        fpr, tpr, thresholds = roc_curve(y[test], predict)
        roc_auc = auc(fpr, tpr)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        model_predict = [1 if p >= 0.5 else 0 for p in predict]
        tn_rate.append(recall_score(y[test], model_predict))
        tp_rate.append(recall_score([1 - p for p in y[test]], [1 - p for p in model_predict]))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs) * 2
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr * 2, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr * 2, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 2 std. devs.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC Curve: " + name)
    ax.legend(loc="lower right")
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    plt.show()

    fp_rate = [1 - x for x in tp_rate]
    fn_rate = [1 - x for x in tn_rate]

    fpr_avg = np.mean(fp_rate)
    fnr_avg = np.mean(fn_rate)
    fpr_ci = np.std(fp_rate) * 2
    fnr_ci = np.std(fn_rate) * 2
    splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    print(f'False Positive Rate: {fpr_avg:.2f} +- {fpr_ci:.2f}')
    print(f'False Negative Rate: {fnr_avg:.2f} +- {fnr_ci:.2f}')