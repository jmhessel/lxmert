import sklearn.metrics
import numpy as np
import collections


def get_metrics_binary(pred, pred_bin, te_y):
    pred, pred_bin, te_y = map(lambda x: np.array(x).flatten(),
                               [pred, pred_bin, te_y])
    if np.sum(np.isnan(pred)) > 0:
        return {'f1':-1, 'acc':-1, 'roc_auc':-1, 'precision':-1, 'recall':-1}
    f1 = (sklearn.metrics.f1_score(te_y, pred_bin)
          if np.mean(pred_bin) != 0 and np.mean(pred_bin) != 1
          else 0)
    
    acc = sklearn.metrics.accuracy_score(te_y, pred_bin)
    roc_auc = sklearn.metrics.roc_auc_score(te_y, pred)
    prec = (sklearn.metrics.precision_score(te_y, pred_bin)
            if np.mean(pred_bin) != 0 and np.mean(pred_bin) != 1
            else 0)
    rec = sklearn.metrics.recall_score(te_y, pred_bin)
    return {'f1':f1, 'acc':acc, 'roc_auc':roc_auc, 'precision':prec, 'recall':rec}


def get_metrics_multiclass(pred, pred_clf, te_y):
    res = sklearn.metrics.classification_report(te_y, pred_clf, output_dict=True)

    classes = set(te_y)
    per_class_acc = []
    for c in classes:
        test_idxs = te_y == c
        per_class_acc.append(np.mean(pred_clf[test_idxs] == te_y[test_idxs]))
        
    accuracy = np.mean(pred_clf == te_y)
    
    macro_average_stat = collections.defaultdict(list)
    for label, metric_dict in res.items():
        if not type(metric_dict) is dict:
            continue
        for m, v in metric_dict.items():
            if m != 'support':
                macro_average_stat[m].append(v)

    res = {}
    for m, stats in macro_average_stat.items():
        res['macro_average_' + m] = np.mean(stats)
    res['macro_acc'] = np.mean(per_class_acc)
    res['acc'] = accuracy
    print(pred[:4])
    res['macro_auc'] = sklearn.metrics.roc_auc_score(
        sklearn.preprocessing.label_binarize(te_y, list(range(len(classes)))),
        pred,
        )
    res['weighted_average_f1'] = sklearn.metrics.f1_score(
        te_y,
        pred_clf,
        average='weighted')
    return res
