import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


def auroc(y_pred, y_true):
    y_true = y_true.get_label()
    length = len(y_true)
    
    return 'roc_auc', roc_auc_score(y_true, y_pred, average="macro"), True


def train_model(train_ds, train_labels, test_ds, test_labels):
    N_SPLITS = 10

    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2023)

    test_preds = 0
    feature_imp = 0
    mean_val = 0

    for i, (train_idx,val_idx) in enumerate(kfold.split(train_ds,train_labels)):
        print('\n\n--------------------- Running {} of KFold {}-------------------------'.format(i+1,kfold.n_splits))
        xtr,xvl = train_ds.loc[train_idx],train_ds.loc[val_idx]
        ytr,yvl = train_labels[train_idx],train_labels[val_idx]
        
        params = {
            'objective': 'cross_entropy',
            'seed': 2023,
            'num_iterations' : 1000,
            'num_leaves': 100,
            'learning_rate': 0.01,
            'feature_fraction': 0.20,
            'bagging_freq': 10,
            'bagging_fraction': 0.50,
            'n_jobs': -1,
            'lambda_l2': 2,
            'min_data_in_leaf': 40,
            'device_type' : "cpu",
            'eval_metric' : 'auc',
            'verbose' : -1
            }
        
        lgb_train = lgb.Dataset(xtr, ytr) 
        lgb_valid = lgb.Dataset(xvl, yvl)
        model = lgb.train(
                params = params,
                train_set = lgb_train,
                num_boost_round = 1000,
                valid_sets = [lgb_train, lgb_valid],
                early_stopping_rounds = 200,
                verbose_eval=200,
                feval = auroc
                )

        test_preds += model.predict(test_ds)
        
        val_preds = model.predict(xvl)
        val_rocauc = roc_auc_score(yvl,val_preds, average="macro")
        mean_val += val_rocauc
        print(f"Val Predictions: {np.bincount([np.max(i) for i in val_preds])}")
        print("\nValidation Dataset - ROC AUC Score = {}".format(val_rocauc))
        
        feature_imp += model.feature_importance()  


    test_preds = test_preds/N_SPLITS
    test_score = roc_auc_score(test_labels,test_preds, average="macro")
    feature_imp = list(feature_imp/N_SPLITS)
    mean_val = mean_val / N_SPLITS
    print(f"Test Preds: {test_preds}")
    print(f"\n\nVal ROC AUC: {mean_val}")
    print(f"Test ROC AUC: {test_score}")
    #print(f"Feature Importance: {feature_imp}")

    return model
