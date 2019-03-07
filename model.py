import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


class ModelRun(object):
    def __init__(self):
        self.n_fold = 5
        self.kf = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
        self.eval_fun = roc_auc_score

    def run_oof(self, clf, X_train, y_train, X_test):
        print(clf.get_name())
        preds_train = np.zeros((len(X_train)), dtype=np.float)
        preds_test = np.zeros((len(X_test)), dtype=np.float)
        train_loss = []
        test_loss = []

        i = 1
        for train_index, test_index in self.kf.split(X_train, y_train):
            x_tr = X_train[train_index]
            x_te = X_train[test_index]
            y_tr = y_train[train_index]
            y_te = y_train[test_index]

            if clf.get_name() in ['lightGBM', 'XGBoost']:
                clf.fit(x_tr, y_tr, x_te, y_te)

                train_loss.append(self.eval_fun(y_tr, clf.predict(x_tr)))
                test_loss.append(self.eval_fun(y_te, clf.predict(x_te)))

                preds_train[test_index] = clf.predict(x_te)
                preds_test += clf.predict(X_test)
            else:
                clf.fit(x_tr, y_tr)

                train_loss.append(self.eval_fun(y_tr, clf.predict(x_tr)))
                test_loss.append(self.eval_fun(y_te, clf.predict(x_te)))

                preds_train[test_index] = clf.predict(x_te)
                preds_test += clf.predict(X_test)

            print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(i, train_loss[-1], test_loss[-1],
                                                                     np.mean(test_loss)))
            print('-' * 50)
            i += 1
        print('Train: ', train_loss)
        print('Val: ', test_loss)
        print('-' * 50)
        print('{0} Train{1:0.5f}_Test{2:0.5f}\n\n'.format(clf.get_name(), np.mean(train_loss), np.mean(test_loss)))
        preds_test /= self.n_fold
        return preds_train, preds_test


class SklearnWrapper(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf

    def fit(self, X_train, y_train, X_val=None, y_val=None, eval_func=None):
        self.clf.fit(X_train, y_train)

    def predict_proba(self, x):
        proba = self.clf.predict_proba(x)
        return proba

    def predict(self, x):
        return self.clf.predict(x)

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func, is_proba):
        def fun(**params):
            for k in params:
                if type(param_grid[k][0]) is int:
                    params[k] = int(params[k])

            self.clf.set_params(**params)
            self.fit(X_train, y_train)

            if is_proba:
                p_eval = self.predict_proba(X_eval)
            else:
                p_eval = self.predict(X_eval)

            best_score = eval_func(y_eval, p_eval)

            return -best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['max']['max_params']))

    def get_name(self):
        return self.clf.__class__.__name__

    def report(self, y, y_pred):
        print(self.get_name() + ' report：\n', classification_report(y, y_pred))
        print(self.get_name() + ' AUC：\n', roc_auc_score(y, y_pred))
        

class XgbWrapper(object):
    def __init__(self, params):
        self.param = params

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_round=100000, feval=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)

        if X_val is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (deval, 'val')]

        if feval is None:
            self.clf = xgb.train(self.param, dtrain, num_round, evals=watchlist, verbose_eval=200,
                                 early_stopping_rounds=100, )
        else:
            self.clf = xgb.train(self.param, dtrain, num_round, evals=watchlist, feval=feval, verbose_eval=200,
                                 early_stopping_rounds=100)

    def predict_proba(self, x):
        return self.clf.predict(xgb.DMatrix(x))

    def predict(self, x):
        return self.clf.predict(xgb.DMatrix(x))

    def optimize(self, X_train, y_train, param_grid, eval_func=None, is_proba=False, seed=42):
        feval = lambda y_pred, y_true: ('mae', eval_func(y_true.get_label(), y_pred))

        dtrain = xgb.DMatrix(X_train, label=y_train)

        def fun(**kw):
            params = self.param.copy()
            params['seed'] = seed

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            model = xgb.cv(params, dtrain, 100000, 5, feval=feval, verbose_eval=None, early_stopping_rounds=100)

            return - model['test-mae-mean'].values[-1]

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: {0:0.5f}, params: {1}".format(opt.res['max']['max_val'], opt.res['max']['max_params']))

    def get_name(self):
        return 'XGBoost'


class LRWrapper(object):
    def __init__(self, params):
        self.param = params

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_round=100000, feval=None):
        #### 对数据往往先做一步均一化, z-score
        self.clf = Pipeline([('sc', StandardScaler()),
                             ('clf', LogisticRegression())])
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def get_name(self):
        return 'Logistic Regression'


class RgfWrapper(object):
    def __init__(self, clf, params=None):
        if params:
            self.param = params
            self.clf = clf(**params)
        else:
            self.clf = clf

    def fit(self, X_train, y_train, X_val=None, y_val=None, eval_func=None):
        self.clf.fit(X_train, y_train)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def predict(self, x):
        return self.clf.predict(x)

    def get_name(self):
        return 'RegularizedGreedyForest'
