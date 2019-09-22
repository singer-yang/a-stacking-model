'''
@Description: a kind of stacking model, refer to: https://www.kaggle.com/getting-started/18153#post103381
@Author: xinge yang
@Date: 2019-09-18 09:45:43
@LastEditTime: 2019-09-21 21:54:18
@LastEditors: xinge yang
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# load raw data
data = pd.read_excel('winequality-red.xlsx')
train, test = train_test_split(data,
                               test_size=0.2,
                               random_state=np.random.randint(100))

# split train and test data
trainy = train['quality']
trainx = train.drop('quality', axis=1)
testy = test['quality']
testx = test.drop('quality', axis=1)

# show relationship of each catogory with label
columnname = trainx.columns
for i, category in enumerate(columnname):
    plt.subplot(int(columnname.shape[0] / 2) + 1, 2, i + 1)
    plt.scatter(trainx[category], trainy, s=80, marker='*')
    plt.xlabel = (category)
    plt.ylabel = ('quality')
plt.show()

# show corr of all categories
sns.heatmap(train.corr())
plt.show()

# regress and predict
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# define score function
def rmsle_cv(model):
    rmse = np.sqrt(-cross_val_score(model,
                                    trainx.values,
                                    trainy.values,
                                    scoring="neg_mean_squared_error",
                                    cv=5))
    return (rmse)


# define models
ridge = Ridge(alpha=1.0)
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(),
                     ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000,
                                   learning_rate=0.05,
                                   max_depth=4,
                                   max_features='sqrt',
                                   min_samples_leaf=15,
                                   min_samples_split=10,
                                   loss='huber',
                                   random_state=5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,
                             gamma=0.0468,
                             learning_rate=0.05,
                             max_depth=3,
                             min_child_weight=1.7817,
                             n_estimators=2200,
                             reg_alpha=0.4640,
                             reg_lambda=0.8571,
                             subsample=0.5213,
                             silent=1,
                             random_state=7,
                             nthread=-1)


# get scores of single model
score_ridge = rmsle_cv(ridge)
print('score of ridge model is {:.4f}({:.4f})'.format(score_ridge.mean(),
                                                    score_ridge.std()))

score_lasso = rmsle_cv(lasso)
print('score of lasso model is {:.4f}({:.4f})'.format(score_lasso.mean(),
                                                    score_lasso.std()))

score_ENet = rmsle_cv(ENet)
print('score of ENet model is {:.4f}({:.4f})'.format(score_ENet.mean(),
                                                    score_ENet.std()))

score_KRR = rmsle_cv(KRR)
print('score of KRR model is {:.4f}({:.4f})'.format(score_KRR.mean(),
                                                    score_KRR.std()))

score_GB = rmsle_cv(GBoost)
print('score of GB model is {:.4f}({:.4f})'.format(score_GB.mean(),
                                                   score_GB.std()))

score_XGB = rmsle_cv(model_xgb)
print('score of XGB model is {:.4f}({:.4f})'.format(score_XGB.mean(),
                                                    score_XGB.std()))


# define a averaging model
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack(
            [model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


averaged_models = AveragingModels(models=(lasso, ENet, KRR))
score = rmsle_cv(averaged_models)
print("score of averaging model is {:.4f} ({:.4f})".format(
    score.mean(), score.std()))


# define a new ensembled model
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        # self.base_models_ = [clone(x) for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X)
                             for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost,
                                                              KRR),
                                                 meta_model=lasso)
score = rmsle_cv(stacked_averaged_models)
print("score of stacking averaged models is {:.4f} ({:.4f})".format(
    score.mean(), score.std()))

print("if stacking score is smaller than averaging score, this method makes sense for a better regression")
