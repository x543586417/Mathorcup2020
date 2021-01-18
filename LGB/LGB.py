import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import lightgbm as lgb
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
import gc
import time


#############
#############
#train = pd.read_csv(r"/Users/lucifer/Documents/Competition/Mathorcup2020-RuiXing/training_data.csv",encoding="gbk")
train = pd.read_csv(r"/Users/lucifer/Documents/Competition/Mathorcup2020-RuiXing/LGB/训练用数据.csv",encoding="gbk")
#短期测试集
test1 = pd.read_csv(r"/Users/lucifer/Documents/Competition/Mathorcup2020-RuiXing/LGB/test1.csv",encoding="gbk")
#长期测试集
test2 = pd.read_csv(r"/Users/lucifer/Documents/Competition/Mathorcup2020-RuiXing/LGB/test2.csv",encoding="gbk")
train.info()

new_col = ['DATE', 'HOUR','NAME' , 'LABEL1','LABEL2']
train.columns = new_col
test1.columns = new_col
new_col2 = ['DATE','NAME' , 'LABEL1','LABEL2']
test2.columns = new_col2


print(train.head(10))
test1.head()
test2.head()
print("--------------------------------")

#删除训练数据中的重复列，保留一个：
train = train.drop_duplicates(keep='first')
train['HOUR'] = train['HOUR'].apply(lambda x: int(x.split(':')[0]))
train['DAY'] = train['DATE'].apply(lambda x: int(x.split('/')[-1][-2:]))
train['MON'] = train['DATE'].apply(lambda x: int(x[5]))

test1['HOUR'] = test1['HOUR'].apply(lambda x: int(x.split(':')[0]))
test1['DAY'] = test1['DATE'].apply(lambda x: int(x.split('/')[-1]))
test1['MON'] = test1['DATE'].apply(lambda x: int(x[5]))

print(train.head(20))

print("--------------------------------")



#上行
used_feat = ['HOUR','DAY','NAME','MON']
train_x = train[used_feat]
train_y = train['LABEL1']
test_x = test1[used_feat]
print(train_x.shape, test_x.shape)
print("--------------------------------")
# -----------------------------------------------
scores = []

params = {'learning_rate': 0.1,
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'rmse',
          'min_child_samples': 46,
          'min_child_weight': 0.01,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
          'bagging_freq': 2,
          'num_leaves': 16,
          'max_depth': 5,
          'n_jobs': -1,
          'seed': 2019,
          'verbosity': -1,
          }

oof_train = np.zeros(len(train_x))
preds = np.zeros(len(test_x))
folds = 5
seeds = [2048, 1997]
for seed in seeds:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        print('fold ', fold + 1)
        x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[
            val_idx]
        train_set = lgb.Dataset(x_trn, y_trn)
        val_set = lgb.Dataset(x_val, y_val)

        model = lgb.train(params, train_set, num_boost_round=5000,
                          valid_sets=(train_set, val_set), early_stopping_rounds=25,
                          verbose_eval=50)
        oof_train[val_idx] += model.predict(x_val) / len(seeds)
        preds += model.predict(test_x) / folds / len(seeds)
        del x_trn, y_trn, x_val, y_val, model, train_set, val_set
        gc.collect()

    mse = (mean_squared_error(oof_train, train['LABEL1']))

    print('-' * 120)
    print('rmse ', round(mse, 5))

test1['LABEL1'] = preds

#下行
train_x = train[used_feat]
train_y = train['LABEL2']
test_x = test1[used_feat]
print(train_x.shape, test_x.shape)

print("--------------------------------")
# -----------------------------------------------
scores = []

params = {'learning_rate': 0.1,
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'rmse',
          'min_child_samples': 46,
          'min_child_weight': 0.01,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
          'bagging_freq': 2,
          'num_leaves': 16,
          'max_depth': 5,
          'n_jobs': -1,
          'seed': 2019,
          'verbosity': -1,
          }

oof_train = np.zeros(len(train_x))
preds = np.zeros(len(test_x))
folds = 5
seeds = [2048, 1997]
for seed in seeds:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        print('fold ', fold + 1)
        x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[
            val_idx]
        train_set = lgb.Dataset(x_trn, y_trn)
        val_set = lgb.Dataset(x_val, y_val)

        model = lgb.train(params, train_set, num_boost_round=5000,
                          valid_sets=(train_set, val_set), early_stopping_rounds=25,
                          verbose_eval=50)
        oof_train[val_idx] += model.predict(x_val) / len(seeds)
        preds += model.predict(test_x) / folds / len(seeds)
        del x_trn, y_trn, x_val, y_val, model, train_set, val_set
        gc.collect()

    mse = (mean_squared_error(oof_train, train['LABEL2']))

    print('-' * 120)
    print('rmse ', round(mse, 5))

test1['LABEL2'] = preds



test11 = pd.read_csv(r"/Users/lucifer/Documents/Competition/Mathorcup2020-RuiXing/LGB/test1.csv",encoding="gbk")#短期测试集
test11['上行业务量GB'] = test1['LABEL1']
test11['下行业务量GB'] = test1['LABEL2']
test11.to_csv('短期验证选择的小区数据集.csv', index = False)
print(test11.head(20))

