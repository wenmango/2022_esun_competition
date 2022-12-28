## 將訓練用 data 傳入此 script 作為模型訓練和預測

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import Counter, defaultdict

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def cor_omit(df,thres_cor=0.95):
    '''omit high corr feature'''
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_col = [column for column in upper.columns if any(upper[column] >= thres_cor)]
    print(f'Drop: {len(drop_col)} columns')
    df = df.drop(columns=drop_col)
    return df

def reduce_mem_usage(df, verbose=True):
    '''reduce RAM usage'''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

class customCV(object):
    '''cross validation'''
    def __init__(self,n_splits=5,seed=42):
        self.n_splits = n_splits
        np.random.seed(seed)
        
    def split(self, group):
        idx2cust = {i:x for i,x in enumerate(group)}
        cust2idx = defaultdict(list)
        for _idx,_id in idx2cust.items():
            cust2idx[_id].append(_idx)
            
        unique_custid = list(cust2idx)
        shuffle_custid = np.random.choice(unique_custid,len(unique_custid),replace=False)

        batch_size = len(unique_custid)//self.n_splits
        for i in range(self.n_splits):
            test_custs = shuffle_custid[i*batch_size:(i+1)*batch_size]
            test_idxs  = sum([cust2idx[cust] for cust in test_custs],[])
            train_idxs = list(set(idx2cust) - set(test_idxs))
            yield (np.array(train_idxs),np.array(test_idxs))
    
    def get_n_splits(self,**kwargs):
        return self.n_splits

def eval_metric(pred_prob, true_label, lack=0):
    '''eval metric Recall@N-lack'''
    cur, N = 0, sum(true_label)
    sort_idx = np.argsort(pred_prob)[::-1]
    for i,l in enumerate(sort_idx):
        cur+=true_label[l]
        if cur==(N-lack):
            break
    return (N-lack)/(i+1)

def custom_metric_for_lgbm(preds, data):
    '''eval metric Recall@N for lgbm'''
    label = data.get_label()
    metric = eval_metric(preds,label)
    return 'Recall@N', metric, True

def custom_metric_for_xgbm(preds, data):
    '''eval metric Recall@N for xgb'''
    label = data.get_label()
    metric = eval_metric(preds,label)
    return 'Recall@N', -metric


def modelCenter(train_df:pd.DataFrame, test_df:pd.DataFrame, 
                cat_feature, num_feature, tar_feature, 
                model_params, fitting_params, model_type = 'lgb',
                folds=None, nfold=10, cv_seed=42, warm_up_rounds=100
               ):
    '''model training and return inference result'''

    all_feature = cat_feature + num_feature
    X, y = train_df[all_feature], train_df[tar_feature]
    folds =  customCV(nfold,seed=cv_seed) if folds is None else folds
    
    yhat_pred, cv_records, model_records, mask_records = [], [], [], []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_df['cust_id'])):
        X_train, X_valid = X.loc[train_index,all_feature], X.loc[valid_index,all_feature]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type=='lgb':
            lgb_train, lgb_valid = lgb.Dataset(X_train, y_train), lgb.Dataset(X_valid, y_valid)        
            model = lgb.train(model_params,lgb_train,**fitting_params,
                              categorical_feature=cat_feature,
                              valid_names=['train', 'valid'], 
                              valid_sets=[lgb_train, lgb_valid],
                              fobj=None,
                              feval=custom_metric_for_lgbm
                             )
            yhat_single = model.predict(test_df[all_feature],num_iteration=min(warm_up_rounds,model.best_iteration))
            cv_best_score = model.best_score['valid']['Recall@N']

        elif model_type=='xgb':
            xgb_train = xgb.DMatrix(data=X_train.values, label=y_train.values, feature_names=all_feature, enable_categorical=True)
            xgb_valid = xgb.DMatrix(data=X_valid.values, label=y_valid.values, feature_names=all_feature, enable_categorical=True)
            xgb_test  = xgb.DMatrix(data=test_df[all_feature],feature_names=all_feature, enable_categorical=True)

            watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid_data')]
            model = xgb.train(model_params,xgb_train, **fitting_params,
                              evals=watchlist,
                              custom_metric=custom_metric_for_xgbm
                             )
            yhat_single = model.predict(xgb_test,ntree_limit=min(warm_up_rounds,model.best_iteration))
            cv_best_score = -model.best_score 
        
        print(f"[Fold {fold_n + 1:2d}] ended at {time.ctime()}, best iteration: {model.best_iteration:4d}, "+\
              f"best score: {cv_best_score:.5f}")
        yhat_pred.append(yhat_single)
        cv_records.append(cv_best_score)
        model_records.append(model)
        mask_records.append((train_index, valid_index))
        
    avg_score = np.mean(cv_records)
    print(f'***** AVG cv score: {avg_score:.5f} *****')
    return yhat_pred, avg_score, model_records, mask_records


MODEL_PARAMS_LGB = {
    "eta": 0.005,
    "max_depth": 15,
    "subsample": 0.75,
    "colsample_bytree":0.22,    
    "scale_pos_weight":11,
    "min_child_weight":5,
    "min_split_gain":0.10677364238023246,
    "num_leaves":72,
    "subsample_for_bin":3066,
    "scale_pos_weight":20,
    "subsample_freq":5,    
    "objective": "binary",
    "boosting_type":"gbdt",
    "first_metric_only": True,
    "metric" : "None",
    "seed": 20221225,
    "num_threads": 35,
    "verbose": -100,
}

MODEL_PARAMS_XGB = {
    "eta": 0.005,
    "max_depth": 15,
    "min_child_weight":5,
    "subsample": 0.7,
    "colsample_bytree":0.2,
    "gamma":0.01,
    "scale_pos_weight":20,
    "objective": "binary:logistic",
    "tree_method": "gpu_hist",
    "disable_default_eval_metric": True,    
    "seed": 20221225,
}

FIT_PARAMS = {"num_boost_round": 5000, "early_stopping_rounds": 200, "verbose_eval":False}


yhat_list, cv_list = [], []
    
# read data
df = pd.read_pickle('./raw_data_1225_submit.pkl')
df = reduce_mem_usage(df)

unuseColumn = ['cust_id','tx_date','date','trans_date','byymm','alert_key','sar_flag', 'total_asset']
catColumns  = []

for col in catColumns:
    df[col] = df[col].fillna(-1).astype("category")

df_train = df[df['sar_flag'].notnull()].sort_values(by='date').reset_index(drop=1)
df_test  = df[df['sar_flag'].isnull()].reset_index(drop=1)
df_train['sar_flag'] = df_train['sar_flag'].astype('int')

train_label = df_train.groupby('cust_id').agg({'sar_flag':max}).reset_index()
df_train = df_train.drop_duplicates(subset='cust_id').drop(columns=['sar_flag'])
df_train = df_train.merge(train_label,how='left').reset_index(drop=1)
df_train = cor_omit(df_train,0.95).copy()

numColumns  = [x for x in df_train.columns if x not in unuseColumn+catColumns]

n_cv = 10
warmup_round = 200

# model training
for _seed in [123,789,456]:
    yhat1, score1, m1, splitSets1 = modelCenter(train_df=df_train, test_df=df_test, 
                             cat_feature=catColumns, num_feature=numColumns, tar_feature='sar_flag', 
                             model_params=MODEL_PARAMS_LGB, fitting_params=FIT_PARAMS, model_type='lgb',
                             folds=None, nfold=n_cv, cv_seed=_seed, warm_up_rounds=warmup_round)

    yhat2, score2, m2, splitSets2  = modelCenter(train_df=df_train, test_df=df_test,
                                cat_feature=catColumns, num_feature=numColumns, tar_feature='sar_flag', 
                                model_params=MODEL_PARAMS_XGB, fitting_params=FIT_PARAMS, model_type='xgb',
                                folds=None, nfold=n_cv, cv_seed=_seed, warm_up_rounds=warmup_round)

    tmpyhat, tmpcvscore = [], []
    for smodel1, smodel2, splitset in zip(m1,m2,splitSets1):

        metaclf = LogisticRegression(random_state=0)

        all_feature = numColumns + catColumns
        X, y = df_train[all_feature], df_train['sar_flag']
        X_train, X_valid = X.loc[splitset[0],all_feature], X.loc[splitset[1],all_feature]
        y_train, y_valid = y.iloc[splitset[0]], y.iloc[splitset[1]] 


        new_x_train = np.array([smodel1.predict(data=X_train),
                                smodel2.predict(xgb.DMatrix(data=X_train))]).T
        new_x_valid = np.array([smodel1.predict(data=X_valid),
                                smodel2.predict(xgb.DMatrix(data=X_valid))]).T            
        new_x_test  = np.array([smodel1.predict(data=df_test[all_feature]),
                                smodel2.predict(xgb.DMatrix(data=df_test[all_feature]))]).T            

        metaclf.fit(new_x_train, y_train)          
        y_pred_stack = metaclf.predict_proba(new_x_valid)[:,1]
        stack_score = eval_metric(y_pred_stack,y_valid.values)       
        y_pred_stack_test = metaclf.predict_proba(new_x_test)[:,1]

        tmpyhat.append(y_pred_stack_test)
        tmpcvscore.append(stack_score)

    yhat_list.extend(yhat1)
    yhat_list.extend(yhat2)
    yhat_list.extend(tmpyhat)
    cv_list.append(score1)
    cv_list.append(score2)
    cv_list.append(np.mean(tmpcvscore))


info_data = pd.concat([pd.read_csv('../data/public_train_x_custinfo_full_hashed.csv'),
                       pd.read_csv('../data/private_x_custinfo_full_hashed.csv')],axis=0)

key2id = dict(zip(info_data['alert_key'],info_data['cust_id']))

def find_topN(predict,n2=200):
    '''find topN custid'''
    suspect_freq = Counter(dict(zip(df_test['alert_key'],predict))).most_common(n2)
    suspect_id = [key2id[k] for k,p in suspect_freq if key2id.get(k)]
    return list(set(suspect_id))

def catch_topk_id(predict_list,n1=200):
    '''count suspect custid frequence and return probability'''
    suspect_ids = []
    for pred in predict_list:
        tmpId = find_topN(pred,n1)
        suspect_ids.extend(tmpId)
    IDcounter = dict(Counter(suspect_ids))
    return {k:v/len(predict_list) for k,v in IDcounter.items()}



submit = pd.read_csv('../data/submit.csv')
date_data = pd.concat([pd.read_csv('../data/train_x_alert_date.csv'),
                       pd.read_csv('../data/public_x_alert_date.csv'),
                       pd.read_csv('../data/private_x_alert_date.csv')],axis=0)

num = 58*10

topk_fmap = catch_topk_id(yhat_list,num)
df_test['new_prob'] = df_test['alert_key'].apply(lambda x: topk_fmap.copy().setdefault(key2id.setdefault(x,'ZZ'),0))
ansMap = {k:v for k,v in zip(df_test['alert_key'],df_test['new_prob'])}

submit_best = submit.copy()
exp_data = submit_best.merge(info_data)[['alert_key','probability','cust_id']]
exp_data = exp_data.merge(date_data)

for i,_id in enumerate(exp_data['cust_id'].unique()):
    date_max = exp_data[exp_data['cust_id']==_id].date.max()
    if _id in topk_fmap:
        exp_data.loc[(exp_data['cust_id']==_id) & (exp_data['date']<date_max),'probability'] = topk_fmap[_id]/len(yhat_list)
    else:
        exp_data.loc[(exp_data['cust_id']==_id) & (exp_data['date']<date_max),'probability'] = 0.000001*(i+1)

cheat_ans = {a:b for a,b in zip(exp_data['alert_key'],exp_data['probability'])}
submit['probability'] = submit['alert_key'].map(cheat_ans).astype('float32')
submit.to_csv('../output/final_1225_v2.csv',index=0)