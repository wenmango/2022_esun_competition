## 產生訓練用 data

import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import Counter

dp_data = pd.concat([pd.read_csv('../data/public_train_x_dp_full_hashed.csv'),
                     pd.read_csv('../data/private_x_dp_full_hashed.csv')],axis=0)

ccba_data = pd.concat([pd.read_csv('../data/public_train_x_ccba_full_hashed.csv'),
                       pd.read_csv('../data/private_x_ccba_full_hashed.csv')],axis=0)

remit_data = pd.concat([pd.read_csv('../data/public_train_x_remit1_full_hashed.csv'),
                        pd.read_csv('../data/private_x_remit1_full_hashed.csv')],axis=0)

info_data = pd.concat([pd.read_csv('../data/public_train_x_custinfo_full_hashed.csv'),
                       pd.read_csv('../data/private_x_custinfo_full_hashed.csv')],axis=0)

cdtx_data = pd.concat([pd.read_csv('../data/public_train_x_cdtx0001_full_hashed.csv'),
                       pd.read_csv('../data/private_x_cdtx0001_full_hashed.csv')],axis=0)

date_data = pd.concat([pd.read_csv('../data/train_x_alert_date.csv'),
                       pd.read_csv('../data/public_x_alert_date.csv'),
                       pd.read_csv('../data/private_x_alert_date.csv')],axis=0)

y_data = pd.concat([pd.read_csv('../data/train_y_answer.csv'),
                    pd.read_csv('../data/24_ESun_public_y_answer.csv')],axis=0)

dp_data['gobank'] = 0
dp_data.loc[(dp_data['tx_type']==1) & (dp_data['info_asset_code']==12),'gobank'] = 1


def nunique(array):
    '''get number of unique element'''
    return len(set(array))

def find_std(array):
    '''get std of unique element frequence'''
    ncounter = Counter(array)
    return np.std(list(ncounter.values()))

cdtx_data['mix_country'] = cdtx_data['country'].astype('str') + '_' + cdtx_data['cur_type'].astype('str')

cdtx_data_fix = cdtx_data.groupby(['cust_id']).agg({
    'country':[nunique, find_std],
    'cur_type':[nunique, find_std],
    'mix_country':[nunique, find_std],
    'amt': ['sum','count'],
}).reset_index()
cdtx_data_fix.columns = ['_'.join([str(y) for y in x if y]) for x in cdtx_data_fix.columns]

dp_data['tx_amt_tran'] = dp_data['tx_amt'] * dp_data['exchg_rate']
dp_data['tx_time_cat'] = dp_data['tx_time'] //3

dp_data_fix = dp_data.groupby(['cust_id','tx_type']).agg({
    'debit_credit':[nunique],
    'tx_time_cat':[nunique, find_std],
    'info_asset_code':[find_std],
    'fiscTxId':[find_std],
    'txbranch':[find_std],
    'tx_amt_tran': ['sum','count'],
    'cross_bank':['mean'],
    'ATM':['mean'],
}).unstack().reset_index()
dp_data_fix.columns = ['_'.join([str(y) for y in x if y]) for x in dp_data_fix.columns]

remit_data_fix = remit_data.groupby(['cust_id']).agg({
    'trans_no': [nunique,find_std],
    'trade_amount_usd':['sum','count'],
}).reset_index()
remit_data_fix.columns = ['_'.join([str(y) for y in x if y]) for x in remit_data_fix.columns]

ccba_data['month'] = ccba_data['byymm'] //30
ccba_data['cost_ratio'] = ccba_data['usgam'] / (ccba_data['cycam'] + 0.00001)
ccba_data['ratio_1'] = ccba_data['cucah'] / (ccba_data['lupay'] + 0.00001)
ccba_data['ratio_2'] = ccba_data['cucsm'] / (ccba_data['lupay'] + 0.00001)
ccba_data['ratio_3'] = ccba_data['inamt'] / (ccba_data['lupay'] + 0.00001)
ccba_data['ratio_4'] = ccba_data['csamt'] / (ccba_data['lupay'] + 0.00001)
ccba_data['ratio_5'] = ccba_data['clamt'] / (ccba_data['lupay'] + 0.00001)
ccba_data['ratio_all'] = ccba_data['ratio_1'] + ccba_data['ratio_2'] + ccba_data['ratio_3'] + ccba_data['ratio_4'] + ccba_data['ratio_5'] 

ccba_data_fix = ccba_data.groupby(['cust_id']).agg({
    'cycam'      :['max'],
    'cost_ratio' :['sum'],
    'ratio_1'    :['sum'],
    'ratio_2'    :['sum'],
    'ratio_3'    :['sum'],
    'ratio_4'    :['sum'],
    'ratio_5'    :['sum'],
    'ratio_all'  :['sum'],
}).reset_index()
ccba_data_fix.columns = ['_'.join([str(y) for y in x if y]) for x in ccba_data_fix.columns]

df = pd.merge(info_data,y_data,how='outer')
df = pd.merge(df,date_data,how='outer')
df = pd.merge(df,cdtx_data_fix,how='left',left_on=['cust_id'],right_on=['cust_id'])
df = pd.merge(df,dp_data_fix,how='left',left_on=['cust_id'],right_on=['cust_id'])
df = pd.merge(df,remit_data_fix,how='left',left_on=['cust_id'],right_on=['cust_id'])
df = pd.merge(df,ccba_data_fix,how='left',left_on=['cust_id'],right_on=['cust_id'])
df.to_pickle('./raw_data_1225_submit.pkl')