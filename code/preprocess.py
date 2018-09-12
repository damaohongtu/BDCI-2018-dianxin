#!/usr/bin/env pyhton
# coding=utf-8
import pandas as pd

train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')

def total_fee_format(x):
    if x=='\\N':
        return -1.0
    return float(x)
def gender_format(x):
    if x in ['1','2','01','02','0','00']:
        return int(x)
    if x=='\\N':
        return 0
    return x

def age_format(x):
    if x=='\\N':
        return -1
    return int(x)
# train修改缺损值
train_data['2_total_fee']=train_data['2_total_fee'].apply(total_fee_format)
train_data['3_total_fee']=train_data['3_total_fee'].apply(total_fee_format)
train_data['gender']=train_data['gender'].apply(gender_format)
train_data['age']=train_data['age'].apply(age_format)

# train添加缺损值，添加阈值
train_data.loc[train_data['1_total_fee']>800.0,'1_total_fee']=800.0
train_data.loc[train_data['2_total_fee']>800.0,'2_total_fee']=800.0
train_x_mean_2=train_data['2_total_fee'].mean()
train_data.loc[train_data['2_total_fee']<0,'2_total_fee']=train_x_mean_2
train_data.loc[train_data['3_total_fee']>800.0,'3_total_fee']=800.0
train_x_mean_3=train_data['3_total_fee'].mean()
train_data.loc[train_data['3_total_fee']<0,'3_total_fee']=train_x_mean_3
train_data.loc[train_data['4_total_fee']>800.0,'4_total_fee']=800.0
train_data.loc[train_data['month_traffic']>20000.0,'month_traffic']=20000.0
train_data.loc[train_data['pay_num']>500.0,'pay_num']=500.0
train_data.loc[train_data['local_caller_time']>1000.0,'local_caller_time']=1000.0
train_data.loc[train_data['service1_caller_time']>1000.0,'service1_caller_time']=1000.0
train_data.loc[train_data['service2_caller_time']>2500.0,'service2_caller_time']=2500.0

train_data.to_csv('../data/preprocessed_train.csv',index=False)
del  train_data

# test:修改缺损值
test_data['2_total_fee']=test_data['2_total_fee'].apply(total_fee_format)
test_data['3_total_fee']=test_data['3_total_fee'].apply(total_fee_format)
test_data['gender']=test_data['gender'].apply(gender_format)
test_data['age']=test_data['age'].apply(age_format)

# test:添加缺损值，添加阈值
test_data.loc[test_data['1_total_fee']>800.0,'1_total_fee']=800.0
test_data.loc[test_data['2_total_fee']>800.0,'2_total_fee']=800.0
test_x_mean_2=test_data['2_total_fee'].mean()
test_data.loc[test_data['2_total_fee']<0,'2_total_fee']=test_x_mean_2
test_data.loc[test_data['3_total_fee']>800.0,'3_total_fee']=800.0
test_x_mean_3=test_data['3_total_fee'].mean()
test_data.loc[test_data['3_total_fee']<0,'3_total_fee']=test_x_mean_3
test_data.loc[test_data['4_total_fee']>800.0,'4_total_fee']=800.0
test_data.loc[test_data['month_traffic']>20000.0,'month_traffic']=20000.0
test_data.loc[test_data['pay_num']>500.0,'pay_num']=500.0
test_data.loc[test_data['local_caller_time']>1000.0,'local_caller_time']=1000.0
test_data.loc[test_data['service1_caller_time']>1000.0,'service1_caller_time']=1000.0
test_data.loc[test_data['service2_caller_time']>2500.0,'service2_caller_time']=2500.0

test_data.to_csv('../data/preprocessed_test.csv',index=False)
del test_data