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

def former_complaint_num_check(x):
    if x==0:
        return 0
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

#特征组合
train_data['all_total_fee'] = train_data['1_total_fee']+train_data['2_total_fee']+train_data['3_total_fee']+train_data['4_total_fee']
train_data['service_type_mix']=train_data['service_type']*10 + train_data['is_mix_service']
train_data['contract_type_time'] = train_data['contract_type']*100 + train_data['contract_time']
train_data['pay_all'] = train_data['pay_times']*train_data['pay_num']
train_data['call_tima_all']=train_data['local_caller_time']+train_data['service1_caller_time']+train_data['service2_caller_time']
train_data['gender_age'] = train_data['gender']*100 + train_data['age']
#print('train_data[former_complaint_num]:',train_data['former_complaint_num'])
#print("train_data[former_complaint_fee]:",train_data['former_complaint_fee'])
train_data['former_complaint_fee_mean'] = train_data['former_complaint_fee']/train_data['former_complaint_num']
train_data['complaint_level_num'] = train_data['complaint_level']*train_data['former_complaint_num']

train_data.to_csv('../data2/preprocessed_train.csv',index=False,columns=['service_type','is_mix_service','online_time','1_total_fee','2_total_fee','3_total_fee','4_total_fee','month_traffic','many_over_bill','contract_type','contract_time','is_promise_low_consume','net_service','pay_times','pay_num','last_month_traffic','local_trafffic_month','local_caller_time','service1_caller_time','service2_caller_time','gender','age','complaint_level','former_complaint_num','former_complaint_fee','current_service','user_id','all_total_fee','service_type_mix','contract_type_time','pay_all','call_tima_all','gender_age','former_complaint_fee_mean','complaint_level_num'])

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

#特征组合
test_data['all_total_fee'] = test_data['1_total_fee']+test_data['2_total_fee']+test_data['3_total_fee']+test_data['4_total_fee']
test_data['service_type_mix']=test_data['service_type']*10 + test_data['is_mix_service']
test_data['contract_type_time'] = test_data['contract_type']*100 + test_data['contract_time']
test_data['pay_all'] = test_data['pay_times']*test_data['pay_num']
test_data['call_tima_all']=test_data['local_caller_time']+test_data['service1_caller_time']+test_data['service2_caller_time']
test_data['gender_age'] = test_data['gender']*100 + test_data['age']
#print('test_data[former_complaint_num]:',test_data['former_complaint_num'])

test_data['formar_complaint_fee_mean'] = test_data['former_complaint_fee']/test_data['former_complaint_num']
test_data['complaint_level_num'] = test_data['complaint_level']*test_data['former_complaint_num']

test_data.to_csv('../data2/preprocessed_test.csv',index=False,columns=['service_type','is_mix_service','online_time','1_total_fee','2_total_fee','3_total_fee','4_total_fee','month_traffic','many_over_bill','contract_type','contract_time','is_promise_low_consume','net_service','pay_times','pay_num','last_month_traffic','local_trafffic_month','local_caller_time','service1_caller_time','service2_caller_time','gender','age','complaint_level','former_complaint_num','former_complaint_fee','current_service','user_id','all_total_fee','service_type_mix','contract_type_time','pay_all','call_tima_all','gender_age','former_complaint_fee_mean','complaint_level_num'])
del test_data
