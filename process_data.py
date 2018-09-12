import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

columns = ['service_type', 'is_mix_service', 'online_time', '1_total_fee',
           '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic',
           'many_over_bill', 'contract_type', 'contract_time',
           'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num',
           'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
           'service1_caller_time', 'service2_caller_time', 'gender', 'age',
           'complaint_level', 'former_complaint_num', 'former_complaint_fee',
           'current_service', 'user_id']

labels = ['current_service']
'gender这个字段有问题'


# '2_total_fee', '3_total_fee', 'age',
discretization_feature = ['online_time', '1_total_fee',  '4_total_fee',
                          'month_traffic','contract_time', 'pay_times', 'pay_num', 'last_month_traffic',
                          'local_trafffic_month','local_caller_time', 'service1_caller_time','service2_caller_time',
                           'former_complaint_num','former_complaint_fee']

def format(x):

    if type(x) == str:
        try:
            f = float(x)
            return f
        except:
            print(x)
            return 1.1
    else:
        return x

def discretization():
    '''离散化特征
    :return:
    '''
    train = pd.read_csv('input/train.csv', encoding='utf-8')
    test = pd.read_csv('input/test.csv', encoding='utf-8')
    test['current_service'] = 0

    df = pd.concat([train, test], axis=0)


    for column in ['2_total_fee', '3_total_fee', 'age']:
        df[column] = df[column].apply(format)
        df[column] = df[column].apply(format)



    for column in discretization_feature:
        print('当前处理的列是:%s' % column)
        print('当前列的dtype:%s' % df[column].dtype)
        bins = []
        for percent in [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70,
                        77, 84, 93]:
            bins.append(np.percentile(df[column], percent))
        train[column] = np.digitize(train[column], bins, right=True)
        test[column] = np.digitize(test[column], bins, right=True)

    # 处理'age'字段
    train['gender'].replace('1', 1, inplace=True)
    train['gender'].replace('2', 2, inplace=True)
    train['gender'].replace('01', 1, inplace=True)
    train['gender'].replace('02', 2, inplace=True)
    train['gender'].replace('0', 0, inplace=True)
    train['gender'].replace('00', 0, inplace=True)
    train['gender'].replace('\\N', 0, inplace=True)
    include = [x for x in train.columns.values]

    for column in include:
        if column in ['2_total_fee', '3_total_fee', 'age','current_service','user_id']:continue

        def add(x):
            return column + '_lalala_' +str(x)

        train[column] = train[column].apply(add)
        test[column] = test[column].apply(add)

    for column in ['2_total_fee', '3_total_fee', 'age']:
        train.drop(column, axis=1, inplace=True)
        test.drop(column, axis=1, inplace=True)

    train = sample(train)

    test.drop('current_service', axis=1, inplace=True)
    train_df, val_df = train_test_split(train, test_size=0.2, shuffle=True)
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8')
    val_df.to_csv('data/dev.csv', index=False, encoding='utf-8')
    test.to_csv('data/test.csv', index=False, encoding='utf-8')
    train.to_csv('data/final_train.csv', index=False, encoding='utf-8')


def sample(train):
    # dict_class = [(89016259, 7257),
    #              (89016253, 8012),
    #              (99999825, 11476),
    #              (99999830, 11820),
    #              (90155946, 12464),
    #              (99999826, 16369),
    #              (99999827, 18143),
    #              (89950168, 18628),
    #              (90109916, 21365),
    #              (99104722, 29078),
    #              (89016252, 29130),
    #              (99999828, 29824),
    #              (89950167, 41045),
    #              (89950166, 74756),
    #              (90063345, 160754)]
    sample_list = [99999826, 99999827, 89950168, 90109916, 99104722, 89016252,
                   99999828, 89950167, 89950166, 90063345,]
    no_sample_list = [89016259,89016253,99999825,99999830, 90155946,]
    sample_df = []
    no_sample_df = []
    for label in sample_list:
        df = train.iloc[train[train.current_service == label].index.tolist()]
        df = df.sample(10000,random_state=2018)
        sample_df.append(df)
    for label in no_sample_list:
        df = train.iloc[train[train.current_service == label].index.tolist()]
        no_sample_df.append(df)
    sample_df.extend(no_sample_df)
    return pd.concat(sample_df, axis=0)
if __name__ == '__main__':
    discretization()