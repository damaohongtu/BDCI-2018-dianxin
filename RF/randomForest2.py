import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# 基础配置信息
path = './'
n_splits = 5
seed = 42

# lgb 参数
params = {
    "learning_rate": 0.1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "max_depth": 5,
    "objective": "multiclass",
    "num_class": 15,
    "silent": True,
    "verbosity": -1
}

# 读取数据
train = pd.read_csv(path + 'train/train.csv')
test = pd.read_csv(path + 'test/test.csv')

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

'''
简单分析数据：
user_id 为编码后的数据，大小：
train data shape (612652, 27)
train data of user_id shape 612652
简单的1个用户1条样本的题目,标签的范围 current_service
'''
print('标签', set(train.columns) - set(test.columns))

print('train data shape', train.shape)
print('train data of user_id shape', len(set(train['user_id'])))
print('train data of current_service shape', (set(train['current_service'])))

print('train data shape', test.shape)
print('train data of user_id shape', len(set(test['user_id'])))

# 对标签编码 映射关系
label2current_service = dict(
    zip(range(0, len(set(train['current_service']))), sorted(list(set(train['current_service'])))))
current_service2label = dict(
    zip(sorted(list(set(train['current_service']))), range(0, len(set(train['current_service'])))))

# 原始数据的标签映射
train['current_service'] = train['current_service'].map(current_service2label)

# train修改缺损值
train['2_total_fee']=train['2_total_fee'].apply(total_fee_format)
train['3_total_fee']=train['3_total_fee'].apply(total_fee_format)
train['gender']=train['gender'].apply(gender_format)
train['age']=train['age'].apply(age_format)

# train添加缺损值，添加阈值
train.loc[train['1_total_fee']>800.0,'1_total_fee']=800.0
train.loc[train['2_total_fee']>800.0,'2_total_fee']=800.0
train_x_mean_2=train['2_total_fee'].mean()
train.loc[train['2_total_fee']<0,'2_total_fee']=train_x_mean_2
train.loc[train['3_total_fee']>800.0,'3_total_fee']=800.0
train_x_mean_3=train['3_total_fee'].mean()
train.loc[train['3_total_fee']<0,'3_total_fee']=train_x_mean_3
train.loc[train['4_total_fee']>800.0,'4_total_fee']=800.0
train.loc[train['month_traffic']>20000.0,'month_traffic']=20000.0
train.loc[train['pay_num']>500.0,'pay_num']=500.0
train.loc[train['local_caller_time']>1000.0,'local_caller_time']=1000.0
train.loc[train['service1_caller_time']>1000.0,'service1_caller_time']=1000.0
train.loc[train['service2_caller_time']>2500.0,'service2_caller_time']=2500.0

test['2_total_fee']=test['2_total_fee'].apply(total_fee_format)
test['3_total_fee']=test['3_total_fee'].apply(total_fee_format)
test['gender']=test['gender'].apply(gender_format)
test['age']=test['age'].apply(age_format)

# test添加缺损值，添加阈值
test.loc[test['1_total_fee']>800.0,'1_total_fee']=800.0
test.loc[test['2_total_fee']>800.0,'2_total_fee']=800.0
test_x_mean_2=test['2_total_fee'].mean()
test.loc[test['2_total_fee']<0,'2_total_fee']=test_x_mean_2
test.loc[test['3_total_fee']>800.0,'3_total_fee']=800.0
test_x_mean_3=test['3_total_fee'].mean()
test.loc[test['3_total_fee']<0,'3_total_fee']=test_x_mean_3
test.loc[test['4_total_fee']>800.0,'4_total_fee']=800.0
test.loc[test['month_traffic']>20000.0,'month_traffic']=20000.0
test.loc[test['pay_num']>500.0,'pay_num']=500.0
test.loc[test['local_caller_time']>1000.0,'local_caller_time']=1000.0
test.loc[test['service1_caller_time']>1000.0,'service1_caller_time']=1000.0
test.loc[test['service2_caller_time']>2500.0,'service2_caller_time']=2500.0

# 构造原始数据
y = train.pop('current_service')
train_id = train.pop('user_id')
# 这个字段有点问题
X = train
train_col = train.columns

X_test = test[train_col]
test_id = test['user_id']

# 数据有问题数据
for i in train_col:
    X[i] = X[i].replace("\\N", -1)
    X_test[i] = X_test[i].replace("\\N", -1)

X, y, X_test = X.values, y, X_test.values

# 采取k折模型方案
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np


# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True


xx_score = []
cv_pred = []

skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(index)

    X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]

    rfr = RandomForestClassifier(n_estimators=100)
    rfr.fit(X_train, y_train)

    xx_pred = rfr.predict(X_valid)#这里没有选择最好迭代轮次

    #xx_pred = [np.argmax(x) for x in xx_pred]


    xx_score.append(f1_score(y_valid, xx_pred, average='weighted'))

    y_test = rfr.predict(X_test)#这里没有选择最好迭代轮次

    #y_test = [np.argmax(x) for x in y_test]


    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

# 投票
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

# 保存结果
df_test = pd.DataFrame()
df_test['id'] = list(test_id.unique())
df_test['predict'] = submit
df_test['predict'] = df_test['predict'].map(label2current_service)

df_test.to_csv('RF_result3.csv', index=False)

print(xx_score, np.mean(xx_score))