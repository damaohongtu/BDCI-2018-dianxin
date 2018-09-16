#!/usr/bin/env python
#coding=utf-8
import sys
sys.path.append("/home/tm/workplace/BDCI")
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import plot_importance
from matplotlib import pyplot
import numpy as np
import time

train_data=pd.read_csv('../data/preprocessed_train.csv')
label2current_service = dict(zip(range(0,len(set(train_data['current_service']))),sorted(list(set(train_data['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train_data['current_service']))),range(0,len(set(train_data['current_service'])))))
train_data=pd.read_csv('../data/preprocessed_train.csv')
train_data['current_service']=train_data['current_service'].map(current_service2label)

Y=train_data['current_service']
X=train_data.drop(['current_service','user_id'],axis=1)

seed = 7
test_size = 0.33
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=seed)
xgb_train=xgb.DMatrix(X_train,label=y_train)
xgb_val=xgb.DMatrix(X_val,y_val)

params={
    'booster':'gbtree',
    'objective': 'multi:softmax', #多分类的问题
    'num_class':15, # 类别数，与 multisoftmax 并用
    'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':12, # 构建树的深度，越大越容易过拟合
    'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':0.7, # 随机采样训练样本
    'colsample_bytree':0.7, # 生成树时进行的列采样
    'min_child_weight':3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007, # 如同学习率
    'seed':1000,
    'nthread':7,# cpu 线程数
    #'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 5000 # 迭代次数
watchlist = [(xgb_val, 'val')]

#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
start_time=time.time()
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=50)
test_data=pd.read_csv('../data/preprocessed_test.csv')
user_id=test_data.pop('user_id')
xgb_test=xgb.DMatrix(test_data)
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)

result=pd.DataFrame(preds,columns=['predict'])
result['predict']=result['predict'].map(label2current_service)
xgb_submission=pd.concat([user_id,result],axis=1)
xgb_submission.to_csv('./xgb_submission.csv',index=False)

model.save_model('./xgb.model') # 用于存储训练出的模型
print ("best best_ntree_limit",model.best_ntree_limit)

#输出运行时长
cost_time = time.time()-start_time
print ("xgboost success!",'\n',"cost time:",cost_time,"(s)......")






