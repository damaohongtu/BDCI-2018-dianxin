#!/usr/bin/env python
#coding=utf-8
import xgboost as xgb
import pandas as pd
from data import load_data
_,test_data=load_data.load_data_with_header()

user_id=test_data.pop(['user_id'])
label_vocabulary={0:'99999825', 1:'90063345', 2:'90109916', 3:'89950166', 4:'89950168',
                  5:'99104722', 6:'89950167', 7:'89016252', 8:'90155946', 9:'99999828',
                  10:'99999826', 11:'99999827', 12:'89016259', 13:'99999830', 14:'89016253'}

# 预测
model=xgb.load_model('./xgb.model')
xgb_test=xgb.DMatrix(test_data)
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)

result=pd.DataFrame(preds,columns=['predict'])
result['predict']=result['predict'].map(label_vocabulary)
xgb_submission=pd.concat([user_id,result],axis=1)
xgb_submission.to_csv('./xgb_submission.csv',index=False)