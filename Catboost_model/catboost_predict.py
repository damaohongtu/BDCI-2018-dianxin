#!/usr/bin/env python
#coding=utf-8
import pandas as pd
from data import load_data
import catboost as cgb

_,test_data=load_data.load_data_with_header()
cgb_submisstion = pd.DataFrame()
cgb_submisstion['user_id'] = test_data['user_id']
cgb_test=test_data.drop(['user_id'])
model=cgb.load_model('./cgb.model')
cgb_submisstion['result'] = model.predict(cgb_test)
cgb_submisstion.to_csv('./cgb_submission.csv',index=False)