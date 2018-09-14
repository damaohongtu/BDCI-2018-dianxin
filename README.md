# 面向电信行业存量用户的智能套餐个性化匹配模型
*时间节点：2018年10月22日，00:00:01-2018年10月24日，23:59:59【复赛入围审核】*<br>
@anxiangSir,@irene9adler,@maomao1994
## 1.模型：
-（1）xgboost（Try TM）
-（2）catboost(Try TM)
-（3）tidy-xgb（Try TM）
-（4）SVC+RF+LGBM+XGB（Try TM）
-（5）
-（6）wide and deep（Try TM）
-（7）lgb（OK AX）
-（8）RF(random forest)（Try SQG）
-（9）gbdt+LR (Logistic regression)（Try SQG）
-（10）FTRL（https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295）（https://www.kaggle.com/leeyun/ensemble-model）（Try SQG）
-（11）NFM(Factorisation machine)（OK AX）
-（11）深度FFM（Try AX）
-（13）DeepFM（Try AX）
-（14）FNN（Factorisation machine supported neural network）（Try AX）
-（15）CCPM（Convolutional click prediction model）（Try AX）
-（16）PNN-I（Inner product neural network）（Try AX）
-（17）PNN-II（Outer product neural network）（Try AX）
-（18）PNN-III（Inner&outer product ensembled neural network）（Try AX）
## 2.经验：stacking：一句话解释就是：用其他分类器预测的结果作为当前分类器的特征
-（1）用xgboost来stacking训练lightGBM(Model1)
-（2）用lightGBM来stacking训练xgboost(Model2)
-（3）异常变量的处理
-（4）为什么进行特征选择？单独的特征没有用，组合的特征可能有用？
-（5）分布不对称：取对数试试？

## 3.参考：
-（1）https://blog.csdn.net/xiewenbo/article/details/52038493
-（2）xgboost：http://www.52cs.org/?p=429
-（3）https://github.com/guoday/Tencent2018_Lookalike_Rank7th












