# 面向电信行业存量用户的智能套餐个性化匹配模型
*时间节点：2018年10月22日，00:00:01-2018年10月24日，23:59:59【复赛入围审核】*<br>
@anxiangSir,@irene9adler,@maomao1994
## 1.模型：

-（1）xgboost（Try TM<br>
-（2）catboost(Try TM)<br>
-（3）tidy-xgb（Try TM<br>
-（4）SVC+RF+LGBM+XGB（Try TM）<br>
-（5）<br>
-（6）wide and deep（Try TM）<br>
-（7）lgb（OK AX）<br>
-（8）RF(random forest)（Try SQG）<br>
-（9）gbdt+LR (Logistic regression)（Try SQG）<br>
-（10）FTRL（https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295）（https://www.kaggle.com/leeyun/ensemble-model）（Try SQG）<br>
-（11）NFM(Factorisation machine)（OK AX）<br>
-（11）深度FFM（Try AX）<br>
-（13）DeepFM（Try AX）<br>
-（14）FNN（Factorisation machine supported neural network）（Try AX）<br>
-（15）CCPM（Convolutional click prediction model）（Try AX）<br>
-（16）PNN-I（Inner product neural network）（Try AX）<br>
-（17）PNN-II（Outer product neural network）（Try AX）<br>
-（18）PNN-III（Inner&outer product ensembled neural network）（Try AX）<br>
## 2.经验：stacking：一句话解释就是：用其他分类器预测的结果作为当前分类器的特征

-（1）用xgboost来stacking训练lightGBM(Model1)<br>
-（2）用lightGBM来stacking训练xgboost(Model2)<br>
-（3）异常变量的处理<br>
-（4）为什么进行特征选择？单独的特征没有用，组合的特征可能有用？<br>
-（5）分布不对称：取对数试试？<br>

## 3.参考：

-（1）https://blog.csdn.net/xiewenbo/article/details/52038493 <br>
-（2）xgboost：http://www.52cs.org/?p=429 <br>
-（3）https://github.com/guoday/Tencent2018_Lookalike_Rank7th <br>












