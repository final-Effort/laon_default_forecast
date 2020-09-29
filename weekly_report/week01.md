# 数据挖掘——金融风控之贷款违约预测week01

## 课题理解

本次课题是通过要求分析贷款人填写问卷获取其部分信息数据，给其贷款能力进行评分的以判断其是否会出现贷款违约的行为

## 主要模块

本次课题可分为以下几个模块  
![amatar](https://github.com/final-Effort/laon_default_forecast/blob/master/weekly_report/src/module.jpg)

### 模型训练

为了衡量分类器好坏，我们可以使用**AUC性能指标**
> AUC(Area Under Curve)被定义为ROC曲线下的面积。由于ROC曲线一般处于y=x上方，所以AUC取值范围（0.5，1），AUC越接近1，检测方法真实性越高，分类器越好。如果A的ROC曲线完全在B的ROC曲线上方，则说明A的模型好，但是两个模型的ROC曲线通常是相交的，为了比较性能就需要用到AUC

> ROC(Receiver Operating Characteristic)。纵轴为真实label为+的样本里，预测为+的样本比例，横轴为真实label为-的样本里，预测为-的样本比例，绘制的曲线，越靠近左上角，表面模型的性能越好。

> 可以使用skleam库中的roc_curver()和auc()或者自带的函数skleam.metrics.roc_auc_score计算AUC

### 模型验证

可以使用**K折交叉验证**

> K折交叉验证：将训练集分为K折，用K-1折作为训练集，用剩下一折作为验证集，将K-1折训练出的结果来预测测试集，运算K次

![amatar](https://github.com/final-Effort/laon_default_forecast/blob/master/weekly_report/src/K_fold.jpg)

### 数据分析

#### python学习
> 初识python以为虎，逐渐了解后python天下第一！

了解python的数据科学库（numpy,pandas,sklearn,xgboost,keras,etc)
> numpy高效处理数据，提供数组支持，后面的库都依赖于numpy库，必备！
> pandas 用于进行数据的采集和分析

```python
	# 导入csv文件
	train = pandas,read_csv('D:\贷款违约预测\train.csv')   
	# 查看训练集的行列数
	train.shape
	# 查看训练集
	print(train)
```

> sklearn 机器学习模块
> keras 深度学习模块

#### 数据探索性分析(EDA)  
![amatar](https://github.com/final-Effort/laon_default_forecast/blob/master/weekly_report/src/EDA.jpg)  

| Field        | Discription  |  data__type  | sample | note
| --------     | :-----: | :----:  | :----: | :----: |
| id           | 为贷款清单分配的唯一信用证标识      |   int     |  | 数据集主键，无缺省 |
| loanAmnt     | 贷款金额                          |   float   |  | 数据可能较大 |
| term         | 贷款期限（year）                  |   int  |  |  |
| interestRate | 贷款利率  | float |   |  |
| installment  | 分期付款金额 | float |  |  |
| grade        | 贷款等级 | string | A,G  |  可以使用int类型代替，与下行存在信息冗余  |
| subGrade     | 贷款等级之子级 | string | A2,G1 |  涵盖了上一行的信息，可将贷款等级删除  |
| employmentTitle | 就业职称 | float |  | 半识别已脱敏 |
| employmentLength | 就业年限（年）| string | 10+ year, <1 year | 需要对数据进行清洗 |
| homeOwnership | 借款人在登记时提供的房屋所有权状况 |int |   |  |
| aunuallncome  | 年收入 | float |   |   |
| verificationStatus | 验证状态 | int | 0,1,2 | 这是什么，无贷款，已发放，已还款？ |
| issueDate     | 贷款发放的月份 | string | 2017/9/1,###### | 年/月/日,###未发放？ |
| purpose       | 借款人在贷款申请时的贷款用途类别 | int |  | 半识别，已脱敏 |
| postCode      | 借款人在贷款申请中提供的邮政编码的前3位数字 | int | 10,237 | |
| regionCode    | 地区编码 | int | 21,0,8 | 与上一行信息可能存在关联性 |
| dti           | 债务收入比 | float |  | 某期间内平均债务总额与营业收入之比 |
| deliquency_2years  | 借款人过去2年信用档案中逾期30天以上的违约事件数 | int | 0,1,2 | 与违约预测关系较大 |
| ficoRangeLow  | 借款人在贷款发放时的fico所属的下限范围 | int |  | fico：信用分等级（信用、品德、支付能力） |
| ficoRangeHigh | 借款人在贷款发放时的fico所属的上限范围 | int |  | 与上一行信息存在关联性 |
| openAcc       | 借款人信用档案中未结信用额度的数量 | int |  |  |
| pubRec        | 贬损公共记录的数量 | int |  | 信用品德 |
| pubRecBankruptcis  | 公开记录清除的数量 | int |  | 与上一行信息存在关联性 |
| revoBal       | 信贷周转余额合计 | float |  | |
| revoUtil      | 循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额 | float |  | 单位可能是万元？ |
| totalAcc      | 借款人信用档案中当前的信用额度总数 | int |   | 单位可能是万元？其信用等级可能影响该项 |
| initialListStatus | 贷款的初始列表状态 | int | 0,1 | 可能是bool类型 |
| applicationType  | 表明贷款是个人申请还是与两个共同借款人的联合申请 | int | 0,1 | 可能是bool类型，0表示个人申请 |
| earliesCreditLine | 借款人最早报告的信用额度开立的月份 | string | Nov-74,Jul-01 | 月-日，感觉这一项可能与贷款违约无多大关系 |
| title         | 借款人提供的贷款名称 | float |  | 半识别，已脱敏，大部分数据为0，是某类贷款吗？|
| policyCode    | 公开可用的策略_代码=1新产品不公开可用的策略_代码=2 |int | 1,2 |  |
| n系列匿名特征  | 匿名特征n0-n14，为一些贷款人行为计数特征的处理 |  |  |

### 数据清洗  
 - 去除/补全有缺失的数据
 - 去除/修改格式和内容错误的数据
 - 去除/修改逻辑错误的数据
 - 去除不需要的数据
 - 关联性验证
 
### 特征估测  
 - 数据预处理
  * 离群点处理
  * 错误值处理
  * 假标签处理
 - 特征提取
  * 类别特征
  * 数值特征
 - 特征选择
  * 过滤法
  * 封装法
  * 嵌入法
  
### 模型训练
被推荐了使用lightGBM模型
> lightGBM

### 模型融合

### 心得
作为数据小白，初次看到这个课题还是挺棘手的，毕竟第一次实操。刚开始还是无从下手的阶段，从给的学习资料中习得课题的初步框架，稍有头绪。了解到python对数据的处理更加简单方便，自学了python的一些基础语法及推荐的数据科学库，并分析了训练集中的数据。
