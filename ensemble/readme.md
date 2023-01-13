# 本部分是利用lightgbm做learning to rank 排序，主要包括：
- 数据预处理
- 模型训练
- 模型决策可视化
- 预测
- ndcg评估
- 特征重要度
- SHAP特征贡献度解释
- 样本的叶结点输出

(要求安装lightgbm、graphviz、shap等)

## 一.data format (raw data -> (feats.txt, group.txt))

###### python lgb_ltr.py -process
原始特征为./data/train/finetune_train_w_vaild.txt ./data/valid/valid_f.txt ./data/test/test.txt. 它们被处理成ranklib的输入格式.
通过python lgb_ltr.py -process 将其处理为feats.txt和group.txt


## 二.model train (feats.txt, group.txt) -> train -> model.mod

###### python lgb_ltr.py -train
使用python lgb_ltr.py -train进行训练(需要在参数内修改模型加载路径)
模型保存在.data/model/,目前存放的为我们训练好的checkpoint
我们自行构建了评测函数DCG@10优化训练

train params = {
            'task': 'train',  # 执行的任务类型
            'boosting_type': 'gbrt',  # 基学习器
            'objective': 'lambdarank',  # 排序任务(目标函数)
            'metric': 'ndcg',  # 度量的指标(评估函数)
            'max_position': 10,  # @NDCG 位置优化
            'metric_freq': 1,  # 每隔多少次输出一次度量结果
            'train_metric': True,  # 训练时就输出度量结果
            'ndcg_at': [10],
            'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
            'num_iterations': 200,  # 迭代次数，即生成的树的棵数
            'learning_rate': 0.01,  # 学习率
            'num_leaves': 31,  # 叶子数
            'max_depth':6,
            'tree_learner': 'serial',  # 用于并行学习，‘serial’： 单台机器的tree learner
            'min_data_in_leaf': 30,  # 一个叶子节点上包含的最少样本数量
            'verbose': 2  # 显示训练时的信息
    }


## 三.model predict 
###### python lgb_ltr.py -predict
使用python lgb_ltr.py -predict进行预测
在./data/test/内生成最终结果,目前的output.txt为我们最终提交文件

## 四.validate ndcg 数据来自test.txt(data from test.txt)

###### python lgb_ltr.py -ndcg
计算valid或test的ndcg


## 五.features 打印特征重要度(features importance)

###### python lgb_ltr.py -feature

模型中的特征是"Column_number",这里打印重要度时可以映射到真实的特征名


## 六.利用SHAP值解析模型中特征重要度

###### python lgb_ltr.py -shap
这里不同于六中特征重要度的计算，而是利用博弈论的方法--SHAP（SHapley Additive exPlanations）来解析模型。
利用SHAP可以进行特征总体分析、多维特征交叉分析以及单特征分析等。


## 七.利用模型得到样本叶结点的one-hot表示，可以用于像gbdt+lr这种模型的训练

###### python lgb_ltr.py -leaf


## 八.REFERENCES

https://github.com/jiangnanboy/learning_to_rank
