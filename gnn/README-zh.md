# 因果发现与图神经网络分类

本项目利用CausalNex库的NOTEARS算法进行因果发现，并通过图神经网络执行图级别的分类任务。我们采用了scikit-learn中的breast_cancer数据集以及make_classification函数生成的虚拟数据集进行测试。

## 开始使用

在运行算法之前，需要准备好数据集。运行 `load_datasets.py` 脚本将自动准备并处理数据集（默认为breast_cancer），该脚本将执行以下任务：

1. 因果发现 - 使用NOTEARS算法识别数据集中的因果关系。
2. 数据处理 - 生成图神经网络所需的CSV文件，包括：
   - `expression.csv`：样本特征矩阵
   - `target.csv`：标签向量
   - `adjacency_matrix.csv`：邻接矩阵

完成数据准备后，直接运行 `main.py` 即可开始分类任务，并获取分类结果。
在`main.py`中，您可以通过调整不同的参数来影响算法的表现，例如`no_pos`控制是否使用位置编码，`gcn_base_layers`设置图卷积网络的基础层数量。


## 依赖安装

项目的依赖库列在 `requirements.txt` 文件中。安装依赖库前，请确保您的Python环境已经准备就绪。安装依赖库可以使用以下命令：

```bash
pip install -r requirements.txt
```

## 测试数据集

本项目默认使用scikit-learn中的breast_cancer数据集。您也可以通过scikit-learn的make_classification函数构造自定义的虚拟数据集进行测试。
