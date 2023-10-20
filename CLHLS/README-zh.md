# CLHLS 数据分析

## 概览

- **日期**: 2023年10月19日
- **作者**: 白泽阳 (Rvosuke)

本项目专注于一组纵向数据集的分析。该数据集涉及跟踪一组个体，跟踪的时间间隔（采样间隔）是不规则的，一般每两到三年进行一次。数据中可能包含一些失访的个体，这并不一定意味着他们已经去世。对于在新一轮调查时已经去世的个体，会对其近亲进行调查。

## 数据集描述

例如，1998年启动的数据集包括9093名个体。在2000年进行的随访调查中，有4831名是同一组个体，3368名是已经去世的个体的近亲，还有894名失访，总计9093名。

## 分析重点

本项目的主要目标是根据地理区域对数据集进行分类，相应地组织样本量，并进行探索性数据分析（EDA）。

### 重要变量

- **`id`**: 用作每个个体的唯一标识符。即使数据集为不同调查年份的个体提供了数量信息，该变量在进行地域分类时仍然是必需的。
  
- **`prov`**: 表示地理区域。每个编码对应一个不同的位置。在数据组织完成后，将把这些数值编码映射到省份名称。
  
- **`dth98_00`, `dth00_02`, `dth02_05`, `dth05_08`, `dth08_11`, `dth11_14`, `dth14_18`**: 这些字段与特定时期内个体的生存状态有关。

### 数据结构

该数据集包含9093行，代表9093个样本。每一行包括同一样本在不同年份的特征，而不是形成一个新样本。因此，该数据集在2012年之后有严重的缺失值。希望进行时间序列分析的研究人员需要手动提取特征。

## 使用方法

该仓库包含设计用于以下目的的 Python 类：
- 根据生存状态列将数据集分割成更小的部分。
- 按地理区域进行分组进行 EDA。

这两个类都针对大型数据集进行了优化，并可用于生成文本和 CSV 格式的报告。

## 环境要求

- Python 3.10
- Pandas
- NumPy


## 快速开始

```python
from your_module import RegionBasedEDA, FeatureListSplitter

# 初始化并运行 EDA
eda_object = RegionBasedEDA(file_path="your_data_file.csv", status_columns=["your_status_columns"])
eda_object.perform_EDA()
eda_object.save_report_to_txt(txt_path="your_save_path/EDA_report.txt")
eda_object.save_stats_to_csv(csv_path="your_save_path/EDA_stats.csv")

# 初始化并运行特征列表分割器
feature_splitter = FeatureListSplitter(var_file_path="your_var_view_file.csv", status_columns=["your_status_columns"])
feature_splitter.split_features()
feature_splitter.save_to_json(json_path="your_save_path/split_feature_dicts.json")
feature_splitter.save_to_csv(csv_path="your_save_path/split_feature_dicts.csv")
```