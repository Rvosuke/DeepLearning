# %% 模块导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# %% 数据读取
train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
# %% 数据探索
# 略

# %% 特征分组
# 特征分组，以下是通过GPT4分类，并经过了一定的人为修改
group_1 = ['MSSubClass', 'MSZoning', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
           'YearRemodAdd']
group_2 = ['LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2']
group_3 = ['RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond']
group_4 = ['Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
group_5 = ['Heating', 'HeatingQC', 'CentralAir', 'Fireplaces', 'FireplaceQu', 'Electrical']
group_6 = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional']
group_7 = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
           'PavedDrive']
group_8 = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence',
           'MiscFeature', 'MiscVal']
group_9 = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']
y = ['SalePrice']


# %% Group1处理
#
# 通过观察，发现group1中的特征，大部分都是类别型的，所以我们先对其进行标签编码
# 对object类型数据进行标签编码
def encode_categorical_features(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df
