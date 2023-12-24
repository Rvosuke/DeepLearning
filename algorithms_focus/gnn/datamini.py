import pandas as pd
import miceforest as mf

from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Load the data
file_path = 'model.xlsx'
test_data_path = 'testsourcedata.xlsx'

data = pd.read_excel(file_path)
test_data = pd.read_excel(test_data_path)

model_columns = data.columns
selected_columns = [col for col in model_columns if col in test_data.columns]
test_data = test_data[selected_columns]

data['日常生活活动能力总分'] = (data['日常生活活动能力总分'] < 60).astype(int)
test_data['日常生活活动能力总分'] = (test_data['日常生活活动能力总分'] < 60).astype(int)


# Delete data with keywords
keywords = ['卧床', '轮椅']
mask = data.apply(lambda row: row.astype(str).str.contains('|'.join(keywords)).any(), axis=1)
data = data[~mask]
mask = test_data.apply(lambda row: row.astype(str).str.contains('|'.join(keywords)).any(), axis=1)
test_data = test_data[~mask]

X = pd.concat([data, test_data])

# Initializing encoders
label_encoder = LabelEncoder()
# Selecting columns for label encoding
label_columns = ['来源城市', '性别', '文化程度', '婚姻状况', '子女情况', '居住地类别', '居住方式', '目前工作情况',
                 '医保类型', '家庭月收入平均为每人每月（元）', '主要经济来源', '所患慢性疾病（可多选）', '长期用药种类',
                 '饮酒情况', '吸烟情况', '是否每年进行体检', '是否参加社交活动', '是否锻炼身体', '社会支持情况',
                 '您从事看书、看电视等日常生活活动时，是否因视力不佳而受到影响？',
                 '房间内有人用正常声音说话，您是否能听清？', '您是否佩戴助听器？', '行走测试是否通过']
# Applying label encoding
for col in label_columns:
    X[col] = X[col].astype(str)
    X[col] = label_encoder.fit_transform(X[col])
for col in ['身高', '体重']:
    X[col] = X[col].astype(float)

# Imputate missing values
kernel = mf.ImputationKernel(X, save_all_iterations=True, random_state=2023)
kernel.mice(5)
X = kernel.complete_data()

data = X.iloc[:len(data), :]
test_data = X.iloc[len(data):, :]

# Balance the data set
rus = RandomUnderSampler(random_state=0)
data, _ = rus.fit_resample(data, data['日常生活活动能力总分'])

# Sort the data
data.sort_index(inplace=True)

# Delete data with missing values
test_data.dropna(inplace=True)

X = pd.concat([data, test_data])
X.reset_index(drop=True, inplace=True)
y = X['日常生活活动能力总分']

X.to_csv('expression.csv', encoding='utf-8-sig')
y.to_csv('target.csv', encoding='utf-8-sig')
