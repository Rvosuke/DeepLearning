import json
import os
import pandas as pd
import numpy as np


class RegionBasedEDA:
    def __init__(self, file_path, status_columns):
        self.df_group = None
        self.df_selected = None
        self.df_agg = None
        self.file_path = file_path
        self.status_columns = status_columns
        self.df = pd.read_csv(self.file_path, low_memory=False)
        self.report = []

    def perform_EDA(self):
        s1 = f"数据集的维度为：{self.df.shape}"
        self.report.append(s1)

        # 特征选择
        selected_columns = ['prov']
        selected_columns.extend(self.status_columns)
        try:
            self.df_selected = self.df[selected_columns].copy()
        except TypeError:
            raise TypeError(f"Please check the status_columns, {self.status_columns} is not a valid input.")
        s2 = f"特征选择后数据集的信息为：\n{self.df_selected.describe()}"
        self.report.append(s2)

        # 数据清洗
        for column in self.status_columns:
            self.df_selected[column] = self.df_selected[column].replace('#NULL!', -8).astype(np.float16)
        self.df_selected = self.df_selected.astype(np.int8)
        s3 = f"每个年份中的存活总人数：\n{self.df_selected[self.status_columns].apply(lambda x: (x == 0).sum())}"
        self.report.append(s3)

        # 数据探索
        self.df_group = self.df_selected.groupby('prov')
        s4 = f"每个省份中的存活总人数：\n{self.df_group.size()}"
        self.report.append(s4)
        self.df_agg = self.df_group.agg(lambda x: (x == 0).sum())
        s5 = f"{self.df_agg}"
        self.report.append(s5)

    def save_report_to_txt(self, txt_path):
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in self.report:
                f.write(line + '\n')

    def save_stats_to_csv(self, csv_path):
        self.df_agg.to_csv(csv_path)


class FeatureListSplitter:
    def __init__(self, var_file_path, status_columns):
        self.var_file_path = var_file_path
        self.survival_status_columns = status_columns
        self.var_df = pd.read_csv(self.var_file_path)
        self.split_feature_dicts = {}

    def split_features(self):
        for col in self.survival_status_columns:
            col_index = self.var_df[self.var_df['id'] == col].index[0]
            features_dict = self.var_df.iloc[:col_index].set_index('id').iloc[:, 0].to_dict()
            self.split_feature_dicts[col] = features_dict

    def save_to_json(self, json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.split_feature_dicts, f, ensure_ascii=False, indent=4)

    def save_to_csv(self, csv_path):
        df_from_dict = pd.DataFrame.from_dict({(i, j): self.split_feature_dicts[i][j]
                                               for i in self.split_feature_dicts.keys()
                                               for j in self.split_feature_dicts[i].keys()},
                                              orient='index')
        df_from_dict.reset_index(inplace=True)
        df_from_dict.columns = ['Composite_Key', 'Description']

        df_from_dict['Year'] = df_from_dict['Composite_Key'].apply(lambda x: x[0])
        df_from_dict['Feature'] = df_from_dict['Composite_Key'].apply(lambda x: x[1])

        df_from_dict.drop(columns=['Composite_Key'], inplace=True)
        df_from_dict = df_from_dict[['Year', 'Feature', 'Description']]
        df_from_dict.to_csv(csv_path, index=False)


class MultiFileReportGenerator:
    def __init__(self, src_files, var_files, save_paths, status_columns_list):
        self.source_files = src_files
        self.var_view_files = var_files
        self.status_columns_list = status_columns_list
        self.save_path = save_paths

    def generate_reports(self):
        for source_file, var_view_file in zip(self.source_files, self.var_view_files):
            status_columns = self.status_columns_list[self.source_files.index(source_file):]
            # Create a sub-folder for each source_file's reports
            folder_name = os.path.splitext(os.path.basename(source_file))[0]
            report_folder = os.path.join(self.save_path, folder_name)
            os.makedirs(report_folder, exist_ok=True)

            # Generate EDA report
            eda_object = RegionBasedEDA(file_path=source_file, status_columns=status_columns)
            eda_object.perform_EDA()
            eda_object.save_report_to_txt(txt_path=os.path.join(report_folder, 'EDA_report.txt'))
            eda_object.save_stats_to_csv(csv_path=os.path.join(report_folder, 'EDA_stats.csv'))

            # Generate Feature List report
            feature_list_splitter = FeatureListSplitter(var_file_path=var_view_file, status_columns=status_columns)
            feature_list_splitter.split_features()
            feature_list_splitter.save_to_json(json_path=os.path.join(report_folder, 'split_feature_dicts.json'))
            feature_list_splitter.save_to_csv(csv_path=os.path.join(report_folder, 'split_feature_dicts.csv'))


source_files = ['data/clhls98.csv', 'data/clhls00.csv', 'data/clhls02.csv', 'data/clhls05.csv',]
var_view_files = ['data/clhls98_var.csv', 'data/clhls00_var.csv', 'data/clhls02_var.csv', 'data/clhls05_var.csv',]
source_list = ['dth98_00', 'dth00_02', 'dth02_05', 'dth05_08', 'dth08_11', 'dth11_14', 'dth14_18']
save_path = './reports'

report_generator = MultiFileReportGenerator(source_files, var_view_files, save_path, source_list)
report_generator.generate_reports()
