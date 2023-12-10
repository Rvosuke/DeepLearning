import json
from typing import Dict, List, Tuple

import pandas as pd


class FeatureGrouping:
    """
    特征分组类：用于读取CSV文件，去除重复行，并按照特定规则进行分组。
    """

    def __init__(self, csv_path: str):
        """
        初始化方法：读取CSV文件到DataFrame。

        :param csv_path: CSV文件的路径。
        """
        self.df = pd.read_csv(csv_path)

    def remove_duplicates(self) -> None:
        """
        去除重复行：在DataFrame中去除重复的行。
        """
        self.df = self.df.drop_duplicates()

    def group_features(self) -> Dict[str, Dict[str, str]]:
        """
        分组特征：按照特定规则进行特征分组。

        :return: 分组后的特征字典。
        """
        id_group = {}
        grouped_dict = {}

        for idx, row in self.df.iterrows():
            feature_name = row['ID']
            description = row['Individual ID']

            # 检查特征名称的后两个字母是否为 'ID'
            if feature_name[-2:].lower() == 'id':
                id_group[feature_name] = description
            else:
                # 按照前两个字母进行分组
                key_prefix = feature_name[:2]
                if key_prefix not in grouped_dict:
                    grouped_dict[key_prefix] = {}
                grouped_dict[key_prefix][feature_name] = description

        # 将ID组添加到最终的字典中
        final_grouped_dict = {'ID_group': id_group, **grouped_dict}
        return final_grouped_dict

    @staticmethod
    def save2json(json_path: str, grouped_dict: Dict[str, Dict[str, str]]) -> None:
        """
        保存为JSON文件：将分组后的特征字典保存为JSON文件。

        :param json_path: 要保存的JSON文件的路径。
        :param grouped_dict: 分组后的特征字典。
        """
        grouped_json = json.dumps(grouped_dict, indent=4, ensure_ascii=False)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(grouped_json)


class RegionalSampleReport:
    """
    地域样本报告类：用于读取花名册，检查ID的重复项，并生成地域样本数量的报告。
    """

    def __init__(self, csv_path: str):
        """
        初始化方法：读取CSV文件到DataFrame。

        :param csv_path: CSV文件的路径。
        """
        self.df = pd.read_csv(csv_path, low_memory=False)

    def check_duplicate_ids(self) -> List[int]:
        """
        检查ID列是否有重复的值。

        :return: 重复ID的列表。
        """
        duplicate_ids = self.df[self.df.duplicated(['ID'], keep=False)]['ID'].unique().tolist()
        return duplicate_ids

    def generate_province_column_and_group(self) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        根据communityID生成新的省份列，并按省份进行分组，计算每个组的样本数量。
        省份编码将按照数值大小进行排序。

        :return: 包含新省份列的数据框和各省份的样本数量字典。
        """
        province = self.df['communityID'].apply(lambda x: str(x)[:-5])
        province_count = province.value_counts().to_dict()

        # 按照数值大小对省份编码进行排序
        sorted_province_count = {k: province_count[k] for k in sorted(province_count.keys(), key=int)}

        return province, sorted_province_count

    def save2txt(self, report_path: str, province_count: Dict[str, int]) -> None:
        """
        保存地域样本数量报告为TXT文件。

        :param report_path: 要保存的TXT文件的路径。
        :param province_count: 各省份的样本数量字典。
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            for province, count in province_count.items():
                f.write(f"Province Code: {province}, Sample Count: {count}\n")
            f.write(f"Total Sample Count: {self.df.shape[0]}\n\n")


class DataAnalysisPipeline:
    """
    数据分析流水线：用于简化特征分组和地域样本报告的生成。
    """

    def __init__(self, charls_csv: str, roster_csv: str, json_report_path: str, txt_report_path: str):
        """
        初始化方法：设置文件路径。

        :param charls_csv: CHARLS CSV文件的路径。
        :param roster_csv: 花名册CSV文件的路径。
        :param json_report_path: JSON报告文件的路径。
        :param txt_report_path: TXT报告文件的路径。
        """
        self.charls_csv = charls_csv
        self.roster_csv = roster_csv
        self.json_report_path = json_report_path
        self.txt_report_path = txt_report_path

    def run(self) -> Tuple[str, List[int]]:
        """
        运行流水线：执行特征分组和生成地域样本报告的所有步骤。

        :return: TXT报告文件的路径和重复ID列表（如果有）。
        """
        # 使用 FeatureGrouping 类
        feature_grouping = FeatureGrouping(self.charls_csv)

        # 去除重复行
        feature_grouping.remove_duplicates()

        # 分组特征
        grouped_dict = feature_grouping.group_features()

        # 保存为JSON文件
        feature_grouping.save2json(self.json_report_path, grouped_dict)

        # 使用 RegionalSampleReport 类
        regional_sample_report = RegionalSampleReport(self.roster_csv)

        # 检查ID的重复项
        duplicate_ids = regional_sample_report.check_duplicate_ids()

        # 生成省份列和各省份的样本数量字典
        _, province_count = regional_sample_report.generate_province_column_and_group()

        # 保存报告为TXT文件
        regional_sample_report.save2txt(self.txt_report_path, province_count)

        return self.txt_report_path, duplicate_ids


if __name__ == '__main__':
    pipeline = DataAnalysisPipeline('data/charls11.csv', 'data/charls18.csv',
                                    'report/11.json',
                                    'report/18.txt')
    txt_report_path, duplicate_ids_example = pipeline.run()

    print(duplicate_ids_example)
