# CLHLS Data Analysis

## Overview

- **Date**: 19-10-2023
- **Author**: Zeyang Bai (Rvosuke)

This project focuses on the analysis of a longitudinal dataset, which involves tracking a group of individuals over a series of irregular time intervals—typically every two to three years. The data may include instances where certain individuals are lost to follow-up, which does not necessarily indicate their demise. For those who have passed away by the time of the new survey, their close relatives are interviewed instead.

## Dataset Description

For instance, the dataset initiated in 1998 included 9093 individuals. In the follow-up survey conducted in 2000, 4831 were the same individuals, 3368 were the relatives of deceased individuals, and 894 were lost to follow-up, totaling again to 9093.

## Analysis Focus

The primary focus of this project is to categorize the dataset based on geographic regions, organize the sample size accordingly, and perform Exploratory Data Analysis (EDA).

### Key Variables

- **`id`**: Serves as the unique identifier for each individual. Even though the dataset provides the number of individuals for different years of the survey, this variable is essential for categorizing based on regions.
  
- **`prov`**: Denotes the geographic region. Each code corresponds to a different location. The numerical codes will be mapped to province names after the data is organized.
  
- **`dth98_00`, `dth00_02`, `dth02_05`, `dth05_08`, `dth08_11`, `dth11_14`, `dth14_18`**: These fields are related to the survival status of individuals during specific periods.

### Data Structure

The dataset consists of 9093 rows, which represent 9093 samples. Each row includes features from different years for the same sample rather than forming a new sample. Therefore, the dataset has severe missing values for the years following 2012. Researchers intending to perform time-series analysis will need to manually extract features.

## Usage

This repository contains Python classes designed for:
- Splitting the dataset into smaller parts based on the survival status columns.
- Performing EDA grouped by geographic regions.
  
Both classes are optimized for handling large datasets and can be used to generate reports in both text and CSV formats.

## Requirements

- Python 3.10
- Pandas
- NumPy


## Quick Start

```python
from your_module import RegionBasedEDA, FeatureListSplitter

# Initialize and run EDA
eda_object = RegionBasedEDA(file_path="your_data_file.csv", status_columns=["your_status_columns"])
eda_object.perform_EDA()
eda_object.save_report_to_txt(txt_path="your_save_path/EDA_report.txt")
eda_object.save_stats_to_csv(csv_path="your_save_path/EDA_stats.csv")

# Initialize and run Feature Splitter
feature_splitter = FeatureListSplitter(var_file_path="your_var_view_file.csv", status_columns=["your_status_columns"])
feature_splitter.split_features()
feature_splitter.save_to_json(json_path="your_save_path/split_feature_dicts.json")
feature_splitter.save_to_csv(csv_path="your_save_path/split_feature_dicts.csv")
```
