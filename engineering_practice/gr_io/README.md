## 项目名称

青光眼预测模型站点构建

## 项目描述

这个项目使用XGBoost模型来预测患者是否患有青光眼。模型接受5个特征作为输入，包括患者的性别、年龄、最佳矫正视力、视盘盘沿径比和眼压，并返回一个布尔值作为预测结果。

## 项目结构

- `main.py`: 包含主要的预测函数和Gradio应用的启动函数。
- `xgb_4.joblib`: 训练好的XGBoost模型。

## 使用方法

1. 安装必要的Python库：

```bash
pip install xgboost joblib gradio
```

2. 运行`main.py`文件以启动Gradio应用：

```bash
python main.py
```

3. 在Gradio应用中输入患者的特征，点击"Submit"按钮获取预测结果。

## 注意事项

- 请确保所有的Python库都已经更新到最新版本。

## 作者

Rvosuke

## 许可证

此项目遵循MIT许可证。