import joblib
import pandas as pd
from typing import List, Union
import gradio as gr


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


def predict(ID: str, Gender: int, Age: int, BCVA: float, CDR: float, IOP: int) -> bool:
    """
    预测函数，接受一个特征列表作为输入，返回预测结果。
    :param ID: 患者ID
    :param Gender: 患者性别
    :param Age: 患者年龄
    :param BCVA: 最佳矫正视力
    :param CDR: 视盘盘沿径比
    :param IOP: 眼压
    :return: 预测结果，为布尔值。
    """
    # 将输入特征转换为适当的格式
    # 例如，将特征列表转换为二维数组
    features = [[Gender, Age, BCVA, CDR, IOP]]

    model = joblib.load('xgb_4.joblib')  # 将'model.joblib'替换为实际的文件名
    # 使用加载的模型进行预测
    prediction = model.predict(features)

    # 返回预测结果
    return prediction[0]


def gradio_launch():
    """
    启动Gradio应用。
    """
    # 启动Gradio应用
    gr.Interface(
        fn=predict,
        inputs=["text", "number", "number", "number", "number", "number"],
        outputs=["number"],
        examples=[
            ["一个十八岁男人", 1, 18, 0.9, 0.5, 18],
            ["一个十九岁女人", 0, 18, 0.7, 0.6, 19],
        ],
        title="预测是否患青光眼",
        description="输入患者的特征，预测是否患青光眼。",
    ).launch(share=True)


def test_predict() -> None:
    """
    测试预测函数的功能。
    """
    # 读取CSV文件
    data = pd.read_csv('TJ_ifglaucoma.csv', encoding='utf-8')

    # 删除第一列
    # data = data.drop(data.columns[0], axis=1)

    # 初始化计数器
    correct_predictions = 0
    total_predictions = 0

    # 遍历每一行数据
    for index, row in data.iterrows():
        # 获取特征和ground truth
        features = row[:-1].tolist()
        ground_truth = bool(row.iloc[-1])

        # 使用预测函数进行预测
        prediction = predict(features)

        # 如果预测结果与ground truth相同，计数器加一
        if prediction == ground_truth:
            correct_predictions += 1

        total_predictions += 1

    # 打印正确预测的数量和总数量
    print(f'Correct predictions: {correct_predictions}/{total_predictions}')


if __name__ == '__main__':
    # test_predict()
    gradio_launch()
