import joblib
import pandas as pd
from typing import List, Union
import gradio as gr
import numpy as np


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


def predict(
        Name_Optional: str = "Optional",
        gender: str = None,
        age: str = None,
        BCVA: str = None,
        CDR: str = None,
        IOP: str = None,
        fundus_image: np.ndarray = None,
        OCT_image: np.ndarray = None,
        VF_image: np.ndarray = None
) -> Union[str]:
    """
    预测函数，接受一个特征列表作为输入，返回预测结果。
    :param Name_Optional: 患者ID
    :param gender: 患者性别
    :param age: 患者年龄
    :param BCVA: 最佳矫正视力
    :param CDR: 视盘盘沿径比
    :param IOP: 眼压
    :return: 预测结果，为布尔值。
    """
    # 将输入特征转换为适当的格式
    # 例如，将特征列表转换为二维数组
    if gender == "Male":
        gender = 1
    else:
        gender = 0

    # 创建特征字典
    features = {
        "gender": gender,
        "age": age,
        "BCVA": BCVA,
        "CDR": CDR,
        "IOP": IOP
    }

    # features = [[gender, age, BCVA, CDR, IOP]]
    inputs = []
    # 检测是否有缺失值，并指出缺失值所在的特征变量名称
    for key, value in features.items():
        if value == '':
            return f"Error: {key} is missing."
        else:
            value = float(value)
            inputs.append(value)
    # inputs = [list(features.values())]
    print(inputs)

    model = joblib.load('xgb_4.joblib')  # 将'model.joblib'替换为实际的文件名
    # 使用加载的模型进行预测
    prediction = model.predict([inputs])

    # 返回预测结果
    return prediction[0]


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
    with gr.Blocks() as demo:
        with gr.Row():
            text1 = gr.Textbox(label="t1")
            slider2 = gr.Textbox(label="s2")
            drop3 = gr.Dropdown(["a", "b", "c"], label="d3")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                text1 = gr.Textbox(label="prompt 1")
                text2 = gr.Textbox(label="prompt 2")
                inbtw = gr.Button("Between")
                text4 = gr.Textbox(label="prompt 1")
                text5 = gr.Textbox(label="prompt 2")
            with gr.Column(scale=2, min_width=600):
                img1 = gr.Image("images/cheetah.jpg")
                btn = gr.Button("Go")
    # test_predict()
    gr.Interface(
        fn=predict,
        inputs=[
            "text",
            gr.Radio(["Male", "Female"]),
            "text",
            "text",
            "text",
            "text",
            gr.Image(),
            gr.Image(),
            gr.Image()
        ],
        outputs=["text"],
        examples=[
            ["David", "Male", 18, 0.9, 0.5, 18],
            ["Mary", "Female", 18, 0.7, 0.6, 19],
        ],
        title="Glaucoma Prediction App",
        description="Input the patient's features and predict if the patient has glaucoma.",
        live=True
    ).launch(share=False)
