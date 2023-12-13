import numpy as np
import matplotlib.pyplot as plt

np.random.seed(52)
# 模拟生成一组基音周期轨迹
framerate = np.arange(0, 100)
pitch = np.concatenate([np.ones(50)*100, np.ones(50)*50], axis=0)

# 模拟"野点", 偏离到正常值的2倍或1/2, 设置比例
def wild_points(pitch, p=0.1):
    wild_num = int(len(pitch) * p)
    indexs = np.random.randint(0, len(pitch), size=wild_num)
    for i in indexs:
        if np.random.rand() > 0.5:
            pitch[i] = pitch[i] * 2
        else:
            pitch[i] = pitch[i] / 2
    return pitch

# 添加高斯噪声
def add_gaussian_noise(pitch, sigma=5):
    noise = np.random.normal(0, sigma, pitch.shape)
    return pitch + noise


# 中值平滑算法
def median_smoothing(pitch, window_size=5):
    smoothed_pitch = np.copy(pitch)
    edge = window_size // 2
    for i in range(edge, len(pitch) - edge):
        smoothed_pitch[i] = np.median(pitch[i - edge:i + edge + 1])
    return smoothed_pitch


# 线性平滑函数
def linear_smoothing(pitch, L):
	w = window_function(L)
	smoothed_pitch = np.convolve(pitch, w, mode='same')  # 使用'same'模式返回与x长度相同的输出
	return smoothed_pitch


# 定义窗函数 w(m)
def window_function(L):
    # 根据公式，窗函数是对称的，所以只需要定义一半然后翻转拼接即可
    if L == 1:
        # 根据图中的例子，窗函数 w(m) 的值
        w = [0.25, 0.5, 0.25]
    elif L == 2:
        # 如果窗长度不是3，可以使用其它方法生成窗函数，这里用等权重的例子
        w = [0.125, 0.2, 0.35, 0.2, 0.125]
    elif L == 3:
    	w = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
    return w


def delay_signal(x, D):
	
	x = x.copy()
    # 创建一个长度等于原信号加上延时长度的数组，并初始化为零
	delayed_x = np.zeros(len(x) + D)
    # 将原信号的值复制到新数组中，从索引D开始
	delayed_x[D:] = x
	x[:D] = 100
	x[D:] = delayed_x[D:len(x)]
	return x


def combined_smoothing1(pitch):
	pitch = median_smoothing(pitch, 5)
	pitch = median_smoothing(pitch, 3)
	return pitch


def combined_smoothing2(pitch):
	pitch = median_smoothing(pitch, 5)
	pitch = linear_smoothing(pitch, 3)
	return pitch


def quadratic_combined_smoothing1(pitch):
	pitch_smoothing_1 = combined_smoothing2(pitch)
	pitch_dif = pitch - pitch_smoothing_1
	pitch_smoothing_2 = combined_smoothing2(pitch_dif) + pitch_smoothing_1
	return pitch_smoothing_2


def quadratic_combined_smoothing2(pitch):
	pitch_smoothing_1 = combined_smoothing2(pitch)
	pitch_delay = delay_signal(pitch, 3)
	pitch_dif = pitch_delay - pitch_smoothing_1
	pitch_smoothing_2 = combined_smoothing2(pitch_dif) + pitch_smoothing_1
	return pitch_smoothing_2


# 添加高斯噪声
pitch = add_gaussian_noise(pitch)

# 生成"野点"
pitch = wild_points(pitch)

# 应用平滑
smoothed_pitch = median_smoothing(pitch)

# 绘图展示
plt.figure(figsize=(10, 5), dpi=100)
plt.scatter(framerate, pitch, color='r', label='Pitch Points')
plt.legend()
plt.scatter(framerate, smoothed_pitch, color='g', label='After Smoothing')
plt.legend()
plt.savefig('savefig_example.png')

# 这里保存了处理后的图像，路径为 '/mnt/data/savefig_example.png'
