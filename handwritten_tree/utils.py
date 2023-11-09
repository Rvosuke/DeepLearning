import numpy as np
from scipy import ndimage
from PIL import Image


# def canny(image: np.ndarray) -> np.ndarray:
#     """
#     Canny 边缘检测
#     :param image: 输入灰度图
#     :return:
#     """
#     # 高斯滤波去噪
#     blurred = ndimage.gaussian_filter(gray, sigma=1)
#
#     # 计算梯度强度和方向
#     gx = ndimage.sobel(blurred, axis=0)
#     gy = ndimage.sobel(blurred, axis=1)
#     grad_intensity = np.hypot(gx, gy)
#     grad_direction = np.arctan2(gy, gx) * 180 / np.pi
#
#     # 非极大值抑制
#     grad_thinned = non_max_suppression(grad_intensity, grad_direction)
#
#     # 双阈值处理
#     low_threshold = np.mean(grad_intensity) * 0.5
#     high_threshold = low_threshold * 3
#     grad_thinned = double_threshold(grad_thinned, low_threshold, high_threshold)
#
#     # 将输出转换为 uint8 类型, 并将1, 0转换为255, 0
#     grad_thinned = np.array(grad_thinned, dtype=np.uint8)
#     grad_thinned = grad_thinned * 255
#
#     return grad_thinned
#
#
# # 其他函数实现
# def non_max_suppression(grad_intensity, grad_direction):
#     """
#     非极大值抑制
#     :param grad_intensity: 梯度强度
#     :param grad_direction: 梯度方向
#     :return: 非极大值抑制结果
#     """
#     m, n = grad_intensity.shape
#
#     grad_suppressed = np.copy(grad_intensity)
#     for i in range(1, m - 1):
#         for j in range(1, n - 1):
#
#             # 获取当前点的梯度方向
#             direction = grad_direction[i, j]
#
#             # 根据梯度方向选择比较点坐标
#             if direction < 22.5 or direction > 157.5:
#                 x1, y1, x2, y2 = i, j + 1, i, j - 1
#             elif direction < 67.5:
#                 x1, y1, x2, y2 = i - 1, j - 1, i + 1, j + 1
#             elif direction < 112.5:
#                 x1, y1, x2, y2 = i - 1, j, i + 1, j
#             else:
#                 x1, y1, x2, y2 = i - 1, j + 1, i + 1, j - 1
#
#             # 比较梯度强度,保留局部极大值
#             if grad_intensity[i, j] >= grad_intensity[x1, y1] and grad_intensity[i, j] >= grad_intensity[x2, y2]:
#                 grad_suppressed[i, j] = grad_intensity[i, j]
#             else:
#                 grad_suppressed[i, j] = 0
#
#     return grad_suppressed
#
#
# def double_threshold(image, low, high):
#     strong_edges = np.zeros_like(image)
#
#     # Strong edges
#     strong_indices = np.where(image >= high)
#     strong_edges[strong_indices] = 1
#
#     # Weak edges
#     middle_indices = np.where((image <= high) & (image >= low))
#     weak_edges = np.zeros_like(image)
#     weak_edges[middle_indices] = 1
#
#     # Final edge map
#     edge_map = strong_edges + weak_edges
#
#     return edge_map

if __name__ == '__main__':
    img = Image.open('csu.jpg')
    # 缩放图片到 720*960
    img = img.resize((960, 720))
    gray = np.array(img.convert('L'))
    from skimage.feature import canny
    canny_edges = canny(gray)
    # print(canny_edges.shape)
    Image.fromarray(canny_edges).save('canny_edges.jpg')
