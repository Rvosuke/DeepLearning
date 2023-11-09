from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage, spatial
from sklearn.cluster import MeanShift
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


image = Image.open('ykf.png').convert('L')
image = np.array(image)

# 1. canny边缘检测
edges = canny(image, sigma=1.4, low_threshold=50, high_threshold=150)

# 2. 计算边的梯度方向
gx = ndimage.sobel(edges, axis=0, mode='constant')
gy = ndimage.sobel(edges, axis=1, mode='constant')
grad_direction = np.arctan2(gy, gx)

# 3. 检测线段
lines = hough_line(image, np.array([100, 0.1, np.pi / 180, 200, 15, 15]))

# 4. 生成圈子候选人
circles = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        l1 = lines[i][0]
        l2 = lines[j][0]

        # 计算中点和法线
        mx = (l1[0] + l1[2]) / 2
        my = (l1[1] + l1[3]) / 2
        nx = np.cos(grad_direction[mx, my])
        ny = np.sin(grad_direction[mx, my])

        # 计算交点
        x, y = hough_line_peaks([l1[0], l1[2]], [l1[1], l1[3]], [l2[0], l2[2]], [l2[1], l2[3]])
        r = np.linalg.norm([l1[0] - x, l1[1] - y])
        circles.append((x, y, r))


# 5. 非最大删除
def meanShiftClustering(circles):
    circles = np.array(circles)
    return MeanShift(bandwidth=50).fit_predict(circles)


def filterCircles(circles):
    # 计算每个圆圈的边缘支持
    edge_support = []
    for x, y, r in circles:
        edge_support.append(len(np.where(edges[y - r:y + r, x - r:x + r] > 0)[0]))

    # 计算每个圆圈的完整性
    completeness = []
    for x, y, r in circles:
        x1, y1 = x - r, y - r
        x2, y2 = x + r, y + r
        completeness.append(np.sum(edges[y1:y2, x1:x2]) / (2 * r * r))

    # 筛选圆圈
    final_circles = []
    for i in range(len(circles)):
        if edge_support[i] > 100 and completeness[i] > 0.5:
            final_circles.append(circles[i])

    return final_circles


final_circles = meanShiftClustering(circles)

# 6. 基于边缘支持和完整性的筛选
final_circles = filterCircles(final_circles)

# 转化为PIL Image
image = Image.fromarray(image)
# 画最后的圆
draw = ImageDraw.Draw(image)

for x, y, r in final_circles:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0), outline=(0, 255, 0), width=2)

image.save('detected_circles.jpg')
