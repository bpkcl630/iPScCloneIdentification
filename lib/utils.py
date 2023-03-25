import numpy as np
import cv2


# 获取圆孔信息
def get_hole_info(im: np.array, r_limit=None, threshold: int = 10, show_flag: bool = False):
    im_binary = None
    im_border = None
    x = None
    y = None
    r = None
    if r_limit is None:
        r_limit = im.shape[0]
    len_limit1 = 1.1
    len_limit2 = 0.7
    ratio = 0.1
    while r is None and threshold > 2:
        im_binary = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 301, threshold)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(im_binary)
        im_border = np.zeros(im_binary.shape, np.uint8)
        id1 = np.intersect1d(np.argwhere(stats[1:, 2] > len_limit2 * r_limit), np.argwhere(stats[1:, 2] < len_limit1 * r_limit)).tolist()
        id2 = np.intersect1d(np.argwhere(stats[1:, 3] > len_limit2 * r_limit), np.argwhere(stats[1:, 3] < len_limit1 * r_limit)).tolist()
        ids = id1 + id2
        for id in ids:
            im_border[labels == id + 1] = 255
            x, y, w, h, _ = stats[id + 1]
            im_border[y:y + h, x:x + w][labels[y:y + h, x:x + w] == id + 1] = 255
        if im_border.max() == 0:
            threshold -= 1
            continue
        x1, x2, y1, y2 = get_loc(im_border)
        if x2 - x1 < len_limit2 * r_limit or y2 - y1 < len_limit2 * r_limit:
            threshold -= 1
            continue
        temp = cv2.resize(im_border, (0, 0), fx=ratio, fy=ratio)
        # 第一次检测圆
        circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, r_limit * ratio, param1=700, param2=20,
                                   minRadius=int(0.4 * r_limit * ratio), maxRadius=int(0.55 * r_limit * ratio))
        if circles is not None:
            x, y, r = np.int16(np.around(circles))[0, 0]
            mask = np.zeros(temp.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r - 20,  255, -1)
            temp[mask != 255] = 0
            # 第二次检测圆
            circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, r_limit * ratio, param1=700, param2=20,
                                       minRadius=int(0.3 * r_limit * ratio), maxRadius=int(0.55 * r_limit * ratio))
            if circles is not None:
                x, y, r = np.int16(np.around(circles))[0, 0]
            x = int(x / ratio)
            y = int(y / ratio)
            r = int(r / ratio)
        else:
            threshold -= 1

    if show_flag:
        show1 = cv2.resize(im_binary, (0, 0), fx=0.05, fy=0.05)
        show2 = cv2.resize(im_border, (0, 0), fx=0.05, fy=0.05)
        mask = np.zeros(im.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (x, y), r, 255, -1)
        show3 = cv2.resize(mask, (0, 0), fx=0.05, fy=0.05)
        cv2.imshow('1', np.hstack([show1, show2, show3]))
        cv2.waitKey()
    return [x, y, r]


# 获取连通域的坐标信息
def get_loc(img: np.array):
    if len(img.shape) != 2:
        raise Exception("This function only can be used for Gray image!")
    h, w = img.shape
    x1 = x2 = y1 = y2 = None
    for i in range(h):
        row = img[i, :]
        if row.max() == 255:
            y1 = i
            break
    for i in range(h):
        row = img[h - 1 - i, :]
        if row.max() == 255:
            y2 = h - i
            break
    for i in range(w):
        col = img[:, i]
        if col.max() == 255:
            x1 = i
            break
    for i in range(w):
        col = img[:, w - 1 - i]
        if col.max() == 255:
            x2 = w - i
            break
    return [x1, x2, y1, y2]


# 填补孔洞
def fill_hole(im: np.array, seedpoint=(0, 0)):
    """
    这个函数只用来给分割出来的单个细胞图的小空洞进行填补
    :param seedpoint: 种子点
    :param im: 需要进行填补小空洞的图像
    :return: 填补后的图像
    """
    im_floodfill = im.copy()
    h, w = im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    x, y = seedpoint[0], seedpoint[1]
    # floodFill函数中的seedPoint对应像素必须是背景
    if im_floodfill[y, x] != 0:
        return im
    cv2.floodFill(im_floodfill, mask, (x, y), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im | im_floodfill_inv
    return im_out


# 面积筛选
def select_obj(im: np.array, area_min=None, area_max=None, lw_ratio=None, len_limit=None, ratio=None):
    """
    筛选符合特定面积范围的细胞并保留，其他去除
    :param im: 整张预测图，二值化的
    :param area_min:最小面积 单位：像素个数
    :param area_max: 最大面积
    :param lw_ratio: 长宽比限制
    :param len_limit: 长宽长度限制
    :param ratio: 能忍受的最小空缺比例（掩膜图中占有的比例）
    :return: 筛选后的图像
    """
    im_out = np.zeros(im.shape, dtype=np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(im)
    gap = 5
    for i in range(1, num):
        if area_min is not None and stats[i][4] < area_min:
            continue
        if area_max is not None and stats[i][4] > area_max:
            continue
        if len_limit is not None and (stats[i][3] > len_limit or stats[i][2] > len_limit):
            continue
        if lw_ratio is not None and stats[i][2] / stats[i][3] > lw_ratio:
            continue
        if lw_ratio is not None and stats[i][3] / stats[i][2] > lw_ratio:
            continue
        x1 = stats[i][0] - gap if stats[i][0] - gap > 0 else 0
        y1 = stats[i][1] - gap if stats[i][1] - gap > 0 else 0
        x2 = stats[i][0] + stats[i][2] + gap
        if x2 > im_out.shape[1]:
            x2 = im_out.shape[1]
        y2 = stats[i][1] + stats[i][3] + gap
        if y2 > im_out.shape[0]:
            y2 = im_out.shape[0]
        temp2 = labels[y1:y2, x1:x2]
        if ratio is not None and (temp2 == i).sum() / temp2.size < ratio:
            continue
        temp1 = im_out[y1:y2, x1:x2]
        temp1[temp2 == i] = 255
    return im_out



