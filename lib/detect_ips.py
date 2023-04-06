import time
from model.unet_model import *
from lib.utils import *


class Detector_Ips:
    pth = './model/ips_2t_2022_1013_1325.pth'
    width, height = 540, 640
    clipLimit = 20
    tileGridSize = 10

    batch_size = 32

    um_per_pixel = 0.648199975  # 每像素对应的实际长度（um）
    ratio = 4  # 计算偏差前先缩小的比例

    # 机械补偿误差
    offset_x = -786
    offset_y = 744

    # 细胞挑取面积限制
    area_limit = [5000, 500000]
    # 半径限制um  4000um / 2.592 um_per_pixel
    radius_limit = 4000 // 2.592
    # 允许的空缺比例
    vacancy = 0.1
    # 取框的半径
    center_len = 135

    # 使用Unet进行前后景分类
    def segment(self, im_ori: np.array):
        # 先进行clahe处理
        im_clahe = np.zeros(im_ori.shape, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileGridSize, self.tileGridSize))
        w, h = self.width, self.height
        row = im_clahe.shape[0] // h
        col = im_clahe.shape[1] // w
        for m in range(row):
            for n in range(col):
                temp = im_ori[m * h: (m + 1) * h, n * w: (n + 1) * w]
                im_clahe[m * h: (m + 1) * h, n * w: (n + 1) * w] = clahe.apply(temp)

        # 读取模型
        pth = torch.load(self.pth)
        model = pth['model']
        # ips前后景分割
        im_eval = self._unet_eval_npy(img_clahe=im_clahe, model=model, color_dict=pth['data_dict'], batch_size=self.batch_size)

        # 去除边缘外的部分
        c_x, c_y, r = get_hole_info(im_ori)
        if r is not None:
            mask = np.zeros(im_ori.shape, dtype=np.uint8)
            mask = cv2.circle(mask, (c_x, c_y), r, 255, -1)
            im_eval[mask != 255] = 0

        # 去掉小面积
        im_eval = select_obj(im_eval, area_min=self.area_limit[0])

        # 填补小孔洞
        im_eval = fill_hole(im_eval)
        return im_eval

    # 获取坐标
    def get_target(self, im_ori: np.array, im_eval: np.array):
        # 获取目标坐标
        targets, _, im_label = self._get_targets(im_ori, im_eval)
        locs = []
        for k, target in enumerate(targets):
            temp = {}
            # 换算机械坐标，添加补偿
            temp['center_x'] = round(target[0] * self.um_per_pixel * self.ratio) + self.offset_x
            temp['center_y'] = round(target[1] * self.um_per_pixel * self.ratio) + self.offset_y
            locs.append(temp)
        return im_label, locs

    # Unet模型专用的函数
    def _unet_eval_npy(self, img_clahe, model, color_dict: dict = None, batch_size: int = 32, img_ori: np.array = None):
        print('evaluating......')
        timer = time.time()
        origin_h, origin_w = img_clahe.shape[0], img_clahe.shape[1]  # 用于最后恢复尺寸的参数

        #  开始切割
        iw, ih = model.input_size
        gap = model.gap
        cw, ch = iw - 2 * gap, ih - 2 * gap  # 通过Unet网络后的图是经过裁剪的，上下左右各裁剪一个gap
        num_w = math.ceil((img_clahe.shape[1] - 2 * gap) / cw)
        num_h = math.ceil((img_clahe.shape[0] - 2 * gap) / ch)
        dif_w = int(num_w * cw + 2 * gap - img_clahe.shape[1])
        dif_h = int(num_h * ch + 2 * gap - img_clahe.shape[0])
        img_clahe = np.hstack([img_clahe, np.zeros((img_clahe.shape[0], dif_w), dtype=np.uint8)])
        img_clahe = np.vstack([img_clahe, np.zeros((dif_h, img_clahe.shape[1]), dtype=np.uint8)])
        img_clahe = torch.tensor(img_clahe, dtype=torch.float32)
        if img_ori is not None:
            img_ori = np.hstack([img_ori, np.zeros((img_ori.shape[0], dif_w), dtype=np.uint8)])
            img_ori = np.vstack([img_ori, np.zeros((dif_h, img_ori.shape[1]), dtype=np.uint8)])
            img_ori = torch.tensor(img_ori, dtype=torch.float32)
        tensors = []
        print('slicing the image...')
        for i in range(num_h):
            for j in range(num_w):
                im = img_clahe[i * ch:i * ch + ih, j * cw:j * cw + iw].clone() / 255
                if img_ori is None:
                    tensors.append(im.unsqueeze(0))
                else:
                    im2 = img_ori[i * ch:i * ch + ih, j * cw:j * cw + iw].clone() / 255
                    tensors.append(torch.stack([im2, im]))
                if len(tensors) > 0 and len(tensors) % 400 == 0:
                    print('total:' + str(len(tensors)))
        timer = time.time() - timer
        print('finish slicing, total: {}, cost time: {:.2f}s'.format(len(tensors), timer))
        # 分组
        residue = None
        if len(tensors) % batch_size == 0:
            times = int(len(tensors) / batch_size)
        else:
            residue = len(tensors) % batch_size
            times = int((len(tensors) - residue) / batch_size) + 1
        print('predicting... batch total: {}'.format(times))
        timer = time.time()
        list_pred = []
        # 识别
        model = model.eval()
        if torch.cuda.is_available():
            model.cuda()
        per = 10
        with torch.no_grad():
            for i in range(times):
                if i == times - 1 and residue is not None:
                    batch = torch.stack(tensors[-residue:])
                else:
                    batch = torch.stack(tensors[i * batch_size:(i + 1) * batch_size])
                if torch.cuda.is_available():
                    batch = batch.cuda()
                preds = model(batch)
                preds = preds.cpu()
                value, index = torch.max(preds, 1)
                del value
                for j in range(preds.shape[0]):
                    list_pred.append(index[j].numpy())
                if int((i + 1) / times * 100) >= per:
                    per += 10
                    print('predict: {}/{}'.format(i + 1, times))
                torch.cuda.empty_cache()

        img_eval = np.zeros([num_h * ch, num_w * cw], dtype=np.uint8)
        per = 10
        for i, im in enumerate(list_pred):
            r, c = i // num_w, i % num_w
            img_eval[r * ch: (r + 1) * ch, c * cw: (c + 1) * cw] = im
            if (i + 1) / len(list_pred) * 100 >= per:
                per += 10
                print('splice: {}/{}'.format(i + 1, len(list_pred)))
        # 填补四周的gap图像
        temp = np.zeros((img_eval.shape[0], gap), dtype=np.uint8)
        img_eval = np.hstack([temp, img_eval, temp])
        temp = np.zeros((gap, img_eval.shape[1]), dtype=np.uint8)
        img_eval = np.vstack([temp, img_eval, temp])
        # 裁剪出原来大小
        img_eval = img_eval[:origin_h, :origin_w]
        # 根据data_dict还原灰度值
        if color_dict is not None:
            for gray, v in color_dict.items():
                img_eval[img_eval == v] = gray
        timer = time.time() - timer
        print('finish predicting! cost time: {:.2f}s'.format(timer))
        return img_eval

    # im_eval是预测结果图,大小与原图相同，info_hole是孔所在行列，info_plate是板子信息
    def _get_targets(self, img_ori, img_eval):
        # im_label是用来将target按照要求进行分类标记
        img_label = cv2.merge([img_eval, img_eval, img_eval])
        targets = []
        centers = []
        dis_l = {}
        c_x, c_y, r = get_hole_info(img_ori)
        if r is None:
            c_x, c_y = img_ori.shape[1] // 2, img_ori.shape[0] // 2
        cv2.circle(img_label, (c_x, c_y), 15, (0, 0, 255), -1)
        num, label, stats, centroids = cv2.connectedComponentsWithStats(img_eval)
        # 筛选面积
        indexs = np.where((stats[1:, 4] >= self.area_limit[0]) & (stats[1:, 4] <= self.area_limit[1]))[0]
        print('检测到的所有目标：' + str(len(indexs)))
        for i, k in enumerate(indexs):
            k += 1
            dis1 = (centroids[k][0] - c_x) ** 2 + (centroids[k][1] - c_y) ** 2
            dis2 = int(r - self.radius_limit) ** 2
            if dis1 < dis2:
                loc_x, loc_y = int(centroids[k][0]), int(centroids[k][1])
                center = img_eval[loc_y - self.center_len:loc_y + self.center_len,
                         loc_x - self.center_len:loc_x + self.center_len]
                if (center == 0).sum() / center.size < self.vacancy:
                    center_x = int(centroids[k][0])
                    center_y = int(centroids[k][1])
                    topleft_x = stats[k][0]
                    topleft_y = stats[k][1]
                    w = stats[k][2]
                    h = stats[k][3]
                    area = stats[k][4]
                    dis_l[k] = dis1
                    targets.append([center_x, center_y, topleft_x, topleft_y, w, h, area])
                    center_ori = img_ori[center_y - self.center_len:center_y + self.center_len,
                                 center_x - self.center_len:center_x + self.center_len]
                    center_eval = img_eval[center_y - self.center_len:center_y + self.center_len,
                                  center_x - self.center_len:center_x + self.center_len]
                    centers.append([center_ori, center_eval])
        del i, k
        d_v = list(dis_l.values())
        d_v.sort()
        res = []
        for v in d_v:
            i = list(dis_l.values()).index(v)
            k = list(dis_l.keys())[i]
            res.append(targets[i])
            xc, yc, x, y, w, h = targets[i][:6]
            cut_label = img_label[y:y + h, x:x + w]
            cut_mask = label[y:y + h, x:x + w]
            cut_label[cut_mask == k] = (0, 255, 0)
            cv2.circle(img_label, (xc, yc), 10, (255, 255, 255), -1)
            cv2.line(img_label, (c_x, c_y), (xc, yc), (255, 0, 0), 3)
            cv2.putText(img_label, str(len(res)), (xc + 10, yc + 10), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 5)
        print('检测到的合适目标：' + str(len(res)))
        return res, centers, img_label





