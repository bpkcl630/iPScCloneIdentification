from lib.detect_ips import Detector_Ips
import cv2
import os
import pytest

path_data = './data'
path_save = './result'


def test_eval_imgs():
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    for file in os.listdir(path_data):
        im_ori = cv2.imread(os.path.join(path_data, file), 0)
        tool = Detector_Ips()
        show = tool.segment(im_ori)
        cv2.imwrite(path_save + '/' + file[:file.rfind('_')] + '_eval.png', show)


def test_get_locs():
    for file in os.listdir(path_data):
        im_ori = cv2.imread(os.path.join(path_data, file), 0)
        temp = file[:file.rfind('_')] + '_eval.png'
        if not os.path.exists(os.path.join(path_save, temp)):
            raise Exception(temp + ' do not exist!!!')
        im_eval = cv2.imread(os.path.join(path_save, file[:file.rfind('_')] + '_eval.png'), 0)
        tool = Detector_Ips()
        show, locs = tool.get_target(im_ori, im_eval)
        txt = open(path_save + '/' + file[:-4] + '_loc.txt', mode='w')
        for i, target in enumerate(locs):
            txt.write('id: ' + str(i + 1) + ' x:' + str(target['center_x']) + ' y:' + str(target['center_y']) + '\n')
        txt.close()
        cv2.imwrite(path_save + '/' + file[:-4] + '_targets.png', show)




