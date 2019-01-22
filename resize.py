from __future__ import division
import cv2
import numpy as np

target = (304,432)
blank_image = np.zeros((target + (3,)), np.uint8)
blank_image[:] = (255,255,255)

def resize_proportional(source_img, target_size=(304,432), inter = cv2.INTER_CUBIC):
    (source_h, source_w) = source_img.shape[:2]
    (target_h, target_w) = target_size

    if target_h < target_w and source_h < source_w:
        r = float(target_w) / float(source_w)
        dim = (target_w, int(source_h * r))
    
    if target_h < target_w and source_h > source_w:
        rot_img = cv2.rotate(source_img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        r = float(target_w) / float(rot_img.shape[1])
        dim = (target_w, int(rot_img.shape[0] * r))
    
    if target_h > target_w and source_h < source_w:
        rot_img = cv2.rotate(source_img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        r = float(target_h) / float(rot_img.shape[0])
        dim = (int(rot_img.shape[1] * r), target_h)
    
    if target_h > target_w and source_h > source_w:
        r = float(target_h) / float(source_h)
        dim = (int(source_w * r), target_h)
    
    resized = cv2.resize(source_img, dim, interpolation = inter)

    return resized

def blend_images(fore, back):

    (fore_h, fore_w) = fore.shape[:2]

    blend = cv2.addWeighted(fore, 1, back[0:fore_h,0:fore_w], 0, 0, back)
    back[0:fore_h,0:fore_w] = blend
    return back