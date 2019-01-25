## Modified version of https://github.com/zhixuhao/unet/data.py

from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from skimage import img_as_ubyte
import numpy as np
import cv2

def resize_proportional(source_img, target_size, inter = cv2.INTER_AREA):
    (source_h, source_w) = source_img.shape[:2]
    (target_h, target_w) = target_size
    offset_h = target_h/10
    offset_w = target_w/10
    
    if float(source_w) / float(source_h) > 1.5 or float(source_h) / float(source_w) > 1.5 :
        if target_h < target_w and source_h < source_w:
            r = float(target_w-offset_w) / float(source_w)
            dim = (int(target_w-offset_w), int(source_h * r))
            
        if target_h < target_w and source_h > source_w:
            rot_img = cv2.rotate(source_img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            r = float(target_w-offset_w) / float(rot_img.shape[1])
            dim = (int(target_w-offset_w), int(rot_img.shape[0] * r))
            
        if target_h > target_w and source_h < source_w:
            rot_img = cv2.rotate(source_img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            r = float(target_h-offset_h) / float(rot_img.shape[0])
            dim = (int(rot_img.shape[1] * r), int(target_h-offset_h))
            
        if target_h > target_w and source_h > source_w:
            r = float(target_h-offset_h) / float(source_h)
            dim = (int(source_w * r), int(target_h-offset_h))
        
        resized = cv2.resize(source_img, dim, interpolation = inter)

        return resized
    
    else:
        if target_h < target_w and source_h < source_w:
            r = float(target_h-offset_h) / float(source_h)
            dim = (int(source_w * r), int(target_h-offset_h))
            
        if target_h < target_w and source_h > source_w:
            rot_img = cv2.rotate(source_img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            r = float(target_h-offset_h) / float(rot_img.shape[0])
            dim = (int(rot_img.shape[1] * r), int(target_h-offset_h))
            
        if target_h > target_w and source_h < source_w:
            rot_img = cv2.rotate(source_img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            r = float(target_w-offset_w) / float(rot_img.shape[1])
            dim = (int(target_w-offset_w), int(rot_img.shape[0] * r))
            
        if target_h > target_w and source_h > source_w:
            r = float(target_w-offset_w) / float(source_w)
            dim = ((target_w-offset_w), int(source_h * r))
        
        resized = cv2.resize(source_img, dim, interpolation = inter)

        return resized

def blend_images(fore, back):

    (fore_h, fore_w) = fore.shape[:2]
    (back_h, back_w) = back.shape[:2]
    offset_h = (back_h - fore_h) / 2
    offset_w = (back_w - fore_w) / 2

    blend = cv2.addWeighted(fore, 1, back[int(offset_h):int(offset_h)+fore_h, int(offset_w):int(offset_w)+fore_w], 0, 0, back)
    back[int(offset_h):int(offset_h)+fore_h, int(offset_w):int(offset_w)+fore_w] = blend
    return back

def testGenerator(test_path,num_image,target_size, blank):
    if num_image == 1:
        img = cv2.imread(test_path+"/1.png")
        img = resize_proportional(img, target_size)
        img = blend_images(img, blank)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

    else:    
        for i in range(num_image):
            img = cv2.imread(test_path+"/%d.png"%(i+2))
            img = resize_proportional(img, target_size)
            img = blend_images(img, blank)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255
            img = np.reshape(img,img.shape+(1,))
            img = np.reshape(img,(1,)+img.shape)
            yield img

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        img = img_as_ubyte(img)
        cv2.imwrite(save_path+"/%d_predict.png"%(i+1), img)

def test():
        test_path = "./uploads"
        num_img = 4
        target = (304,432)
        blank_image = np.zeros((target + (3,)), np.uint8)
        blank_image[:] = (255,255,255)
        model = load_model("./models/unet_ElevationData_epoch10_stepperepoch500.hdf5", custom_objects={'tf':tf})
        testGene = testGenerator(test_path, num_image=num_img, target_size=target, blank=blank_image)
        results = model.predict_generator(testGene, num_img)
        saveResult("./elevs_predict",results)

        num_img = 1
        target = (432,304)
        blank_image = np.zeros((target + (3,)), np.uint8)
        blank_image[:] = (255,255,255)
        model = load_model("./models/unet_FloorPlanData_epoch50_stepperepoch500.hdf5", custom_objects={'tf':tf})
        testGene = testGenerator(test_path, num_image=num_img, target_size=target, blank=blank_image)
        results = model.predict_generator(testGene, num_img)
        saveResult("./plan_predict",results)