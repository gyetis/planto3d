from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from skimage import img_as_ubyte
import numpy as np
import cv2
from resize import *
from morph import *
from regen import *

def testGenerator(test_path,num_image,target_size, offset_height, offset_width):
    if num_image == 1:
        img = cv2.imread(test_path+"/1.png")
        img = resize_proportional(img, target_size, offset_height, offset_width)
        blank_image = np.zeros((target_size + (3,)), np.uint8)
        blank_image[:] = (255,255,255)
        img = blend_images(img, blank_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

    else:    
        for i in range(num_image):
            img = cv2.imread(test_path+"/%d.png"%(i+2))
            img = resize_proportional(img, target_size, offset_height, offset_width)
            blank_image = np.zeros((target_size + (3,)), np.uint8)
            blank_image[:] = (255,255,255)
            img = blend_images(img, blank_image)
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
        target = (256,512)
        model = load_model("./models/elevation.hdf5", custom_objects={'tf':tf})
        testGene = testGenerator(test_path, num_image=num_img, target_size=target, \
                                offset_height=target[0]/2, offset_width=target[1]/2)
        results = model.predict_generator(testGene, num_img)
        saveResult("./elev_predict",results)

        num_img = 1
        target = (512,256)
        model = load_model("./models/floorplan.hdf5", custom_objects={'tf':tf})
        testGene = testGenerator(test_path, num_image=num_img, target_size=target, \
                                offset_height=target[0]/5, offset_width=target[1]/5)
        results = model.predict_generator(testGene, num_img)
        saveResult("./plan_predict",results)
