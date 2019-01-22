from keras import backend as K
from keras.models import load_model
import glob
import tensorflow as tf
from skimage import img_as_ubyte
import numpy as np
import cv2

def testGenerator(test_path,num_image = 30,target_size = (864,608)):
    for i in range(num_image):
        img = cv2.imread(test_path+"/%d.png"%(i+1), 0)
        img = img / 255
        img = cv2.resize(img, (0,0), fx=target_size[0]/img.shape[0], fy=target_size[1]/img.shape[1])
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        img = img_as_ubyte(img)
        cv2.imwrite(save_path+"/%d_predict.png"%(i+1), img)

test_path = "./elevs"
num_img = len(glob.glob(test_path + "/*png"))
model = load_model("./models/unet_ElevationData_epoch10_stepperepoch500.hdf5", custom_objects={'tf':tf})
testGene = testGenerator(test_path, num_image=num_img, target_size=(304,432))
results = model.predict_generator(testGene, num_img, verbose=1)
saveResult("./elevs_predict",results)

test_path = "./plan"
num_img = len(glob.glob(test_path + "/*png"))
model = load_model("./models/unet_FloorPlanData_epoch50_stepperepoch500.hdf5", custom_objects={'tf':tf})
testGene = testGenerator(test_path, num_image=num_img, target_size=(432,304))
results = model.predict_generator(testGene, num_img, verbose=1)
saveResult("./plan_predict",results)