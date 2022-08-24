import h5py
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from models.models import getModel
from skimage.color import label2rgb
from config import config
import os
import cv2

from utils.utils import readImageAndGetClicks, getClickMapAndBoundingBox, \
    getPatchs, sharpnessEnhancement, contrastEnhancement, \
    predictPatchs, postProcessing, generateInstanceMap, readImageAndGetSignals, predictSingleImage

seeddd = 1
img_rows = config.img_rows  # 480#640
img_cols = config.img_cols  # 768#1024
img_chnls = config.img_chnls
input_shape = (img_rows, img_cols)
testTimeAug = config.testTimeAug

def custom_get_click(img_path, posh5, negh5):
    x=[]
    y=[]
    pos_dset = h5py.File(posh5,'r')
    for pts in pos_dset["coordinates"]:
        x.append(pts[0])
        y.append(pts[1])
    neg_dset = h5py.File(negh5, 'r')
    for pts in neg_dset["coordinates"]:
        x.append(pts[0])
        y.append(pts[1])
    img_return = cv2.imread(img_path)
    return img_return[:, :, ::-1], np.round(x).astype(int), np.round(y).astype(int)


def main():
    modelType = config.modelType  # ['MultiScaleResUnet']
    lossType = config.lossType
    modelBaseName = 'NuClick_%s_%s_%s' % (config.application, modelType, lossType)
    modelSaveName = "%s/weights-%s.h5" % (config.weights_path, modelBaseName)

    # loading models
    model = getModel(modelType, lossType, input_shape)
    model.load_weights(modelSaveName)

    data_dir = ".../BCData/images/test/"
    result_dir = ".../BCData/images/nuclick/"
    pos_h5_dir = ".../BCData/annotations/test/positive"
    neg_h5_dir = ".../BCData/annotations/test/negative"

    imgPaths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".png")]
    pos_h5Paths = [os.path.join(pos_h5_dir, file) for file in os.listdir(pos_h5_dir) if file.endswith(".h5")]
    neg_h5Paths = [os.path.join(neg_h5_dir, file) for file in os.listdir(neg_h5_dir) if file.endswith(".h5")]
    imgPaths.sort()
    pos_h5Paths.sort()
    neg_h5Paths.sort()
    for imgPath, pos_h5Path, neg_h5Path in zip(imgPaths, pos_h5Paths, neg_h5Paths):
        if config.application in ['Cell', 'Nucleus']:
            img, cx, cy = custom_get_click(imgPath, pos_h5Path, neg_h5Path)
            m, n = img.shape[0:2]
            clickMap, boundingBoxes = getClickMapAndBoundingBox(cx, cy, m, n)
            patchs, nucPoints, otherPoints = getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n)
            dists = np.float32(
                np.concatenate((nucPoints, otherPoints, otherPoints), axis=3))  # the last one is only dummy!
            # prediction with test time augmentation
            predNum = 0  # augNum*numModel
            preds = np.zeros((len(patchs), img_rows, img_cols), dtype=np.float32)
            preds += predictPatchs(model, patchs, dists, config.testTimeJittering)
            predNum += 1
            print("Original images prediction, DONE!")
            if testTimeAug:
                print("Test Time Augmentation Started")
                # sharpenning the image
                patchs_shappened = patchs.copy()
                for i in range(len(patchs)):
                    patchs_shappened[i] = sharpnessEnhancement(patchs[i])
                temp = predictPatchs(model, patchs_shappened[:, :, ::-1], dists[:, :, ::-1], config.testTimeJittering)
                preds += temp[:, :, ::-1]
                predNum += 1
                print("Sharpenned images prediction, DONE!")

                # contrast enhancing the image
                patchs_contrasted = patchs.copy()
                for i in range(len(patchs)):
                    patchs_contrasted[i] = contrastEnhancement(patchs[i])
                temp = predictPatchs(model, patchs_contrasted[:, ::-1, ::-1], dists[:, ::-1, ::-1],
                                     config.testTimeJittering)
                preds += temp[:, ::-1, ::-1]
                predNum += 1
                print("Contrasted images prediction, DONE!")
            preds /= predNum
            try:
                masks = postProcessing(preds, thresh=config.Thresh, minSize=config.minSize, minHole=config.minHole,
                                       doReconstruction=True, nucPoints=nucPoints)
            except:
                masks = postProcessing(preds, thresh=config.Thresh, minSize=config.minSize, minHole=config.minHole,
                                       doReconstruction=False, nucPoints=nucPoints)
            instanceMap = generateInstanceMap(masks, boundingBoxes, m, n)
            instanceMap_RGB = label2rgb(instanceMap, image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0),
                                        image_alpha=1,
                                        kind='overlay')
            # plt.figure(), plt.imshow(instanceMap_RGB)
            # plt.show()
            imsave(os.path.join(result_dir,imgPath.split("/")[-1].split(".")[0]) + '_overlay.png', instanceMap_RGB)
            imsave(os.path.join(result_dir,imgPath.split("/")[-1].split(".")[0]) + '_instances.png', instanceMap * 255)
            imsave(os.path.join(result_dir,imgPath.split("/")[-1].split(".")[0]) + '_points.png', np.uint8(255 * np.sum(nucPoints, axis=(0, 3))))
            # plt.figure(),plt.imshow(img)

        if config.application == 'Gland':
            img, markups, imgPath = readImageAndGetSignals(os.getcwd())
            instanceMap = predictSingleImage(model, img, markups)
            instanceMap_RGB = label2rgb(np.uint8(instanceMap), image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0),
                                        image_alpha=1, kind='overlay')
            plt.figure(), plt.imshow(instanceMap_RGB), plt.show()
            imsave(imgPath[:-4] + '_instances.png', instanceMap)
            imsave(imgPath[:-4] + '_signals.png', markups)


if __name__ == '__main__':
    main()