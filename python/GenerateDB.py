import cv2
import copy
import random
from scipy import ndimage
import os
import shutil


def imgToBoxes(img, tab=None):
    if tab is None:
        tab = []
    x = y = 0
    for i in range(27):
        for j in range(27):
            tmp = img[x:x + 50, y:y + 50]
            tmp = cv2.resize(tmp, (34, 34), interpolation=cv2.INTER_AREA)
            tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 8)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            tmp = cv2.erode(tmp, kernel, iterations=1)
            tab.append(tmp)
            x = x + 50
        x = 0
        y = y + 50
    return tab


def imageMerge(background, figure):
    for i in range(len(background)):
        for j in range(len(background[i])):
            if figure[i][j] == 0:
                background[i][j] = 0


def createDBElem(tab, path, ret=None):
    if ret is None:
        ret = []
    for i in range(1, 10):
        dPath = path + "/" + str(i)
        tmpTab = copy.deepcopy(tab)
        dir = os.listdir(dPath)
        for j in range(len(tmpTab)):
            tmpFigure = dir[j+1]
            tmpFigure = cv2.imread(dPath + "/" + tmpFigure, 0)
            _, tmpFigure = cv2.threshold(tmpFigure, 127, 255, cv2.THRESH_BINARY_INV)
            tmpFigure = tmpFigure[2:26, 2:26]
            tmpFigure = cv2.resize(tmpFigure, (28, 28), interpolation=cv2.INTER_AREA)
            rdm = random.randint(1, 4)
            if rdm != 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                tmpFigure = cv2.erode(tmpFigure, kernel, iterations=1)
            rdmXOff = random.randint(0, 8)
            rdmYOff = random.randint(0, 8)
            rdmX = rdmXOff
            rdmY = rdmYOff
            if rdmXOff < 0:
                rdmX = 0
            if rdmYOff < 0:
                rdmY = 0
            imageMerge(tmpTab[j][rdmX:rdmXOff + 28, rdmY:rdmYOff + 28], tmpFigure)
        ret.append(tmpTab)
        print("Progression ", i * 10, "%")
    ret.append(tab)
    return ret


def createDataBaseDir(elems):
    dirname = 'GeneratedDB'
    if os.path.exists(dirname) and os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    dirType = dirname + "/" + "train"
    if os.path.exists(dirType) and os.path.isdir(dirType):
        shutil.rmtree(dirType)
    os.makedirs(dirType)

    dirType = dirname + "/" + "test"
    if os.path.exists(dirType) and os.path.isdir(dirType):
        shutil.rmtree(dirType)
    os.makedirs(dirType)

    for i in range(10):
        random.shuffle(elems[i])
        dirType = dirname + "/" + "test"
        elemDirPath = dirType + '/' + str(i+1)
        if os.path.exists(elemDirPath) and os.path.isdir(elemDirPath):
            shutil.rmtree(elemDirPath)
        os.makedirs(elemDirPath)

        dirType = dirname + "/" + "train"
        elemDirPath = dirType + '/' + str(i+1)
        if os.path.exists(elemDirPath) and os.path.isdir(elemDirPath):
            shutil.rmtree(elemDirPath)
        os.makedirs(elemDirPath)
        trainLimit = int(len(elems[i]) * 0.85)
        for j in range(trainLimit):
            cv2.imwrite(elemDirPath + "/" + str(i+1) + '_' + str(j) + ".png", elems[i][j])

        dirType = dirname + "/" + "test"
        elemDirPath = dirType + '/' + str(i+1)
        for j in range(trainLimit, len(elems[i])):
            cv2.imwrite(elemDirPath + "/" + str(i+1) + '_' + str(j) + ".png", elems[i][j])

        print("Progression ", i * 10, "%")


print("Creation des cases vides...")
img1 = cv2.imread('sudokuv.jpg', 0)
img2 = cv2.imread('sudokuv.jpg', 0)
img3 = cv2.imread('sudokuv.jpg', 0)

r1_img = cv2.resize(ndimage.rotate(img1, 0.1), (1350, 1350), interpolation=cv2.INTER_AREA)
r2_img = cv2.resize(ndimage.rotate(img1, 0.1), (1350, 1350), interpolation=cv2.INTER_AREA)
r3_img = cv2.resize(ndimage.rotate(img1, 0.2), (1350, 1350), interpolation=cv2.INTER_AREA)
r4_img = cv2.resize(ndimage.rotate(img1, 0.25), (1350, 1350), interpolation=cv2.INTER_AREA)
r5_img = cv2.resize(ndimage.rotate(img1, 0.15), (1350, 1350), interpolation=cv2.INTER_AREA)
r6_img = cv2.resize(ndimage.rotate(img1, 0.1), (1350, 1350), interpolation=cv2.INTER_AREA)
r7_img = cv2.resize(ndimage.rotate(img1, 0.35), (1350, 1350), interpolation=cv2.INTER_AREA)
r8_img = cv2.resize(ndimage.rotate(img1, 0.15), (1350, 1350), interpolation=cv2.INTER_AREA)

r9_img = cv2.resize(ndimage.rotate(img1, -0.1), (1350, 1350), interpolation=cv2.INTER_AREA)
r10_img = cv2.resize(ndimage.rotate(img1, -0.1), (1350, 1350), interpolation=cv2.INTER_AREA)
r11_img = cv2.resize(ndimage.rotate(img1, -0.15), (1350, 1350), interpolation=cv2.INTER_AREA)
r12_img = cv2.resize(ndimage.rotate(img1, -0.2), (1350, 1350), interpolation=cv2.INTER_AREA)
r13_img = cv2.resize(ndimage.rotate(img1, -0.25), (1350, 1350), interpolation=cv2.INTER_AREA)
r14_img = cv2.resize(ndimage.rotate(img1, -0.1), (1350, 1350), interpolation=cv2.INTER_AREA)
r15_img = cv2.resize(ndimage.rotate(img1, -0.35), (1350, 1350), interpolation=cv2.INTER_AREA)
r16_img = cv2.resize(ndimage.rotate(img1, -0.15), (1350, 1350), interpolation=cv2.INTER_AREA)

cropped_boxes = imgToBoxes(img1)
cropped_boxes = imgToBoxes(img2, cropped_boxes)
cropped_boxes = imgToBoxes(img3, cropped_boxes)
cropped_boxes = imgToBoxes(r1_img, cropped_boxes)
cropped_boxes = imgToBoxes(r2_img, cropped_boxes)
cropped_boxes = imgToBoxes(r3_img, cropped_boxes)
cropped_boxes = imgToBoxes(r4_img, cropped_boxes)
cropped_boxes = imgToBoxes(r5_img, cropped_boxes)
cropped_boxes = imgToBoxes(r6_img, cropped_boxes)
cropped_boxes = imgToBoxes(r7_img, cropped_boxes)
cropped_boxes = imgToBoxes(r8_img, cropped_boxes)
cropped_boxes = imgToBoxes(r9_img, cropped_boxes)
cropped_boxes = imgToBoxes(r10_img, cropped_boxes)
cropped_boxes = imgToBoxes(r11_img, cropped_boxes)
cropped_boxes = imgToBoxes(r12_img, cropped_boxes)
cropped_boxes = imgToBoxes(r13_img, cropped_boxes)
cropped_boxes = imgToBoxes(r14_img, cropped_boxes)
cropped_boxes = imgToBoxes(r15_img, cropped_boxes)
cropped_boxes = imgToBoxes(r16_img, cropped_boxes)

print(len(cropped_boxes), "crées...")
print("Création des cases vides terminée !")

print("Création des chiffres...")
path = "data/dataset(1)/dataset"
figures = createDBElem(cropped_boxes, path)
print("Creation des chiffres terminé !")
# cv2.imshow("test", figures[0][0])
# cv2.waitKey(0)
print("Création de la base de donnée...")
createDataBaseDir(figures)
print("Base de donnée crée !")
