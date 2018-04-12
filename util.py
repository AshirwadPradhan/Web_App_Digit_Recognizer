import numpy as np
from skimage.transform import resize, SimilarityTransform, warp
from skimage.util import invert
from PIL import Image
import matplotlib.pyplot as plt

def img_resize(img):

    #inverting the image
    img = invert(img)
    row, col = img.shape
    pad=400
    tmp = np.zeros((row+2*pad, col+2*pad)).astype(int)
    tmp[pad:pad+row,pad:pad+col] = img

    zY, zX = np.where(tmp)
    # bounding rectangle left upper coordinate
    ly, lx = zY.min(), zX.min()
    # bounding rectangle right bottom coordinate
    ry, rx = zY.max(), zX.max()


    if (rx-lx) < (ry-ly):
        rx = lx+(ry-ly)

    if (rx-lx) > (ry-ly):
        ry = ly+(rx-lx)

    img = resize(tmp[ly:ry,lx:rx].astype(float), (20, 20))
    # Now inserting the 20x20 image
    tmp = np.zeros((28,28))
    tmp[0:20,0:20] = img

    # Calculating translation
    
    Y, X = np.where(tmp)
    R, C = tmp.shape

    tsy, tsx = np.round(R/2-Y.mean()), np.round(C/2-X.mean())
    # Moving the digit
    tf = SimilarityTransform(translation=(-tsx, -tsy))
    tmp = warp(tmp, tf)
    tmp = np.round(tmp).astype(int)
    return tmp
    
    



if __name__ == '__main__':
    
    img = Image.open('test_img/five.png').convert('L')
    img_arr = np.array(img)
    print(img_arr)
    res_img_arr = img_resize(img_arr)
    print(res_img_arr)
    plt.imshow(res_img_arr, cmap=plt.get_cmap('gray'))
    plt.show()
    