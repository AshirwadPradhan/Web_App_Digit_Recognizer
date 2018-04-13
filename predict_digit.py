import pandas as pd
import numpy as np
from keras.models import load_model
from util import img_resize
from PIL import Image

def lModel():
    model = load_model('model/mnist_model.h5')
    return model

def pred_dig(X_test):
    X_test = X_test.reshape(1,28, 28, 1).astype('float32')

    X_test /= 255
    model = lModel()
    y_pred = model.predict_proba(X_test)

    y_conv = []
    for i in range(len(y_pred)):
        max_val = max(y_pred[i])
        y_arr = list(y_pred[i])
        max_index = y_arr.index(max_val)
        y_conv.append(max_index)

    return y_conv[0]

if __name__ == '__main__':
    img = Image.open('test_img/two.png').convert('L')
    img_arr = np.array(img)
    X_test = img_resize(img_arr)
    print(pred_dig(X_test))
