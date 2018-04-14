from flask import Flask, render_template, request, make_response
from predict_digit import lModel, pred_dig
from util import img_resize
from skimage import io as skio
from io import BytesIO
import base64
from PIL import Image
import re
import numpy as np
import base64

model = lModel()

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def homepage():
    return render_template('index.html')

def convertImage(imgData1):
    imgstr = str(imgData1)
    imgstr = re.search(r'base64,(.*)',imgstr).group(1)
    print(imgstr)
    # imgstr = imgData1
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict', methods=['POST'])
def predict():
    imgData = request.get_data()
    # print(imgData)
    convertImage(imgData)
    img = Image.open('output.png').convert('L')
    img_arr = np.array(img)
    X_test = img_resize(img_arr)
    num = pred_dig(X_test)
    print(num)

    return make_response(str(num), 200)


if __name__ == '__main__':
    app.run()