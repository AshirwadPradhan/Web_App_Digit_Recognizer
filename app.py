from flask import Flask, render_template
from predict_digit import lModel

model = lModel()

app = Flask(__name__)

@app.route('/')
def homepage():
    return 'Hi'

if __name__ == '__main__':
    app.run()