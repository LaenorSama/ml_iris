import flask
from flask import render_template
import pickle
import sklearn
import numpy as np

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        temp = 1
        with open('model.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)

        sepal_length = float(flask.request.form['sepal_length'])
        sepal_with = float(flask.request.form['sepal_with'])
        petal_length = float(flask.request.form['petal_length'])
        petal_with = float(flask.request.form['petal_with'])
        iris_class = ['setosa', 'versicolor', 'virginica']
        X = np.array([sepal_length, sepal_with, petal_length, petal_with])
        X = X.reshape(1, -1)
        temp = iris_class[loaded_model.predict(X)[0]]
        return render_template('main.html', result=temp)


if __name__ == '__main__':
    app.run()
