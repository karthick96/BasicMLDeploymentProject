import numpy as np
import pickle
from flask import Flask, flash, request, render_template

app = Flask(__name__)
app.secret_key = "dec@2021"
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    final_features = [np.array([int(x) for x in request.form.values()])]
    pred_out = np.round(model.predict(final_features), 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(pred_out))


if __name__ == "__main__":
    app.run(debug=True)
