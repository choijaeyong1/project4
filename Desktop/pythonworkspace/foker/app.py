from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
loaded_model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/error', methods=['POST'])
def error():
    return render_template('error.html')

@app.route('/predict', methods=['POST'])
def predict():
    foker = ['탑', '원페어', '투페어', '트리플', '스트레이트', '플러시', 
             '풀하우스', '포카드', '스트레이트플러시', '로얄스트레이트플러시']
    a = 0
    input_data = []
    for i in range(1, 63):
        input_field = f'input_data{i}'
        input_value = request.form.get(input_field)
        if input_value:
            input_data.append(int(input_value))
        else:
            a = 1
            break

    if a == 1:
        return render_template('error.html')

    for j in range(0, len(input_data), 2):
        for k in range(j+2, len(input_data), 2):
            if input_data[j:j+2] == input_data[k:k+2]:
                a = 1
                break
            elif input_data[j] < 1 or input_data[j] > 4:
                a = 2
                break
            elif input_data[j+1] < 1 or input_data[j+1] > 13:
                a = 3
                break

    if a == 1 or a == 2 or a == 3:
        return render_template('error.html')

    input_data = np.array([input_data])
    predicted_value = loaded_model.predict(input_data)
    for i in range(10):
        if predicted_value == i:
            result = foker[i]
            break
    return render_template('result.html', predicted_value=result)

if __name__ == '__main__':
    app.run(debug=True)