from flask import Flask,request,render_template,render_template
import pickle
import numpy as np

app = Flask("heartDiseasePreidiction")
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Heart Disease $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)