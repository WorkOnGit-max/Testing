from flask import Flask, request, render_template, send_file
import pickle


TSA_Model= Flask(__name__)


with open('model/svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


@TSA_Model.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@TSA_Model.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        user_input = request.form.get('text')
        if user_input: 
            transformed_input = vectorizer.transform([user_input])
            predicted_sentiment = svm_model.predict(transformed_input)
            result = predicted_sentiment[0]

            return render_template('result.html', result=result)
        else:
            return "No input provided"    

if __name__ == '__main__':
    TSA_Model.run(host='0.0.0.0', port=5000)
