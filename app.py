from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle

model = pickle.load(open("nlp_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [[message]]
		# vect = cv.transform(data).toarray()
		my_prediction = model.predict(message)
	return render_template('home.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)