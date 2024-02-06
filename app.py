# app.py
from flask import Flask, request, jsonify, render_template, Markup
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/static')


@app.route('/result', methods=['POST'])
def result():
	model = xgb.XGBClassifier()
	booster= xgb.Booster()
	booster.load_model("asset/model.model")
	model._Booster=booster
	model._le=LabelEncoder().fit(['0','1'])

	acousticness = int(request.form['acousticness']) 
	danceability = int(request.form['danceability']) 
	energy = int(request.form['energy']) 
	instrumentalness = int(request.form['instrumentalness']) 
	key = int(request.form['key']) 
	liveness = int(request.form['liveness']) 
	mode = int(request.form['mode']) 
	tempo = int(request.form['tempo']) 
	valence = int(request.form['valence']) 
	
	colNames=[[acousticness,danceability,energy,instrumentalness,key,liveness,mode,tempo,valence]]
	x_test=pd.DataFrame(colNames)
	x_test.columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'mode', 'tempo', 'valence']
	hasil=model.predict(x_test)
	hasilproba=model.predict_proba(x_test)
	probaclass0=str(round(hasilproba[0][0],3)*100.0)+"%" #data ke 0 index ke 0
	probaclass1=str(round(hasilproba[0][1],3)*100.0)+"%" #data ke 0 index ke 1
	print(hasil)
	if(hasil[0]=='1'):
		hasiltext="Populer"
	elif(hasil[0]=='0'):
		hasiltext="Tidak Populer"
	
	return render_template('result.html',p0=probaclass0,p1=probaclass1,hasiltext=hasiltext)
	#sabriend
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,threaded=True, port=7000,host='0.0.0.0')