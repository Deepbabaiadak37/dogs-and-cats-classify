from flask import Flask,url_for,render_template,request,jsonify,redirect
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os

UPLOAD_FOLDER='static/uploads/'

app=Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


@app.route('/',methods=['GET','POST'])
def home():
	str=""
	return render_template('index.html')




@app.route('/result',methods=['GET','POST'])
def function():
	file = request.files['file']
	name=file.filename
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
	#return redirect(url_for('static',filename="uploads/"+name),code=301)

	model = load_model('model_rcat_dog.h5')
	#test_image = image.load_img({url_for('static', filename='dog1.jpg')}, target_size = (64,64))
	test_image = image.load_img('./static/uploads/'+name, target_size = (64,64))
	test_image = image.img_to_array(test_image)
	test_image=test_image/255
	test_image = np.expand_dims(test_image, axis = 0)
	result = model.predict(test_image)
	
	str=""
	if result[0]<0:
		str="The image is predicted as  cat"
	else:
		str="The image is predicted as  dog"
	
	return render_template('home.html',predict=str,name=name)

if __name__=='__main__':
	app.run(debug=True)