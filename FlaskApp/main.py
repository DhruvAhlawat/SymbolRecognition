from flask import Flask, render_template, request, jsonify
import numpy as np;
import keras; 
import cv2;
import base64; 

app = Flask(__name__);
model = keras.models.load_model('FlaskApp/mnist_classification.h5'); #loads my prebuilt model 

@app.route('/',methods=['GET'])
def drawing():
    return render_template('drawing.html'); 

@app.route('/',methods=['POST'])
def canvas():
    canvasdata = request.form['canvasimg']; 
    encoded_data = request.form['canvasimg'].split(',')[1]; 
    
    #decoding this into a numpy ndarray
    nparr = np.fromstring(base64.b64decode(encoded_data),np.uint8); 
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR);     
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY); #// to convert it to grey pixels only

    gray_img = cv2.resize(gray_img,(28,28), interpolation=cv2.INTER_LINEAR); 

    img = np.expand_dims(gray_img,axis = 0); #//converts to (1,28,28)

    try:
        prediction = np.argmax(model.predict(img)); 
        print(f"predicted character is: {str(prediction)}"); 
        return render_template('drawing.html', response = str(prediction), canvasdata = canvasdata,success = True); 
    except Exception as t:
        return render_template('drawing.html', response=str(t), canvasdata=canvasdata); 
