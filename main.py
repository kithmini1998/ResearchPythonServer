from email import message
from fileinput import filename
from flask import Flask, request
from flask import render_template
import settings
import utils
import numpy as np
import cv2
import Predictions as pred

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'medical_document_scanner_app'

docscan = utils.DocumentScan()

@app.route('/',methods=['GET','POST'])
def scandoc():
    if request.method == 'POST':
        file = request.files['image_name']
        upload_image_path = utils.save_upload_image(file)
        print('Image saved in = ',upload_image_path)
        # Predict the coordinates of the document
        four_points, size = docscan.document_scanner(upload_image_path)
        print(four_points,size)
        
        if four_points is None:
            message = 'Unable to locate the coordinates: points displayed are random'
            points = [
                {'x':10 , 'y': 10},
                {'x':120 , 'y': 10},
                {'x':120 , 'y': 120},
                {'x':10 , 'y': 120}
            ]       
            return render_template('scanner.html',
                                   points=points,
                                   fileupload=True,
                                   message=message)
        else:
            points = utils.array_to_json_format(four_points)
            message = 'Located the Coordinates of Document using OpenCV'
            return render_template('scanner.html',
                                   points=points,
                                   fileupload=True,
                                   message=message)
                                   
        return render_template('scanner.html')
      
        
    return render_template('scanner.html')


@app.route('/transform',methods=['POST'])
def transform():
    try:
        points = request.json['data']
        array = np.array(points)
        magic_color = docscan.calibrate_to_original_size(array)
        #utils.save_image(magic_color,'magic_color.jpg')
        filename = 'magic_color.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR,filename)
        cv2.imwrite(magic_image_path,magic_color)
        return 'success'
    except:
        return 'fail'
    
@app.route('/prediction')
def prediction():
    #load the wrap image
    wrap_image_filepath = settings.join_path(settings.MEDIA_DIR,'magic_color.jpg')
    image = cv2.imread(wrap_image_filepath)
    image_bb,results = pred.getPredictions(image)
    bb_filename = settings.join_path(settings.MEDIA_DIR,'bounding_box.jpg')
    cv2.imwrite(bb_filename,image_bb)
    print(results)
    return render_template('predictions.html',results=results)
        
    
@app.route('/diabetes', methods=['GET','POST'])
def diabetesPred():
    if request.method == 'GET':
        return render_template('diabetes.html')
    if request.method == 'POST':
        
        data = pd.read_csv("diabetes.csv")
        X = data.drop("Outcome", axis=1)
        Y = data['Outcome']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        val1 = float(request.form['n1'])
        val2 = float(request.form['n2'])
        val3 = float(request.form['n3'])
        val4 = float(request.form['n4'])
        val5 = float(request.form['n5'])
        val6 = float(request.form['n6'])
        val7 = float(request.form['n7'])
        val8 = float(request.form['n8'])
        
        # val1 = float(request.GET['n1'])
        # val2 = float(request.GET['n2'])
        # val3 = float(request.GET['n3'])
        # val4 = float(request.GET['n4'])
        # val5 = float(request.GET['n5'])
        # val6 = float(request.GET['n6'])
        # val7 = float(request.GET['n7'])
        # val8 = float(request.GET['n8'])
        
        pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
        
        result1 = ""
        if pred==[1]:
            result1 = "Positive"
        else:
            result1 = "Negative"
        return render_template('diabetes.html',results2=result1)
  
#@app.route('/getData', methods=['GET','POST'])
#def getDocumentData():   
    #return
    
    


if __name__ == "__main__":
    app.run(debug=True)