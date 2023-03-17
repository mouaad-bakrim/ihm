from statistics import mode
from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


# create route for index.html
@app.route('/')
def index():
    return render_template('index.html')
    
# create route for after.html after button click on index.html
@app.route('/after', methods=['GET', 'POST'])
def after():
    # store the uploaded image by user in static folder
    img = request.files['file1']
    img.save('static/file.jpg')


    # using opencv to detect faces in image
     
    img1 = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = img1[y:y+h, x:x+w]

    cv2.imwrite('static/after.jpg', img1)

    try:
        cv2.imwrite('static/cropped.jpg', cropped)

    except:
        pass


    # load the image using cv2 library and reshape to 48*48 pixel size
    try:
        img = cv2.imread('static/cropped.jpg', 0)
    except:
        img = cv2.imread('static/file.jpg', 0)


    
    img = cv2.resize(img,(48,48))
    img = img/255.0

    img = np.reshape(img, (1,48,48,1))


    # load the CNN model trained previously and make predictions
    model = load_model('emotion_detection_model.h5')
    prediction = model.predict(img)

    label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]

    
    return render_template("after.html", data=final_prediction)


if __name__ == "__main__":
    app.run(debug=True)
