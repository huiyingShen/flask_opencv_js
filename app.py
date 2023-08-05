from flask import Flask,render_template,request,jsonify
from wtforms import Form, StringField

import cv2
import numpy as np
import os,time



def data_uri_to_cv2_img(data_uri):
    nparr = np.fromstring(data_uri.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# board = charuco_board.TheBoard();
app = Flask(__name__, static_folder="static", template_folder='templates')



@app.route("/")
def camio():
    return render_template('camio.html')

@app.route("/ps1")
def ps1():
    return render_template('ps1.html')

@app.route("/hands")
def hands():
    return render_template('hands.html')

@app.route("/pnp")
def pnp():
    return render_template('solvePnP.html')


@app.route("/pnp2")
def pnp2():
    return render_template('solvePnP2.html')
    
@app.route("/aruco")
def aruco():
    return render_template('aruco.html')

@app.route("/cam")
def cam():
    return render_template('arucoCamera.html')

if __name__ == "__main__":
    # cer = os.path.join(os.path.dirname(__file__), 'server.crt')
    # key = os.path.join(os.path.dirname(__file__), 'server.key')
    # app.run(host='0.0.0.0', port=8443, debug = True, ssl_context=(cer, key))
    app.run(host='localhost', port=8000, debug = True)

