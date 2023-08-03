var cv = require('./opencv.js');
var jpeg = require('jpeg-js');
var fs = require('fs');
const path = require("path");

var cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [9.663557e+02, 0., 2.0679e+02, 0.,9.663557e+02, 2.9370e+02, 0., 0., 1.]);
var distCoeffs = cv.matFromArray(5, 1, cv.CV_64F, [-1.5007354e-03, 9.87223898e-01, 1.718845e-02, -2.68059e-02,-2.33139e+00]);

const jpeg_data = fs.readFileSync(path.resolve(__dirname, "./aruco.jpg"));
var raw_data = jpeg.decode(jpeg_data);
var img = cv.matFromImageData(raw_data);
console.log(img);

var rvec = cv.matFromArray(3,1,cv.CV_64F,[-1.5007354e-03, 9.87223898e-01, 1.718845e-02]);
var rMat = new cv.Mat(3, 3, cv.CV_64F);
cv.Rodrigues(rvec,rMat);

