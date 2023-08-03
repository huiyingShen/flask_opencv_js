



let rvec = new Math();
let rvec = new Math();
let cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [628.560, 0., 306.369, 0., 628.560, 238.215, 0., 0., 1.]);
let distCoeffs = cv.matFromArray(5, 1, cv.CV_64F, [-5.06355728e-02, 1.75634643e+00, 5.08400958e-03, -3.19544459e-03, -8.04546377e+00]);

cv.solvePnP	(	InputArray 	objectPoints,
    InputArray 	imagePoints,
    InputArray 	cameraMatrix,
    InputArray 	distCoeffs,
    OutputArray 	rvec,
    OutputArray 	tvec,
    bool 	useExtrinsicGuess = false,
    int 	flags = SOLVEPNP_ITERATIVE 
    )	