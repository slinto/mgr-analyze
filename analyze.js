/**
 * node analyze leaf_1.jpg
 */
const cv = require('opencv');
const pixelmatch = require('pixelmatch');

/**
 * CONSTANTS
 */
// B,G,R
const GREEN = [0, 255, 0];
const WHITE = [255, 255, 255];
const RED = [0, 0, 255];

const LINE_TYPE = 8;
const THICKNESS = 3;

/**
 * Save image with specified name.
 * @param img
 * @param name
 * @param methodName
 */
let saveImage = function (img, name, methodName) {
  img.save(`./img/results/${name}#${methodName}.jpg`);
};

/**
 * Checks if array elements lie between the elements of two other arrays.
 * @param img
 * @returns {cv.Matrix}
 */
let inRange = function (img) {
  let temp_img = img.copy();
  let lower_threshold = [200, 200, 200];
  let upper_threshold = [255, 255, 255];

  temp_img.inRange(lower_threshold, upper_threshold);
  saveImage(temp_img, IMG_NAME_SPLITTED[0], 'inRange');
  return temp_img;
};

/**
 * Find leaf steam.
 * TODO: odčítanie 2 obrázkov od seba po dilate+erode
 * @param img
 * @returns {cv.Matrix}
 */
let findLeafStem = function (img) {
  var imgWithoutStem = img.copy();
  var imgBasic = img.copy();
  var imgModified = new cv.Matrix(imgWithoutStem.height(), imgWithoutStem.width());

  var size = imgWithoutStem.size()[0] / 50;
  var verticalStructure = cv.imgproc.getStructuringElement(1, [3, 3]);
  imgWithoutStem.dilate(45, verticalStructure);
  imgWithoutStem.erode(45, verticalStructure);

  // imgBasic.erode(1, verticalStructure);

  saveImage(imgWithoutStem, IMG_NAME_SPLITTED[0], 'findLeafStem_imgWithoutStem');

  // console.log(imgWithoutStem.width())
  // console.log(imgWithoutStem.height())
  // console.log(imgWithoutStem.getData());

  let diff = new cv.Matrix(imgWithoutStem.height(), imgWithoutStem.width());
  diff.absDiff(imgWithoutStem, imgBasic);
  diff.threshold(80, 255, 'Binary');
  diff.erode(10, verticalStructure);

  saveImage(diff, IMG_NAME_SPLITTED[0], 'findLeafStem_diff');


  // var mat = new cv.Matrix(256, 256, cv.Constants.CV_8UC3);
  // var buf = Buffer(256 * 256);
  // buf.fill(0);
  //
  //
  // for (var i = 0; i < 256 * 256; i++) {
  //   buf[i] = (i % 2 === 1) ? 230 : 0;
  // }
  //
  // mat.put(buf);
  // saveImage(mat, IMG_NAME_SPLITTED[0], 'findLeafStem_MAT');

  // var verticalStructure = cv.imgproc.getStructuringElement(1, [5, 5]);
  // imgDilate.dilate(3, verticalStructure);
  // //
  // imgDilate.save(`./img/${IMG_NAME_SPLITTED[0]}-2-dilate.${IMG_NAME_SPLITTED[1]}`);
  //
  // imgBasic.subtract(imgDilate);
  //
  // imgBasic.save(`./img/results/${IMG_NAME_SPLITTED[0]}-2-erode.${IMG_NAME_SPLITTED[1]}`);

  //return img;
};

/**
 * Return new rotated image without crop.
 * @param img
 * @param contours
 * @param index
 * @returns {cv.Matrix}
 */
let rotateWithoutCrop = function (img, contours, index) {
  let temp_img = img.copy();

  let rect = contours.minAreaRect(index);

  let diagonal = Math.round(Math.sqrt(Math.pow(temp_img.size()[1], 2) + Math.pow(temp_img.size()[0], 2)));
  let bgImg = new cv.Matrix(diagonal, diagonal, cv.Constants.CV_8UC3, [255, 255, 255]);
  let offsetX = (diagonal - temp_img.size()[1]) / 2;
  let offsetY = (diagonal - temp_img.size()[0]) / 2;

  console.log(rect.angle);

  temp_img.copyTo(bgImg, offsetX, offsetY);
  bgImg.rotate(rect.angle + 90);
  saveImage(bgImg, IMG_NAME_SPLITTED[0], 'rotateWithoutCrop');

  return bgImg;
};

// ----------------------------------------
// Initialize
if (process.argv.slice(2)[0] === undefined) {
  console.log('Error: missing file name.');
  return true;
}

const IMG_NAME = process.argv.slice(2)[0];
const IMG_NAME_SPLITTED = IMG_NAME.split('.');

cv.readImage(`./img/${IMG_NAME}`, function (err, img) {
  if (err) throw err;
  let width = img.width();
  let height = img.height();
  if (width < 1 || height < 1) throw new Error('Image has no size.');

  let allContoursImg = new cv.Matrix(height, width);
  let bigContoursImg = new cv.Matrix(height, width);

  let WORKING_IMG = img.copy();
  let CANNY_IMG;

  WORKING_IMG.convertGrayscale();
  WORKING_IMG.gaussianBlur([1, 1]);
  CANNY_IMG = WORKING_IMG.copy();


  // ----------------------------------------
  inRange(WORKING_IMG);
  findLeafStem(WORKING_IMG);


  // ----------------------------------------
  // Canny edge detector & print contours
  var lowThresh = 0;
  var highThresh = 100;
  var nIters = 1;

  CANNY_IMG.canny(lowThresh, highThresh);
  CANNY_IMG.dilate(nIters);

  let largestArea = 0;
  let largestAreaIndex = 0;
  let contours = CANNY_IMG.findContours();

  for (let i = 0; i < contours.size(); i++) {
    if (contours.area(i) > largestArea) {
      largestArea = contours.area(i);
      largestAreaIndex = i;
    }
  }

  allContoursImg.drawAllContours(contours, WHITE);
  bigContoursImg.drawContour(contours, largestAreaIndex, GREEN, THICKNESS, LINE_TYPE);

  // Draw rectangle around contour.
  let rect = contours.minAreaRect(largestAreaIndex);
  for (let i = 0; i < 4; i++) {
    bigContoursImg.line([rect.points[i].x, rect.points[i].y], [rect.points[(i + 1) % 4].x, rect.points[(i + 1) % 4].y], RED, 3);
  }

  saveImage(bigContoursImg, IMG_NAME_SPLITTED[0], 'drawBiggestContour');
  saveImage(allContoursImg, IMG_NAME_SPLITTED[0], 'allContours');


  //------------------------------------------------------------
  // Polygon approximation
  let polyApproxImage = new cv.Matrix(height, width);
  let arcLength = contours.arcLength(largestAreaIndex, true);
  contours.approxPolyDP(largestAreaIndex, 0.01 * arcLength, true);
  contours.approxPolyDP(largestAreaIndex, arcLength * 0.05, true);
  polyApproxImage.drawContour(contours, largestAreaIndex, RED, 2);
  saveImage(polyApproxImage, IMG_NAME_SPLITTED[0], 'approxPolyDP');


  //------------------------------------------------------------
  // Rotate image without cropping it
  let bgImg = rotateWithoutCrop(img, contours, largestAreaIndex);


  //------------------------------------------------------------
  // Canny edge detector on rotated image
  let diagonal = Math.round(Math.sqrt(Math.pow(img.size()[1], 2) + Math.pow(img.size()[0], 2)));
  let rotatedContour = new cv.Matrix(diagonal, diagonal);
  bgImg.canny(lowThresh, highThresh);
  bgImg.dilate(nIters);
  contours = bgImg.findContours();

  for (let i = 0; i < contours.size(); i++) {
    if (contours.area(i) > largestArea) {
      largestArea = contours.area(i);
      largestAreaIndex = i;
    }
  }

  rotatedContour.drawContour(contours, largestAreaIndex, GREEN, THICKNESS, LINE_TYPE);
  saveImage(rotatedContour, IMG_NAME_SPLITTED[0], 'rotateWithoutCropContour');
});