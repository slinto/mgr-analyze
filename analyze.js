/**
 * node analyze leaf_1.jpg
 */
var cv = require('opencv');

var GREEN = [0, 255, 0]; // B, G, R
var WHITE = [255, 255, 255]; // B, G, R
var RED = [0, 0, 255]; // B, G, R


let cannyEdge = function(img) {
  let cannyImg = img.copy();
};

//

if (process.argv.slice(2)[0] === undefined) {
  console.log('Error: chyba nazov suboru.');
  return true;
}

const IMG_NAME = process.argv.slice(2)[0];
const IMG_NAME_SPLITTED = IMG_NAME.split('.');

cv.readImage(`./img/${IMG_NAME}`, function (err, im) {
  if (err) throw err;
  var width = im.width();
  var height = im.height();
  if (width < 1 || height < 1) throw new Error('Image has no size');

  var big = new cv.Matrix(height, width);
  var poly = new cv.Matrix(height, width);
  var all = new cv.Matrix(height, width);

  let IMG_ORIGINAL = im.copy();

  let WORKING_IMG = im.copy();

  WORKING_IMG.convertGrayscale();
  // Pozn: horsie vysledky pri canny edge
  //im.gaussianBlur([5, 5], 0);
  im_canny = WORKING_IMG.copy();

  // im.houghLinesP();
  // im.save(`./img/${IMG_NAME_SPLITTED[0]}-1.5-inRange.${IMG_NAME_SPLITTED[1]}`);

  // ----------------------------------------
  // IN_RANGE 
  // B,G,R 
  var lower_threshold = [200, 200, 200];
  var upper_threshold = [255, 255, 255];
  WORKING_IMG.inRange(lower_threshold, upper_threshold);
  // im.save(`./img/${IMG_NAME_SPLITTED[0]}-1.5-inRange.${IMG_NAME_SPLITTED[1]}`);

  // ----------------------------------------
  // Vyhladavanie stonky 
  var imgDilate = WORKING_IMG.copy();
  var imgBasic = WORKING_IMG.copy();
  var imgModified;

  var size = imgDilate.size()[0] / 40;
  size = 110;
  var verticalStructure = cv.imgproc.getStructuringElement(1, [size, size]);

  imgDilate.dilate(1, verticalStructure);
  imgDilate.erode(1, verticalStructure);

  // TODO: odčítanie 2 obrázkov od seba (dilate)

  // imgDilate.save(`./img/${IMG_NAME_SPLITTED[0]}-2-erode.${IMG_NAME_SPLITTED[1]}`);
  //
  // imgBasic.subtract(imgDilate);
  //
  // imgBasic.save(`./img/${IMG_NAME_SPLITTED[0]}-2-erode.${IMG_NAME_SPLITTED[1]}`);


  // ----------------------------------------
  // CANNY EDGE
  var lowThresh = 0;
  var highThresh = 100;
  var nIters = 4;
  var maxArea = 2500;

  im_canny.canny(lowThresh, highThresh);
  im_canny.dilate(nIters);

  contours = im_canny.findContours();
  const lineType = 8;
  const maxLevel = 0;
  const thickness = 3;
  let largestArea = 0;
  let largestAreaIndex = 0;

  for (let i = 0; i < contours.size(); i++) {
    if (contours.area(i) > largestArea) {
      largestArea = contours.area(i);
      largestAreaIndex = i;
    }
  }

  // Otočenie obrázku podla polohy obdlznika!
  let rect = contours.minAreaRect(largestAreaIndex);
  let diagonal = Math.round(Math.sqrt(Math.pow(im.size()[1], 2) + Math.pow(im.size()[0], 2)));

  cv.readImage(`./img/leaf_bg.jpg`, function (err, bgImg) {
    bgImg.resize(diagonal, diagonal);

    let offsetX = (diagonal - im.size()[1]) / 2;
    let offsetY = (diagonal - im.size()[0]) / 2;

    IMG_ORIGINAL.copyTo(bgImg, offsetX, offsetY);
    bgImg.rotate(rect.angle + 90);

    bgImg.save(`./img/${IMG_NAME_SPLITTED[0]}-X-rotated.${IMG_NAME_SPLITTED[1]}`);

    var rotatedContour = new cv.Matrix(diagonal, diagonal);
    let rotatedCanny = bgImg.copy();
    rotatedCanny.canny(lowThresh, highThresh);
    rotatedCanny.dilate(nIters);

    contours = rotatedCanny.findContours();
    const lineType = 8;
    const maxLevel = 0;
    const thickness = 3;
    let largestArea = 0;
    let largestAreaIndex = 0;

    for (let i = 0; i < contours.size(); i++) {
      if (contours.area(i) > largestArea) {
        largestArea = contours.area(i);
        largestAreaIndex = i;
      }
    }
    var moments = contours.moments(largestAreaIndex);
    var cgx = Math.round(moments.m10 / moments.m00);
    var cgy = Math.round(moments.m01 / moments.m00);

    rotatedContour.drawContour(contours, largestAreaIndex, GREEN, thickness, lineType, maxLevel, [0, 0]);
    rotatedContour.line([cgx - 5, cgy], [cgx + 5, cgy], RED);
    rotatedContour.line([cgx, cgy - 5], [cgx, cgy + 5], RED);
    // rotatedContour.rotate(rect.angle + 90);
    rotatedContour.save(`./img/${IMG_NAME_SPLITTED[0]}-X-rotatedContour.${IMG_NAME_SPLITTED[1]}`);


  });

  // DRAW BIGGEST CONTOUR
  var moments = contours.moments(largestAreaIndex);
  var cgx = Math.round(moments.m10 / moments.m00);
  var cgy = Math.round(moments.m01 / moments.m00);
  big.drawContour(contours, largestAreaIndex, GREEN, thickness, lineType, maxLevel, [0, 0]);
  big.line([cgx - 5, cgy], [cgx + 5, cgy], RED);
  big.line([cgx, cgy - 5], [cgx, cgy + 5], RED);

  // APPROX POLYGON
  let arcLength = contours.arcLength(largestAreaIndex, true);
  // contours.approxPolyDP(largestAreaIndex, 0.01 * arcLength, true);
  // contours.approxPolyDP(largestAreaIndex, arcLength * 0.05, true);

  // RECTANGLE around contour
  var bound = contours.boundingRect(largestAreaIndex);
  big.rectangle([bound.x, bound.y], [bound.width, bound.height], WHITE, 2);


  poly.drawContour(contours, largestAreaIndex, RED);


  // FINAL PRINT
  all.drawAllContours(contours, WHITE);

  big.save(`./img/${IMG_NAME_SPLITTED[0]}-X-final.${IMG_NAME_SPLITTED[1]}`);
  poly.save(`./img/${IMG_NAME_SPLITTED[0]}-X-poly.${IMG_NAME_SPLITTED[1]}`);
  //all.save(`./img/ALL_2${IMG_NAME}`);   console.log(`Img ${IMG_NAME} saved.`);
});