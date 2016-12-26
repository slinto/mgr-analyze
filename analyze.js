var cv = require('opencv');

var GREEN = [0, 255, 0]; // B, G, R
var WHITE = [255, 255, 255]; // B, G, R
var RED = [0, 0, 255]; // B, G, R

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
  var all = new cv.Matrix(height, width);

  im.convertGrayscale();
  // Pozn: horsie vysledky pri canny edge
  //im.gaussianBlur([5, 5], 0);
  im_canny = im.copy();

  // ----------------------------------------
  // IN_RANGE 
  // B,G,R 
  //  var lower_threshold = [200, 200, 200]; 
  //  var upper_threshold = [255, 255, 255]; 
  // im.inRange(lower_threshold, upper_threshold); 
  // im.save(`./img/${IMG_NAME_SPLITTED[0]}-1.5-inRange.${IMG_NAME_SPLITTED[1]}`);



  // ----------------------------------------
  // CANNY EDGE
  var lowThresh = 0;
  var highThresh = 100;
  var nIters = 2;
  var maxArea = 2500;

  im_canny.canny(lowThresh, highThresh);
  im_canny.dilate(nIters);

  contours = im_canny.findContours();
  const lineType = 8;
  const maxLevel = 0;
  const thickness = 3;

  for (i = 0; i < contours.size(); i++) {
    if (contours.area(i) > maxArea) {
      var moments = contours.moments(i);
      var cgx = Math.round(moments.m10 / moments.m00);
      var cgy = Math.round(moments.m01 / moments.m00);
      big.drawContour(contours, i, GREEN, thickness, lineType, maxLevel, [0, 0]);
      big.line([cgx - 5, cgy], [cgx + 5, cgy], RED);
      big.line([cgx, cgy - 5], [cgx, cgy + 5], RED);
    }
  }

  all.drawAllContours(contours, WHITE);

  big.save(`./img/${IMG_NAME_SPLITTED[0]}-X-final.${IMG_NAME_SPLITTED[1]}`);
  //all.save(`./img/ALL_2${IMG_NAME}`);   console.log(`Img ${IMG_NAME} saved.`);
});