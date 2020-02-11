function img = readAndResizeImages(filename)

% read image
im = imread(filename);

img_gray= rgb2gray(im);% resize image

img = imresize(img_gray,[28 28]);

