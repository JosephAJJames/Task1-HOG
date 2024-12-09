function extractedHOGFeatures = extractHOGFeatures (inputImage)
% Your code here
end


inputImage = imread("cameraman.tif");

% Load the input image

% Ensure the image is grayscale
[rows, cols, colours] = size(inputImage)
if colours > 1
    inputImage = mean(inputImage, 3);
end


% Ensure image dimensions are divisible by 8, otherwise Resize the image
new_rows = ceil(rows/8) * 8;
new_cols = ceil(cols/8) * 8;

if newHeight ~= height || newWidth ~= width
    inputImage = imresize(inputImage, [newHeight, newWidth]);
end




% Call the HOG feature extraction function
%extractedHOGFeatures = extractHOGFeatures(inputImage);
% Display the length of the extracted features
%disp(['Length of extracted HOG features: ',num2str(length(extractedHOGFeatures))]);