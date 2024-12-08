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
% Call the HOG feature extraction function
%extractedHOGFeatures = extractHOGFeatures(inputImage);
% Display the length of the extracted features
%disp(['Length of extracted HOG features: ',num2str(length(extractedHOGFeatures))]);