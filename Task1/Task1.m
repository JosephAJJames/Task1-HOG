%Each block 




function extractHOGFeatures(inputImage)

    %function to calculate the gradient magnitude and angle of a 8x8 cell
    function [magnitued, angle] = calculateGradientAndAngle(cell)

        %defining the sobel filters
        sobel_vert = [-1 -2 -1; 0 0 0; 1 2 1]; %filter to be used along y axis
        sobel_horz = [-1 0 1; -2 0 2; -1 0 1]; %filter to be used along x axis

        %applying filters to cell
        gx = imfilter(cell, sobel_horz, "conv", "same");
        gy = imfilter(cell, sobel_vert, "conv", "same");

        %calculating the 
        magnitued = sqrt(gx.^2 + gy.^2);
        angle = rad2deg(atan2(gy, gx));

        magnitued = magnitued / max(magnitued(:));

        angle = mod(angle + 180, 180);
    end




    I_cell = [
    0   32  64  96  128 160 192 224;
    0   32  64  96  128 160 192 224;
    0   32  64  96  128 160 192 224;
    0   32  64  96  128 160 192 224;
    0   32  64  96  128 160 192 224;
    0   32  64  96  128 160 192 224;
    0   32  64  96  128 160 192 224;
    0   32  64  96  128 160 192 224
    ];


    [magnitued, angle] = calculateGradientAndAngle(I_cell);



    disp(magnitued);
    disp(angle);

end


inputImage = imread("cameraman.tif");
inputImage = im2double(inputImage);

% Load the input image

% Ensure the image is grayscale
[rows, cols, colours] = size(inputImage);
if colours > 1 %if the image has more than 1 3rd dimenison(colour channel)
    inputImage = mean(inputImage, 3); %take the mean of the 3rd dimension
end


% Ensure image dimensions are divisible by 8, otherwise Resize the image
new_rows = ceil(rows/8) * 8;
new_cols = ceil(cols/8) * 8;

if new_rows ~= rows || new_cols ~= cols
    inputImage = imresize(inputImage, [new_cols, new_rows]);
end




% Call the HOG feature extraction function
extractHOGFeatures(inputImage);
% Display the length of the extracted features
%disp(['Length of extracted HOG features: ',num2str(length(extractedHOGFeatures))]);
