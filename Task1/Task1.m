function extractedHOGFeatures = extractHOGFeatures(inputImage)

    extractedHOGFeatures = [];


    %function to calculate the gradient magnitude and angle of a 8x8 cell
    function [magnitued, angle] = apply_sobel_filter(cell)

        %defining the sobel filters
        sobel_vert = [-1 -2 -1; 0 0 0; 1 2 1]; %filter to be used along y axis
        sobel_horz = [-1 0 1; -2 0 2; -1 0 1]; %filter to be used along x axis

        %applying filters to cell
        gx = imfilter(cell, sobel_horz, "symmetric");
        gy = imfilter(cell, sobel_vert, "symmetric");

        %calculating the magnitudes and angles
        magnitued = sqrt(gx.^2 + gy.^2);
        angle = rad2deg(atan2(gy, gx));

        %normalise magnitudes
        magnitued = magnitued / max(magnitued(:));

        %convert negative angles to positive
        angle = mod(angle + 180, 180); 
    end


    function blocks = extract_blocks(inputImage, block_size)

        stride_horizontal = 1;
        stride_vertical = 1;
        [rows, cols] = size(inputImage);

        % Compute the number of blocks based on horizontal stride
        numBlocksX = floor((cols - block_size) / stride_horizontal) + 1;
        numBlocksY = floor((rows - block_size) / stride_vertical) + 1;
    
        % Preallocate the blocks matrix
        blocks = zeros(block_size^2, numBlocksX * numBlocksY);
    
        % Index for blocks
        blockIdx = 1;
    
        for i = 1:stride_vertical:(rows - block_size + 1)
            for j = 1:stride_horizontal:(cols - block_size + 1)
                % Extract the block
                block = inputImage(i:i+block_size-1, j:j+block_size-1);
                
                % Flatten the block and store it
                blocks(:, blockIdx) = block(:);
                
                % Increment block index
                blockIdx = blockIdx + 1;
            end
        end

    end


    function [cell_1, cell_2, cell_3, cell_4] = split_block_into_cells(block)
        [blockHeight, blockWidth] = size(block);

        % Compute the size of each cell
        cellHeight = blockHeight / 2;
        cellWidth = blockWidth / 2;

        % split the block into four cells
        cell_1 = block(1:cellHeight, 1:cellWidth);             % Top-left
        cell_2 = block(1:cellHeight, cellWidth+1:end);         % Top-right
        cell_3 = block(cellHeight+1:end, 1:cellWidth);         % Bottom-left
        cell_4 = block(cellHeight+1:end, cellWidth+1:end);     % Bottom-right
    end

    function histogram = computre_cell_gradient_histogram(magnitude, angle)
        binEdges = linspace(0, 180, 10); 
        histogram = zeros(1, 9); 
    
        
        magnitude = magnitude(:);
        angle = angle(:);

        
        for b = 1:9
            inBin = (angle >= binEdges(b)) & (angle < binEdges(b + 1));

            histogram(b) = sum(magnitude(inBin));
        end
    end


    %function call to extract the blocks (with horizontal stride of 1)
    blocks = extract_blocks(inputImage, 16); %retuns a matrix of flatternd blocks

    size(blocks)

    for i = 1:size(blocks, 2)

        [cell_1 , cell_2, cell_3, cell_4] = split_block_into_cells(reshape(blocks(:, i), [16, 16]));
 
        %computre magnitueds and angles for cells
        [magnitude_1, angle_1] = apply_sobel_filter(cell_1); %magnitudes and angles for top left cell of current block
        [magnitude_2, angle_2] = apply_sobel_filter(cell_2); %magnitudes and angles for top right cell of current block
        [magnitude_3, angle_3] = apply_sobel_filter(cell_3); %magnitudes and angles for bottom left cell of current block
        [magnitude_4, angle_4] = apply_sobel_filter(cell_4); %magnitudes and angles for bottom right cell of current block

        %compute histogram for each cell
        [histogram_1] = computre_cell_gradient_histogram(magnitude_1, angle_1); %compute histogram for first cell
        [histogram_2] = computre_cell_gradient_histogram(magnitude_2, angle_2); %computre histogram for second cell
        [histogram_3] = computre_cell_gradient_histogram(magnitude_3, angle_3); %computte histogram for second cell
        [histogram_4] = computre_cell_gradient_histogram(magnitude_4, angle_4); %compute histogram for third cell

        
        block_histogram = [histogram_1, histogram_2, histogram_3, histogram_4]; %concatante the 4 histograms together

        extractedHOGFeatures = [extractedHOGFeatures, block_histogram];

    end
end


inputImage = imread("cameraman.tif");
inputImage = im2double(inputImage);

% Load the input image

% Ensure the image is grayscale
[rows, cols, colours] = size(inputImage);
if colours > 1 %if the image has more than 1 3rd dimenison(colour channel)
    inputImage = rgb2gray(inputImage);
end

% Ensure image dimensions are divisible by 8, otherwise Resize the image
new_rows = ceil(rows/8) * 8;
new_cols = ceil(cols/8) * 8;

if new_rows ~= rows || new_cols ~= cols
    inputImage = imresize(inputImage, [new_cols, new_rows]);
end




% Call the HOG feature extraction function
extractedHOGFeatures = extractHOGFeatures(inputImage);
disp("bosh")
% Display the length of the extracted features
disp(['Length of extracted HOG features: ',num2str(length(extractedHOGFeatures))]);
