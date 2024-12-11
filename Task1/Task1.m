function extractedHOGFeatures = extractHOGFeatures(inputImage)
    %defining the extractedHOGFeatures to concatante to later
    extractedHOGFeatures = [];

%defining helper functions

function [magnitued, angle] = apply_sobel_filter(cell)

    %defining the sobel filters
    sobel_vert = [-1 -2 -1; 0 0 0; 1 2 1]; %filter to be used along y axis
    sobel_horz = [-1 0 1; -2 0 2; -1 0 1]; %filter to be used along x axis

    %applying filters to cell
    gx = imfilter(cell, sobel_horz, "symmetric"); %apply the horiontal filter to the image
    gy = imfilter(cell, sobel_vert, "symmetric"); %apply the horizontal filter to the image

    %calculating the magnitudes and angles
    magnitued = sqrt(gx.^2 + gy.^2);
    angle = rad2deg(atan2(gy, gx));
    %convert negative angles to positive
    angle = mod(angle + 180, 180); %will turn any negative angles into their positive equivalaent

    %normalise magnitudes
    magnitued = magnitued / max(magnitued(:)); %normalise the magnitueds
    end


    function blocks = extract_blocks(inputImage)
        %defining block size
        block_size = 16; %block size is 16 as blocks are 2x2 cells

        %extracting dimensions of the image
        [rows, cols] = size(inputImage); %extracting height and width of the inputImage

        %computing the number of blocks in the imag
        num_x = floor((cols - block_size) / 8) + 1; %calculating the horizionatal blocks
        num_y = floor((rows - block_size) / 8) + 1; %calculating the vertical blocks
    
        %preallocate block matrix
        blocks = zeros(block_size^2, num_x * num_y); %number of rows is ^2 of block size as they are going to be flatternd, number of columns is number of blocks in the image
    
        %used to track which block we're considering, starting from first
        block_num = 1;
    
        %loops to extract each block and place it in the, stride of 8 as size of 1 cell is 8
        for i = 1:8:(rows - block_size + 1)
            for j = 1:8:(cols - block_size + 1)
                % extract the block
                block = inputImage(i:i+block_size-1, j:j+block_size-1);
                
                % store the block in its own column
                blocks(:, block_num) = block(:);
                
                %increase block index as we're moving onto the next one
                block_num = block_num + 1;
            end
        end

    end

    %helper function to split a block down into its cells
    function [cell_1, cell_2, cell_3, cell_4] = split_block_into_cells(block)
        %
        [block_height, block_width] = size(block);

        %calculate the size of each cell
        cell_height = block_height / 2; % halfing the height of a block to get the cell height
        cell_width = block_width / 2; % halfing the width of the block to get the cell width

        % split the block into four cells
        cell_1 = block(1:cell_height, 1:cell_width); % top left
        cell_2 = block(1:cell_height, cell_width+1:end); % top right
        cell_3 = block(cell_height+1:end, 1:cell_width); % bottom left
        cell_4 = block(cell_height+1:end, cell_width+1:end); % bottom right
    end

    %helper function to create the histogram for a partiular cell
    function histogram = computre_cell_gradient_histogram(cell)
        [magnitude, angle] = apply_sobel_filter(cell); % calculate the magnitude and orientation of the current cell

        binEdges = linspace(0, 180, 10); %define our bins
        histogram = zeros(1, 9);  %pre-allocate the histogram
    
        %prepare the magnitudes and orientations for binning
        magnitude = magnitude(:); %falltern the magnitude
        angle = angle(:); %falltern the magnitude

        %put the data in its corrosponding bin
        for b = 1:9 %loop over each bin
            inBin = (angle >= binEdges(b)) & (angle < binEdges(b + 1)); %create a logical array of angles that fit into the current bin

            histogram(b) = sum(magnitude(inBin)); %get the sum of the magnitudes that would fit into this angle bin
        end
    end

    %function call to extract the blocks (with horizontal stride of 1)
    blocks = extract_blocks(inputImage); %retuns a matrix of flatternd blocks

    %loops over each block
    for i = 1:size(blocks, 2) %from the first block to the last block

        %function call to split the current block down into cells
        [cell_1 , cell_2, cell_3, cell_4] = split_block_into_cells(reshape(blocks(:, i), [16, 16])); %reshape the column back into a 16x16 block

        %compute histogram for each cell
        [histogram_1] = computre_cell_gradient_histogram(cell_1); %compute histogram for first cell
        [histogram_2] = computre_cell_gradient_histogram(cell_2); %computre histogram for second cell
        [histogram_3] = computre_cell_gradient_histogram(cell_3); %computte histogram for second cell
        [histogram_4] = computre_cell_gradient_histogram(cell_4); %compute histogram for third cell

        block_histogram = [histogram_1, histogram_2, histogram_3, histogram_4]; %concatante the 4 histograms together
        l2_value = sqrt(sum(block_histogram.^2)); %calcualte the l2 value of the histogram

        block_histogram = block_histogram / l2_value; %normalise the histogram

        extractedHOGFeatures = [extractedHOGFeatures, block_histogram]; %concatonate the new feature vector with the rest of them

    end
end

%loading and converting the image
inputImage = imread("cameraman.tif"); %reading in my image
inputImage = im2double(inputImage); %converting it to double



%gray scale check
[rows, cols, colours] = size(inputImage); %get dimensions of image matrix
if colours > 1 %if the image has more than 1 3rd dimenison(colour channel)
    inputImage = rgb2gray(inputImage); %convert inputImage to grayscale
end

%calculating new image dimensions, which are a factor of 8.
new_rows = ceil(rows/8) * 8;
new_cols = ceil(cols/8) * 8;

%checks to see if the images dimensions are different from out new, divisible by 8, dimenions
if new_rows ~= rows || new_cols ~= cols
    inputImage = imresize(inputImage, [new_cols, new_rows]); %resize image with the new factor of 8 dimensions
end




% Call the HOG feature extraction function
extractedHOGFeatures = extractHOGFeatures(inputImage);
% Display the length of the extracted features
disp(['Length of extracted HOG features: ',num2str(length(extractedHOGFeatures))]);
