clear; clc; close all;

% --- Complex channel matrix input (a+bi only) ---
Nr = input('Enter number of rows (Nr): ');
Nt = input('Enter number of columns (Nt): ');
H  = complex(zeros(Nr, Nt));

disp('Enter each row using MATLAB complex syntax (a+bi), e.g.:  0.8-0.2i  0.4+0.1i  ...');
for r = 1:Nr
    rowStr = input(sprintf('Row %d: ', r), 's');
    vals   = str2num(rowStr); %#ok<ST2NM>  % parses complex tokens like 0.3+0.4i
    if numel(vals) ~= Nt || ~isnumeric(vals)
        error('Row %d: expected exactly %d complex elements like a+bi.', r, Nt);
    end
    H(r,:) = vals;
end

% --- Grayscale encoding (magnitude) ---
Mag   = abs(H);           % non-negative magnitudes
gray  = mat2gray(Mag);    % scale to [0,1] for imaging
gray224 = imresize(gray, [224 224]);  % ResNet size if needed later

% --- Display and save as true grayscale (single-channel) ---
figure; imshow(gray224, []); title('Grayscale (|H|)');

imwrite(gray224, 'user_input_resnet_ready_gray.png');  % single-channel grayscale PNG
save('user_input_H.mat', 'H', 'gray224', 'Mag');

fprintf('Saved: user_input_resnet_ready_gray.png and user_input_H.mat\n');

% If a 3-channel tensor is required at inference time:
% imgForCNN = repmat(gray224, [1 1 3]);