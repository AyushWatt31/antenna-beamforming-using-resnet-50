% Step 3: Train the Modified ResNet-50 for Beamforming Regression
% File: train_beamforming_network.m

clear; clc; close all;

% ---------------- Load modified network and targets ----------------
load('modifiedResNet50.mat', 'lgraph', 'numElements', 'numOutputs');
load('beam_targets.mat', 'weightsComplex');     % [numSamples x numElements] complex

% ---------------- Images datastore ----------------
datasetFolder = 'BeamPatterns';
imds = imageDatastore(datasetFolder, 'FileExtensions', '.png');
numSamples = numel(imds.Files);

% Safety check: targets must match image count
if size(weightsComplex,1) ~= numSamples
    error('Mismatch: weightsComplex rows (%d) ~= number of images (%d).', ...
           size(weightsComplex,1), numSamples);
end

% ---------------- Target preprocessing ----------------
% Unit-norm per sample for stable scale
mag = sqrt(sum(abs(weightsComplex).^2, 2));
mag(mag==0) = 1;
weightsComplex = weightsComplex ./ mag;

% Convert complex matrix to [Re1 Im1 Re2 Im2 ...]
targets = zeros(numSamples, 2*numElements);
for e = 1:numElements
    targets(:, 2*e-1) = real(weightsComplex(:, e));
    targets(:, 2*e)   = imag(weightsComplex(:, e));
end

% ---------------- Train/validation split (reproducible) ----------------
rng(42);
trainRatio = 0.8;
numTrain   = floor(trainRatio * numSamples);
idx        = randperm(numSamples);
idxTrain   = idx(1:numTrain);
idxVal     = idx(numTrain+1:end);

imdsTrain  = subset(imds, idxTrain);
imdsVal    = subset(imds, idxVal);
Ytrain     = targets(idxTrain, :);
Yval       = targets(idxVal,   :);

% ---------------- Build paired datastores ----------------
adsTrain = arrayDatastore(Ytrain, 'IterationDimension', 1);
adsVal   = arrayDatastore(Yval,   'IterationDimension', 1);
cdsTrain = combine(imdsTrain, adsTrain);
cdsVal   = combine(imdsVal,   adsVal);

% ---------------- Preprocessing and augmentation ----------------
inputSize = [224 224 3];
imageAug = imageDataAugmenter('RandXReflection', true, ...
                              'RandRotation', [-5 5]); % small rotation only
augTrain = transform(cdsTrain, @(d) preprocessPair(d, inputSize, imageAug));
augVal   = transform(cdsVal,   @(d) preprocessPair(d, inputSize, []));

% ---------------- Training options ----------------
miniBatch = 32;                                    % larger batch for 1000 samples
itersPerEpoch = ceil(numTrain / miniBatch);
targetIters   = 900;                                % increase for larger data
maxEpochs     = ceil(targetIters / itersPerEpoch);  % compute from target iterations

options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatch, ...
    'MaxEpochs', maxEpochs, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', max(1, itersPerEpoch), ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% ---------------- Train ----------------
netTransfer = trainNetwork(augTrain, lgraph, options);

% ---------------- Save outputs ----------------
save('trainedBeamformingNet.mat', 'netTransfer');
save('training_metadata.mat','idxTrain','idxVal','options','miniBatch','itersPerEpoch','maxEpochs');

disp('Step 3 complete: trainedBeamformingNet.mat and training_metadata.mat saved.');

% ---------------- Utility: preprocessing function ----------------
function out = preprocessPair(d, inputSize, augmenter)
    img = d{1};
    if size(img,3) == 1
        img = repmat(img,1,1,3);
    end
    img = imresize(img, inputSize(1:2));
    if ~isempty(augmenter)
        img = augment(augmenter, img);
    end
    out = {img, d{2}};
end