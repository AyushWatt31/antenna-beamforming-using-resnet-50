% Step 1: Generate Beam Pattern Images + Real Complex Weights
% File: generate_beamforming_dataset.m

clear; clc; close all;

% ---------------- User parameters ----------------
outFolder    = 'BeamPatterns';  % output folder for PNGs
numSamples   = 4500;             % number of samples to generate
numElements  = 8;               % ULA elements
fc           = 3.5e9;           % carrier frequency (Hz)
c            = 3e8;             % speed of light (m/s)
lambda       = c/fc;            % wavelength (m)
d            = lambda/2;        % inter-element spacing (m) = 0.5λ
anglesGrid   = -90:1:90;        % pattern evaluation grid (deg)
imgSize      = [224 224];       % image size for ResNet-50
plotRangeDb  = 40;              % display range

% ---------------- Build ULA and steering grid ----------------
antArray = phased.ULA('NumElements', numElements, 'ElementSpacing', d);
angRad   = deg2rad(anglesGrid);
svGrid   = exp(1j * 2*pi*(d/lambda) * (0:numElements-1).' * sin(angRad)); % [E x A]

% ---------------- Allocate outputs ----------------
weightsComplex = complex(zeros(numSamples, numElements));
datasetDb      = zeros(numSamples, numel(anglesGrid));

% Make output folder
if ~exist(outFolder,'dir'), mkdir(outFolder); end

% ---------------- Generate samples ----------------
for n = 1:numSamples
    % Pick desired look angle (avoid endfire extremes for stability)
    theta = -80 + 160*rand(1,1); % deg
    
    % Steering vector at theta
    k  = 2*pi/lambda;                   % wavenumber
    sv = exp(1j * k * (0:numElements-1).' * sind(theta)); % [E x 1]
    
    % Conjugate weights, unit-norm
    w = conj(sv);
    w = w / norm(w + 1e-12);
    weightsComplex(n,:) = w.';          % store complex weights
    
    % Compute pattern (power dB) on grid
    resp = sum((w .* ones(1, numel(anglesGrid))) .* svGrid, 1); % [1 x A]
    power = abs(resp).^2;
    power = power / max(power + 1e-12);
    pat_db = 10*log10(power + 1e-12);
    datasetDb(n,:) = pat_db;
    
    % Convert to RGB image and save
    img = mat2gray(pat_db);                 % normalize to [0,1]
    img = imresize(img, imgSize);           % 224x224
    img_rgb = repmat(img, [1 1 3]);         % grayscale -> RGB
    imwrite(img_rgb, fullfile(outFolder, sprintf('pattern_%04d.png', n)));
end

% ---------------- Save targets and metadata ----------------
save('beam_targets.mat','weightsComplex','numElements','fc','lambda','d','anglesGrid');

disp('Step 1 complete: BeamPatterns/*.png and beam_targets.mat created.');