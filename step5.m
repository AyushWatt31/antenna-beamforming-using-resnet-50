% Step 5: Deployment Inference and Demo
% File: deploy_beamforming_inference.m

clear; clc; close all;

% ---------------- Load trained model and metadata ----------------
load('trainedBeamformingNet.mat','netTransfer');
load('modifiedResNet50.mat','numElements');                 % from Step 2
load('beam_targets.mat','lambda','d','anglesGrid');         % from Step 1

% ---------------- Config ----------------
inputSize = [224 224 3];
dataFolder = 'BeamPatterns';           % folder with PNGs
outFolder  = 'Step5_Results';
if ~exist(outFolder,'dir'), mkdir(outFolder); end

% Precompute steering matrix for reconstruction
angRad = deg2rad(anglesGrid);
svGrid = exp(1j * 2*pi*(d/lambda) * (0:numElements-1).' * sin(angRad)); % [E x A]

% ---------------- Helper: single-image inference ----------------
inferFcn = @(imgPath) infer_beam_weights(imgPath, netTransfer, numElements, inputSize);

% ---------------- A) Single-image example ----------------
% Pick one image to test (change filename as needed)
samplePath = fullfile(dataFolder, 'pattern_0001.png');
if exist(samplePath,'file')
    w_pred = inferFcn(samplePath);
    [pat_db] = weights2pattern(w_pred, svGrid, anglesGrid);
    figure; plot(anglesGrid, pat_db, 'LineWidth',1.8); grid on;
    xlabel('Angle (deg)'); ylabel('Power (dB)');
    title(sprintf('Predicted Pattern (single image: %s)', 'pattern_0001.png'));
    saveas(gcf, fullfile(outFolder,'single_inference_pattern.png'));
end

% ---------------- B) Batch inference on N random images ----------------
files = dir(fullfile(dataFolder,'pattern_*.png'));
N = min(10, numel(files));                        % number to visualize
idxPick = randperm(numel(files), N);

results = struct('filename',[],'weights',[],'patternDb',[]);
for k = 1:N
    fname = files(idxPick(k)).name;
    fpath = fullfile(dataFolder, fname);

    % Inference
    w_pred = inferFcn(fpath);

    % Pattern reconstruction
    pat_db = weights2pattern(w_pred, svGrid, anglesGrid);

    % Save plot
    figure; plot(anglesGrid, pat_db, 'LineWidth',1.8); grid on;
    xlabel('Angle (deg)'); ylabel('Power (dB)');
    title(sprintf('Predicted Pattern: %s', fname));
    saveas(gcf, fullfile(outFolder, sprintf('pred_pattern_%s.png', fname)));

    % Store
    results(k).filename  = fname;
    results(k).weights   = w_pred;
    results(k).patternDb = pat_db;
end

% ---------------- C) Export results (optional) ----------------
% Save MATLAB struct
save(fullfile(outFolder,'inference_results.mat'),'results','anglesGrid');

% Save CSV summary of first N patterns (average level and main-lobe angle)
T = table('Size',[N 3],'VariableTypes',{'string','double','double'}, ...
          'VariableNames',{'filename','meanLevelDb','mainLobeDeg'});
for k = 1:N
    [~,idxMax] = max(results(k).patternDb);
    T.filename(k)   = string(results(k).filename);
    T.meanLevelDb(k)= mean(results(k).patternDb);
    T.mainLobeDeg(k)= anglesGrid(idxMax);
end
writetable(T, fullfile(outFolder,'summary.csv'));

disp('Step 5 complete: inference plots and summary saved in Step5_Results.');

% ==================== Local functions ====================

function w_pred = infer_beam_weights(imgPath, net, numElements, inputSize)
    % Load and preprocess image to 224x224x3
    im = imread(imgPath);
    if size(im,3)==1, im = repmat(im,1,1,3); end
    im = imresize(im, inputSize(1:2));
    % Predict vector [Re1 Im1 ...]
    y = predict(net, im);
    % Convert to complex weights and unit-normalize
    w_pred = zeros(1,numElements);
    for e = 1:numElements
        w_pred(e) = complex(y(2*e-1), y(2*e));
    end
    w_pred = w_pred / (norm(w_pred) + 1e-12);
end

function pat_db = weights2pattern(w, svGrid, anglesGrid)
    % Reconstruct dB beam pattern from complex weights and steering matrix
    w = w(:);
    resp = sum((w .* ones(1,numel(anglesGrid))) .* svGrid, 1);
    power = abs(resp).^2;
    power = power / max(power + 1e-12);
    pat_db = 10*log10(power + 1e-12);
end