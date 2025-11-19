% Step 4: Evaluate Trained Beamforming Regressor
% File: evaluate_beamforming_model.m

clear; clc; close all;

% ---------------- Load artifacts ----------------
load('trainedBeamformingNet.mat','netTransfer');
load('training_metadata.mat','idxTrain','idxVal','options'); %#ok<NASGU>
load('beam_targets.mat','weightsComplex','numElements','lambda','d','anglesGrid');

datasetFolder = 'BeamPatterns';
imdsAll = imageDatastore(datasetFolder, 'FileExtensions', '.png');

% Build validation datastore from saved indices
imdsVal = subset(imdsAll, idxVal);

% Make sure targets align with images
numSamples = numel(imdsAll.Files);
if size(weightsComplex,1) ~= numSamples
    error('Mismatch: weightsComplex rows (%d) ~= images (%d).', size(weightsComplex,1), numSamples);
end

% Unit-norm per sample (same as training)
mag = sqrt(sum(abs(weightsComplex).^2, 2)); mag(mag==0)=1;
weightsComplex = weightsComplex ./ mag;

% Targets for validation in [Re1 Im1 ...]
Yall = zeros(numSamples, 2*numElements);
for e = 1:numElements
    Yall(:, 2*e-1) = real(weightsComplex(:, e));
    Yall(:, 2*e)   = imag(weightsComplex(:, e));
end
Ytrue = Yall(idxVal,:);

% ---------------- Build evaluation pipeline ----------------
inputSize = [224 224 3];
augVal = augmentedImageDatastore(inputSize, imdsVal); % no augmentation

% ---------------- Predict ----------------
Ypred = predict(netTransfer, augVal);

% ---------------- Metrics ----------------
rmse_per_output = sqrt(mean((Ypred - Ytrue).^2, 1));
rmse_overall    = sqrt(mean((Ypred - Ytrue).^2, 'all'));
mae_per_output  = mean(abs(Ypred - Ytrue), 1);
mae_overall     = mean(abs(Ypred - Ytrue), 'all');

disp('Overall metrics:');
disp(table(rmse_overall, mae_overall));

figure; 
subplot(2,1,1); bar(rmse_per_output); grid on;
title('Per-output RMSE'); xlabel('Output index'); ylabel('RMSE');
subplot(2,1,2); bar(mae_per_output); grid on;
title('Per-output MAE'); xlabel('Output index'); ylabel('MAE');

% ---------------- Visualize some samples (weights) ----------------
numShow = min(3, size(Ytrue,1));
idxShowLocal = randperm(size(Ytrue,1), numShow);

for k = 1:numShow
    i = idxShowLocal(k);
    figure;
    bar([Ytrue(i,:); Ypred(i,:)]'); grid on;
    legend({'True','Pred'}, 'Location','best');
    title(sprintf('Weights: True vs Pred (val sample %d)', i));
    xlabel('Weight index'); ylabel('Value');
end

% ---------------- Beam pattern comparison ----------------
% Reconstruct beam patterns for a few samples using array model
% Define steering matrix across anglesGrid (same as Step 1)
angRad = deg2rad(anglesGrid);
svGrid = exp(1j * 2*pi*(d/lambda) * (0:numElements-1).' * sin(angRad));

for k = 1:numShow
    i = idxShowLocal(k);

    % Convert predicted [Re Im ...] back to complex weights
    w_pred = zeros(1,numElements);
    w_true = zeros(1,numElements);
    for e = 1:numElements
        w_true(e) = complex(Ytrue(i,2*e-1), Ytrue(i,2*e));
        w_pred(e) = complex(Ypred(i,2*e-1), Ypred(i,2*e));
    end

    % Normalize to unit norm before pattern (to compare shapes)
    w_true = w_true / (norm(w_true) + 1e-12);
    w_pred = w_pred / (norm(w_pred) + 1e-12);

    % Compute patterns
    resp_true = sum((w_true(:) .* ones(1,numel(anglesGrid))) .* svGrid, 1);
    resp_pred = sum((w_pred(:) .* ones(1,numel(anglesGrid))) .* svGrid, 1);
    p_true = abs(resp_true).^2; p_true = p_true / max(p_true + 1e-12);
    p_pred = abs(resp_pred).^2; p_pred = p_pred / max(p_pred + 1e-12);
    pat_true_db = 10*log10(p_true + 1e-12);
    pat_pred_db = 10*log10(p_pred + 1e-12);

    figure;
    plot(anglesGrid, pat_true_db, 'LineWidth',1.8); hold on;
    plot(anglesGrid, pat_pred_db, '--', 'LineWidth',1.8); grid on;
    xlabel('Angle (deg)'); ylabel('Power (dB)');
    legend({'True','Pred'}, 'Location','best');
    title(sprintf('Beam Pattern Comparison (val sample %d)', i));
end

% ---------------- Error analysis ----------------
err = Ypred - Ytrue;
err_mean = mean(err, 1);
err_std  = std(err, 0, 1);

figure; errorbar(1:numel(err_mean), err_mean, err_std, 'o-'); grid on;
xlabel('Output index'); ylabel('Error mean \pm std');
title('Per-output error statistics');

disp('Step 4 evaluation complete.');