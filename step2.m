% Step 2: Modify Pre-trained ResNet-50 for Beamforming Regression
% File: modify_pretrained_network.m

clear; clc; close all;

% ---------------- Parameters (consistent with Step 1) ----------------
numElements = 8;                    % must match Step 1
numOutputs  = 2 * numElements;      % [Re1 Im1 Re2 Im2 ...]

% ---------------- Load pre-trained backbone ----------------
net = resnet50;

% Optional: visualize original graph
lgraph = layerGraph(net);
% figure; plot(lgraph); title('Original ResNet-50');

% ---------------- Remove classification tail ----------------
layersToRemove = {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'};
lgraph = removeLayers(lgraph, layersToRemove);

% ---------------- Add regression head ----------------
newHead = [
    fullyConnectedLayer(numOutputs, 'Name', 'fc_regression', ...
        'WeightLearnRateFactor', 1, 'BiasLearnRateFactor', 1)
    regressionLayer('Name', 'reg_output')
];
lgraph = addLayers(lgraph, newHead);

% Connect avg_pool to new head
lgraph = connectLayers(lgraph, 'avg_pool', 'fc_regression');

% Optional: visualize modified graph
figure; plot(lgraph); title('ResNet-50 for Regression');

% ---------------- Freeze early layers; unfreeze last stage + head ----------------
layers = lgraph.Layers;

% Default: freeze all learnable layers
for i = 1:numel(layers)
    if isprop(layers(i),'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
    end
    if isprop(layers(i),'BiasLearnRateFactor')
        layers(i).BiasLearnRateFactor   = 0;
    end
end

% Unfreeze last residual stage (res5*) to adapt features
for i = 1:numel(layers)
    if startsWith(layers(i).Name, 'res5')
        if isprop(layers(i),'WeightLearnRateFactor'), layers(i).WeightLearnRateFactor = 1; end
        if isprop(layers(i),'BiasLearnRateFactor'),   layers(i).BiasLearnRateFactor   = 1; end
    end
end

% Ensure the new head is unfrozen
for nm = ["fc_regression","reg_output"]
    idx = find(arrayfun(@(L) strcmp(L.Name, nm), layers), 1);
    if ~isempty(idx)
        if isprop(layers(idx),'WeightLearnRateFactor'), layers(idx).WeightLearnRateFactor = 1; end
        if isprop(layers(idx),'BiasLearnRateFactor'),   layers(idx).BiasLearnRateFactor   = 1; end
    end
end

% Write back into the graph
for i = 1:numel(layers)
    lgraph = replaceLayer(lgraph, layers(i).Name, layers(i));
end

% ---------------- Save for Step 3 ----------------
save('modifiedResNet50.mat','lgraph','numElements','numOutputs');
disp('Step 2 complete: modifiedResNet50.mat saved.');