clear; clc; close all;

%% 1) User Input Channel Matrix (complex a+bi)
numRows = input('Enter number of rows in channel matrix (Nr): ');
numCols = input('Enter number of columns (Nt): ');
H  = complex(zeros(numRows, numCols));
disp('Enter each row using MATLAB complex syntax (a+bi), e.g., 0.8-0.2i  0.4+0.1i ...');
for r = 1:numRows
    rowStr = input(sprintf('Row %d: ', r), 's');
    vals = str2num(rowStr); %#ok<ST2NM>
    if numel(vals) ~= numCols
        error('Row %d: expected %d entries.', r, numCols);
    end
    H(r,:) = vals;
end
numElements = numCols;   % one transmit weight per TX element

%% 2) Array/steering params
lambda = 1;                % normalized wavelength
d = lambda/2;              % inter-element spacing
theta0 = input('Enter desired steering angle theta0 (degrees): ');
n = (0:numElements-1).';
sv0 = exp(1j*2*pi*(d/lambda)*n*sind(theta0));   % steering vector at theta0

%% 3) MVDR with diagonal loading (stable)
R = (H' * H) / max(numRows,1);                   % sample covariance
epsLoad = 1e-3 * trace(R)/numElements;           % loading level
Rdl = R + epsLoad*eye(numElements);              % loaded covariance
w_mvdr = (Rdl \ sv0) / (sv0' / Rdl * sv0);       % MVDR solution
w_mvdr = w_mvdr(:) / (norm(w_mvdr) + 1e-12);

%% 4) Choose reference "optimal" (here MVDR)
w_optimal = w_mvdr;

%% 5) LMS weights (stable step and normalization)
mu = 0.05 / (norm(H,'fro')^2/numRows + eps);     % normalized step
dvec = ones(numRows,1);                           % desired response
w_lms = zeros(numElements,1);
for t = 1:numRows
    x = H(t,:).';                                 % snapshot
    yhat = w_lms' * x;
    e = dvec(t) - yhat;
    w_lms = w_lms + mu * conj(e) * x;
end
w_lms = w_lms(:) / (norm(w_lms) + 1e-12);

%% 6) Build CNN input image from complex H (REAL ONLY!)
% Use magnitude-only grayscale; no mat2gray on complex arrays.
Mag = abs(H);                    % real, non-negative
imgGray = mat2gray(Mag);         % safe now (real valued)
imgResized = imresize(imgGray, [224 224]);
imgForCNN = repmat(im2single(imgResized), [1 1 3]);   % 3-channel single

%% 7) ResNet-50 predicted weights (first 2*numElements outputs)
S = load('trainedBeamformingNet.mat','netTransfer');
netTransfer = S.netTransfer;
y = predict(netTransfer, imgForCNN);
y = y(:).';                                    % flatten
if numel(y) < 2*numElements
    error('Network output (%d) shorter than expected 2*numElements (%d).', numel(y), 2*numElements);
end
y_short = y(1:2*numElements);
w_resnet = complex(y_short(1:2:end), y_short(2:2:end));
w_resnet = w_resnet(:) / (norm(w_resnet) + 1e-12);

%% 8) Size check and metrics
fprintf('Size w_optimal: %s\n', mat2str(size(w_optimal)));
fprintf('Size w_mvdr   : %s\n', mat2str(size(w_mvdr)));
fprintf('Size w_lms    : %s\n', mat2str(size(w_lms)));
fprintf('Size w_resnet : %s\n', mat2str(size(w_resnet)));

if isequal(size(w_optimal), size(w_mvdr), size(w_lms), size(w_resnet))
    rmse_resnet = sqrt(mean(abs(w_resnet - w_optimal).^2));
    mae_resnet  = mean(abs(w_resnet - w_optimal));
    rmse_lms    = sqrt(mean(abs(w_lms - w_optimal).^2));
    mae_lms     = mean(abs(w_lms - w_optimal));
    rmse_mvdr   = sqrt(mean(abs(w_mvdr - w_optimal).^2));
    mae_mvdr    = mean(abs(w_mvdr - w_optimal));
    fprintf('\nRMSE (ResNet-50 vs Optimal): %.4f\n', rmse_resnet);
    fprintf('MAE  (ResNet-50 vs Optimal): %.4f\n', mae_resnet);
    fprintf('RMSE (LMS vs Optimal)     : %.4f\n', rmse_lms);
    fprintf('MAE  (LMS vs Optimal)     : %.4f\n', mae_lms);
    fprintf('RMSE (MVDR vs Optimal)    : %.4f\n', rmse_mvdr);
    fprintf('MAE  (MVDR vs Optimal)    : %.4f\n', mae_mvdr);
else
    error('Weight vectors have incompatible sizes.');
end

%% 9) Plots (dark theme for visibility)
fig = figure('Color',[0 0 0]);
ax1 = subplot(2,2,1,'Parent',fig);
set(ax1,'Color',[0 0 0],'XColor',[0.85 0.85 0.85],'YColor',[0.85 0.85 0.85]);
imshow(imgResized,'Parent',ax1); title(ax1,'Grayscale |H|','Color',[0.9 0.9 0.9]);

ax2 = subplot(2,2,2,'Parent',fig);
set(ax2,'Color',[0 0 0],'XColor',[0.85 0.85 0.85],'YColor',[0.85 0.85 0.85], ...
         'GridColor',[0.35 0.35 0.35]); grid(ax2,'on');
plot(ax2, abs([w_optimal w_mvdr w_lms w_resnet]),'.'); % quick view
title(ax2,'Weight magnitudes (quick view)','Color',[0.9 0.9 0.9]);

ax3 = subplot(2,1,2,'Parent',fig);
set(ax3,'Color',[0 0 0],'XColor',[0.85 0.85 0.85],'YColor',[0.85 0.85 0.85], ...
         'GridColor',[0.35 0.35 0.35]); grid(ax3,'on');
plot(abs(w_optimal),'-o','DisplayName','Optimal','LineWidth',1.4); hold on;
plot(abs(w_mvdr),'-x','DisplayName','MVDR','LineWidth',1.2);
plot(abs(w_lms),'-s','DisplayName','LMS','LineWidth',1.2);
plot(abs(w_resnet),'-d','DisplayName','ResNet-50','LineWidth',1.2); hold off;
legend('TextColor',[0.9 0.9 0.9],'Color',[0.1 0.1 0.1]);
xlabel('Antenna index','Color',[0.9 0.9 0.9]);
ylabel('|w|','Color',[0.9 0.9 0.9]);
title('Magnitude comparison','Color',[0.9 0.9 0.9]);