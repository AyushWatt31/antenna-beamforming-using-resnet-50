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
numElements = numCols;
lambda = 1; d = lambda/2;

%% 2) Steering vector for look direction
theta0 = input('Enter desired steering angle theta0 (degrees): ');
n = (0:numElements-1).';
sv0 = exp(1j*2*pi*(d/lambda)*n*sind(theta0));

%% 3) MVDR with diagonal loading (robust)
R = (H' * H) / max(numRows,1);                       % sample covariance
epsLoad = 1e-3 * trace(R)/max(numElements,1);        % loading level
Rdl = R + epsLoad*eye(numElements);                  % diagonal loading
w_mvdr = (Rdl \ sv0) / (sv0' / Rdl * sv0);           % MVDR solution
w_mvdr = w_mvdr(:) / (norm(w_mvdr) + 1e-12);         % unit norm
% Justification for loading: add μI to improve conditioning and robustness [web:369][web:373].

%% 4) LMS (standard stochastic gradient)
mu = 0.05 / (norm(H,'fro')^2/numRows + eps);         % normalized step
dvec = ones(numRows,1);
w_lms = zeros(numElements,1);
for k = 1:numRows
    x = H(k,:).';
    yhat = w_lms' * x;
    e = dvec(k) - yhat;
    w_lms = w_lms + mu * conj(e) * x;
end
w_lms = w_lms(:) / (norm(w_lms) + 1e-12);
% LMS uses negative gradient updates toward minimum MSE; step normalization improves stability [web:372].

%% 5) RLS (optimal reference)
delta = 1;                      % large inverse-initialization
lambda_rls = 0.99;              % forgetting factor
w_rls = zeros(numElements, 1);
P = (1/delta) * eye(numElements);
for k = 1:numRows
    x = H(k,:).';
    y = w_rls' * x;
    e = dvec(k) - y;
    g = P * x / (lambda_rls + x' * P * x);
    w_rls = w_rls + conj(e) * g;
    P = (P - g * x' * P) / lambda_rls;
end
w_rls = w_rls(:) / (norm(w_rls) + 1e-12);
w_optimal = w_rls;
% RLS provides fast convergence via recursive gain and inverse correlation updates [web:372][web:374].

%% 6) ResNet-50 prediction (use real-valued image input only)
% Convert complex H to grayscale safely: magnitude -> mat2gray
Mag = abs(H);                                % real-valued
imgGray = mat2gray(Mag);                     % OK: mat2gray requires real arrays [web:281]
imgResized = imresize(imgGray, [224 224]);
imgForCNN = repmat(im2single(imgResized), [1 1 3]);  % 3-channel single for CNN

S = load('trainedBeamformingNet.mat','netTransfer');
netTransfer = S.netTransfer;
y = predict(netTransfer, imgForCNN);
y = y(:).';
if numel(y) < 2*numElements
    error('Network output (%d) shorter than expected 2*numElements (%d).', numel(y), 2*numElements);
end
y_short = y(1:2*numElements);
w_resnet = complex(y_short(1:2:end), y_short(2:2:end));
w_resnet = w_resnet(:) / (norm(w_resnet) + 1e-12);
% CNN input is real-only by design; mat2gray/imlincomb accept real inputs, avoiding the earlier error [web:280][web:281].

%% 7) Metrics vs RLS ("optimal")
fprintf('Size w_optimal: %s | w_mvdr: %s | w_lms: %s | w_resnet: %s\n', ...
    mat2str(size(w_optimal)), mat2str(size(w_mvdr)), mat2str(size(w_lms)), mat2str(size(w_resnet)));
assert(isequal(size(w_optimal), size(w_mvdr), size(w_lms), size(w_resnet)), ...
       'Weight vectors have incompatible sizes.');

rmse_resnet = sqrt(mean(abs(w_resnet - w_optimal).^2));
mae_resnet  = mean(abs(w_resnet - w_optimal));
rmse_lms    = sqrt(mean(abs(w_lms - w_optimal).^2));
mae_lms     = mean(abs(w_lms - w_optimal));
rmse_mvdr   = sqrt(mean(abs(w_mvdr - w_optimal).^2));
mae_mvdr    = mean(abs(w_mvdr - w_optimal));

fprintf('\nRMSE (ResNet-50 vs RLS/Optimal): %.4f | MAE: %.4f\n', rmse_resnet, mae_resnet);
fprintf('RMSE (LMS vs RLS/Optimal): %.4f | MAE: %.4f\n', rmse_lms, mae_lms);
fprintf('RMSE (MVDR vs RLS/Optimal): %.4f | MAE: %.4f\n', rmse_mvdr, mae_mvdr);
% RMSE/MAE are standard regression errors to compare complex-weight vectors elementwise [web:379].

%% 8) Plot weights (dark theme)
fig1 = figure('Color',[0 0 0]);
ax1 = subplot(2,1,1,'Parent',fig1); set(ax1,'Color',[0 0 0],'XColor',[.85 .85 .85],'YColor',[.85 .85 .85],'GridColor',[.35 .35 .35]);
plot(ax1, abs(w_optimal),'-o','DisplayName','Optimal (RLS)','LineWidth',1.4); hold(ax1,'on');
plot(ax1, abs(w_mvdr),'-x','DisplayName','MVDR','LineWidth',1.2);
plot(ax1, abs(w_lms),'-s','DisplayName','LMS','LineWidth',1.2);
plot(ax1, abs(w_resnet),'-d','DisplayName','ResNet-50','LineWidth',1.2); hold(ax1,'off'); grid(ax1,'on');
title(ax1,'Magnitude of weights','Color',[.9 .9 .9]); legend(ax1,'TextColor',[.9 .9 .9],'Color',[.1 .1 .1]);

ax2 = subplot(2,1,2,'Parent',fig1); set(ax2,'Color',[0 0 0],'XColor',[.85 .85 .85],'YColor',[.85 .85 .85],'GridColor',[.35 .35 .35]);
plot(ax2, angle(w_optimal),'-o','DisplayName','Optimal (RLS)','LineWidth',1.4); hold(ax2,'on');
plot(ax2, angle(w_mvdr),'-x','DisplayName','MVDR','LineWidth',1.2);
plot(ax2, angle(w_lms),'-s','DisplayName','LMS','LineWidth',1.2);
plot(ax2, angle(w_resnet),'-d','DisplayName','ResNet-50','LineWidth',1.2); hold(ax2,'off'); grid(ax2,'on');
xlabel(ax2,'Antenna index','Color',[.9 .9 .9]); title(ax2,'Phase of weights','Color',[.9 .9 .9]); legend(ax2,'TextColor',[.9 .9 .9],'Color',[.1 .1 .1]);

%% 9) Beam patterns (normalized) on dark background
angles = -90:1:90;
sv = exp(1j*2*pi*(d/lambda)*n*sind(angles));
resp_opt = abs(w_optimal' * sv).^2; resp_opt = resp_opt / max(resp_opt + 1e-12);
resp_mv  = abs(w_mvdr'   * sv).^2; resp_mv  = resp_mv  / max(resp_mv  + 1e-12);
resp_lms = abs(w_lms'    * sv).^2; resp_lms = resp_lms / max(resp_lms + 1e-12);
resp_rn  = abs(w_resnet' * sv).^2; resp_rn  = resp_rn  / max(resp_rn  + 1e-12);

fig2 = figure('Color',[0 0 0]);
axp = axes('Parent',fig2,'Color',[0 0 0],'XColor',[.85 .85 .85],'YColor',[.85 .85 .85],'GridColor',[.35 .35 .35]); grid(axp,'on');
plot(axp, angles, 10*log10(resp_opt+1e-12),'w-','LineWidth',2,'DisplayName','Optimal (RLS)'); hold(axp,'on');
plot(axp, angles, 10*log10(resp_mv +1e-12),'c--','LineWidth',2,'DisplayName','MVDR');
plot(axp, angles, 10*log10(resp_lms+1e-12),'m-.','LineWidth',2,'DisplayName','LMS');
plot(axp, angles, 10*log10(resp_rn +1e-12),'y:','LineWidth',2,'DisplayName','ResNet-50'); hold(axp,'off');
xlabel(axp,'Angle (deg)','Color',[.9 .9 .9]); ylabel(axp,'Power (dB)','Color',[.9 .9 .9]);
title(axp,'Beam patterns vs RLS (optimal)','Color',[.9 .9 .9]); legend(axp,'TextColor',[.9 .9 .9],'Color',[.1 .1 .1]);