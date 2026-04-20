%% AO — CM (Characteristic Multivector) vs Versor–SVD
%
%   1) CM_method
%      Closed-form GA-based method using a characteristic multivector.
%      Uses the Clifford toolbox + symbolic optimisation (GAALOP).
%
%   2) Versor–SVD  (Sect. 5.2.2 of "Geometric Algebra with Applications
%      in Engineering" by Christian Perwass)
%      Purely numerical GA method. Solves the versor equation
%              V A V^{-1} = B
%      via a linear system Q_+(a_i,b_i) v_+ = 0 over the even subalgebra,
%      stacked over all point pairs and solved with SVD.
%      Does NOT require the Clifford toolbox.
%
% Experiments
% -----------
%   Experiment 1 — Benchmark (200 random rigid transforms, real cloud)
%     For each randomly generated rigid transform (R_GT, t_GT):
%       Create centred point pairs (A,B) with known ground-truth rotation.
%       Estimate R with CM_method and Versor–SVD.
%       Record runtime, Frobenius error, geodesic angle error,
%       RMS point residual, and determinant deviation |det(R)-1|.
%     Outputs: accuracy boxplots, runtime boxplot + scatter,
%              .mat raw data, .mat/.csv summary table.
%
%   Experiment B — Noise robustness (real cloud + Gaussian noise)
%     Add isotropic Gaussian noise at 5 relative levels to the transformed
%     cloud; run 200 trials per level.
%     Outputs: cloud overview figure, angle-error and RMS vs noise figures,
%              .mat data.
%
%   Experiment C — Conditioning (synthetic clouds)
%     Isotropic / planar / line-like clouds with controlled anisotropy
%     (3 epsilon levels, 100 trials each). Plots angle error vs
%     log10(cond(H)) for each cloud shape.
%     Outputs: 3-panel conditioning figure, .mat data.
%
% The Versor–SVD core function `versorSVD_from_pairs_GA` is a pure numeric
% implementation of Sect. 5.2.2:
%   - builds the Γ^k_{ij} structure constants for G(3,0),
%   - forms Q_+(a_i,b_i) = (Γ_L(a_i) - Γ_R(b_i)) M_+^T for each pair,
%   - stacks Q_+ blocks over all pairs,
%   - uses SVD to find v_+ in the right nullspace,
%   - lifts v_+ to a rotor V in G^+_3 and normalises V (V ~V = 1),
%   - converts V to a 3×3 rotation by sandwiching the basis vectors.
%
% Requirements
%   Clifford algebra toolbox providing:
%       clifford_signature, wedge, reverse, unit, coefficients, e1,e2,e3
%   A 3D point cloud file 'dolphins.ply' (readable by pcread)
%   MATLAB R2016b+ (local functions in a script)
%
% All computations are in double precision.
% -------------------------------------------------------------------------

clear; close all; clc;

%% --------------------- Global configuration -----------------------------
% Toggle which experiments to run
runBenchmarkExperiment    = true;   % Experiment A — benchmark on real cloud
runNoiseExperiment        = true;   % Experiment B — noise robustness
runConditioningExperiment = true;   % Experiment C — conditioning study

% Single output folder for all experiments
outputFolder = 'output_AO_CM_vs_VSVD_combined';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% --------------------- Clifford algebra setup ---------------------------
n = 3;                     % 3D Euclidean algebra G(3,0)
clifford_signature(n, 0);  % sets up e1, e2, e3, ...

%% --------------------- Load real point cloud ----------------------------
% Used by Experiments 1 and B.
ptCloud    = pcread('dolphins.ply');          % ensure this file is on the path
model_Data = double(ptCloud.Location');       % 3×N (each column is a point)

% Bounding-box diagonal — used to scale noise levels in Experiment B
minXYZ   = min(model_Data, [], 2);
maxXYZ   = max(model_Data, [], 2);
bboxDiag = norm(maxXYZ - minXYZ);

% Quick sanity plot (diagnostic; not necessarily for the paper)
figure('Name', 'Model Point Cloud');
scatter3(model_Data(1,:), model_Data(2,:), model_Data(3,:), 5, 'filled');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Model Point Cloud'); axis equal; grid on;


% =========================================================================
%% EXPERIMENT 1 — Benchmark: 200 random rigid transforms on a real cloud
% =========================================================================
if runBenchmarkExperiment

    rng(10);   % deterministic random seed

    % Plot style defaults for this experiment
    set(0, 'DefaultAxesFontSize',  12);
    set(0, 'DefaultTextFontSize',  12);
    set(0, 'DefaultLineLineWidth', 1.5);

    no_test_cases = 200;   % number of random trials

    % Pre-allocate metric arrays for both methods
    rt_CM        = zeros(1, no_test_cases);
    rt_VSVD      = zeros(1, no_test_cases);
    fro_CM       = zeros(1, no_test_cases);
    fro_VSVD     = zeros(1, no_test_cases);
    ang_CM_deg   = zeros(1, no_test_cases);
    ang_VSVD_deg = zeros(1, no_test_cases);
    rms_CM       = zeros(1, no_test_cases);
    rms_VSVD     = zeros(1, no_test_cases);
    det_CM       = zeros(1, no_test_cases);
    det_VSVD     = zeros(1, no_test_cases);

    % (Optional) store estimated and ground-truth rotations for later use
    R_CM_all   = cell(1, no_test_cases);
    R_VSVD_all = cell(1, no_test_cases);
    R_GT_all   = cell(1, no_test_cases);

    %% ---------- Main loop: random transforms & AO solving ---------------
    for k = 1:no_test_cases
        fprintf('Test Case: %d / %d\n', k, no_test_cases);

        % Random rigid transform: per-axis angles (degrees) + translation.
        % Rotation_GT = Rx * Ry * Rz, applied as p' = Rotation_GT * p + t.
        x_angle     = randi([0, 360]);
        y_angle     = randi([0, 360]);
        z_angle     = randi([0, 360]);
        translation = [10*rand, 10*rand, 10*rand];

        [measured_Data, ~, R_GT, ~] = transformation( ...
            model_Data, x_angle, y_angle, z_angle, translation);
        R_GT_all{k} = R_GT;

        % Centre both point sets (AO uses centred correspondences)
        c1 = mean(model_Data,    2)';   % 1×3 centroid of source
        c2 = mean(measured_Data, 2)';   % 1×3 centroid of target

        A = double((model_Data'    - c1))';   % 3×N centred source
        B = double((measured_Data' - c2))';   % 3×N centred target

        % Cross- and auto-covariance matrices (used by CM pipeline)
        H  = (model_Data'    - c1)' * (measured_Data' - c2);   % 3×3
        H2 = (measured_Data' - c2)' * (measured_Data' - c2);   % 3×3

        % ── CM method ────────────────────────────────────────────────────
        % Build reciprocal frame from H and target frame from H2,
        % then solve for the characteristic-multivector rotor.
        [F_clifford_up, G_clifford] = calculateReciprocals(n, double(H), double(H2));

        tic;
        [~, ~, R_CMk] = CM_method(F_clifford_up, G_clifford);
        rt_CM(k) = toc;

        % ── Versor–SVD method ─────────────────────────────────────────────
        % Solve V A - B V = 0 in G(3,0), restrict to even subalgebra,
        % extract smallest singular vector, lift to unit rotor, build R.
        tic;
        R_VSVDk = versorSVD_from_pairs_GA(A, B);
        rt_VSVD(k) = toc;

        R_CM_all{k}   = R_CMk;
        R_VSVD_all{k} = R_VSVDk;

        % ── Metrics ───────────────────────────────────────────────────────
        % 1) Frobenius rotation error
        fro_CM(k)   = norm(R_CMk   - R_GT, 'fro');
        fro_VSVD(k) = norm(R_VSVDk - R_GT, 'fro');

        % 2) Geodesic (SO(3)) angle error [deg]
        ang_CM_deg(k)   = geodesicAngleDeg(R_CMk,   R_GT);
        ang_VSVD_deg(k) = geodesicAngleDeg(R_VSVDk, R_GT);

        % 3) RMS point residual after applying R to centred correspondences
        rms_CM(k)   = rmsResidualCentered(R_CMk,   A, B);
        rms_VSVD(k) = rmsResidualCentered(R_VSVDk, A, B);

        % 4) Determinant deviation |det(R) - 1|
        det_CM(k)   = detDeviation(R_CMk);
        det_VSVD(k) = detDeviation(R_VSVDk);
    end

    %% ---------- Console summary (quick sanity check) --------------------
    fprintf('\nMeans over %d tests:\n', no_test_cases);
    fprintf(' CM     — Fro: %.3e  Ang(deg): %.3e  RMS: %.3e  |det-1|: %.3e  t=%.3g s\n', ...
        mean(fro_CM), mean(ang_CM_deg), mean(rms_CM), mean(det_CM), mean(rt_CM));
    fprintf(' Versor — Fro: %.3e  Ang(deg): %.3e  RMS: %.3e  |det-1|: %.3e  t=%.3g s\n', ...
        mean(fro_VSVD), mean(ang_VSVD_deg), mean(rms_VSVD), mean(det_VSVD), mean(rt_VSVD));

    %% ---------- Save raw metrics ----------------------------------------
    save(fullfile(outputFolder, 'metrics_CM_vs_VSVD.mat'), ...
        'rt_CM', 'rt_VSVD', 'fro_CM', 'fro_VSVD', ...
        'ang_CM_deg', 'ang_VSVD_deg', 'rms_CM', 'rms_VSVD', ...
        'det_CM', 'det_VSVD', ...
        'R_CM_all', 'R_VSVD_all', 'R_GT_all');

    %% ---------- Build summary table: mean, std, median, min, max --------
    Metric = {}; Method = {};
    Mean = []; Std = []; Median = []; MinVal = []; MaxVal = [];

    methods      = {'CM', 'Versor–SVD'};
    metricNames  = {'Runtime (s)', 'Frobenius error', 'Angle error (deg)', ...
                    'RMS residual', '|det(R)-1|'};
    metricArrays = { rt_CM,       rt_VSVD;
                     fro_CM,      fro_VSVD;
                     ang_CM_deg,  ang_VSVD_deg;
                     rms_CM,      rms_VSVD;
                     det_CM,      det_VSVD };

    for i = 1:size(metricArrays, 1)
        for j = 1:size(metricArrays, 2)
            v = metricArrays{i,j}(:);
            Metric{end+1, 1} = metricNames{i};
            Method{end+1, 1} = methods{j};
            Mean(end+1,   1) = mean(v);
            Std(end+1,    1) = std(v);
            Median(end+1, 1) = median(v);
            MinVal(end+1, 1) = min(v);
            MaxVal(end+1, 1) = max(v);
        end
    end

    summaryTable = table(Metric, Method, Mean, Std, Median, MinVal, MaxVal);
    save(fullfile(outputFolder, 'summary_CM_vs_VSVD.mat'), 'summaryTable');
    writetable(summaryTable, fullfile(outputFolder, 'summary_CM_vs_VSVD.csv'));

    %% ---------- Figure 1: Accuracy metrics (boxplots, 2×2) --------------
    % Four panels comparing CM vs Versor–SVD:
    %   (a) Frobenius error   (b) Geodesic angle [deg]
    %   (c) RMS residual      (d) |det(R) - 1|
    h1 = figure('Name', 'Accuracy_Boxplots', ...
                'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);

    subplot(2,2,1);
    boxplot([fro_CM(:), fro_VSVD(:)], 'Labels', {'CM','Versor–SVD'});
    ylabel('||R_{est} - R_{GT}||_F');
    grid on; set(gca, 'YScale', 'log');

    subplot(2,2,2);
    boxplot([ang_CM_deg(:), ang_VSVD_deg(:)], 'Labels', {'CM','Versor–SVD'});
    ylabel('Angle error [deg]'); grid on;

    subplot(2,2,3);
    boxplot([rms_CM(:), rms_VSVD(:)], 'Labels', {'CM','Versor–SVD'});
    ylabel('RMS residual');
    grid on; set(gca, 'YScale', 'log');

    subplot(2,2,4);
    boxplot([det_CM(:), det_VSVD(:)], 'Labels', {'CM','Versor–SVD'});
    ylabel('|det(R) - 1|');
    grid on; set(gca, 'YScale', 'log');

    print(h1, fullfile(outputFolder, '01_Accuracy_Boxplots.png'), '-dpng', '-r300');

    %% ---------- Figure 2: Runtime boxplot + runtime–angle scatter -------
    % Two panels side by side:
    %   (a) Boxplot of runtime per test (CM vs Versor–SVD)
    %   (b) Scatter of runtime vs geodesic angle error
    h2 = figure('Name', 'Runtime_Box_and_Scatter', ...
                'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % (2a) Runtime boxplot with mean markers
    nexttile;
    boxplot([rt_CM(:), rt_VSVD(:)], 'Labels', {'CM','Versor–SVD'});
    ylabel('Runtime per test [s]'); grid on; hold on;
    mCM   = mean(rt_CM);
    mVSVD = mean(rt_VSVD);
    plot(1, mCM,   'kd', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
    plot(2, mVSVD, 'kd', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
    text(1.05, mCM,   sprintf('mean = %.4g s', mCM),   'VerticalAlignment', 'bottom');
    text(2.05, mVSVD, sprintf('mean = %.4g s', mVSVD), 'VerticalAlignment', 'bottom');

    % (2b) Runtime vs angle-error scatter
    nexttile;
    plot(rt_CM,   ang_CM_deg,   'b.', 'MarkerSize', 12); hold on;
    plot(rt_VSVD, ang_VSVD_deg, 'k.', 'MarkerSize', 12);
    xlabel('Runtime per test [s]'); ylabel('Angle error [deg]'); grid on;
    legend('CM', 'Versor–SVD', 'Location', 'best');

    print(h2, fullfile(outputFolder, '02_Runtime_Box_and_Scatter.png'), '-dpng', '-r300');

end % runBenchmarkExperiment


% =========================================================================
%% EXPERIMENT B — Noise robustness: real cloud + additive Gaussian noise
% =========================================================================
if runNoiseExperiment

    rng(5000);   % deterministic random seed

    % Plot style defaults for this experiment
    set(0, 'DefaultAxesFontSize', 14);
    set(0, 'DefaultTextFontSize', 14);

    fprintf('\n=== (B) Noise robustness on real cloud ===\n');

    % Noise levels as fractions of the bounding-box diagonal:
    % 0%, 0.1%, 0.5%, 1.0%, 1.5%
    noiseSigmasRel = [0, 1e-3, 5e-3, 1e-2, 1.5e-2];
    nSig           = numel(noiseSigmasRel);
    nTrialsNoise   = 200;

    % Identify representative noise levels for the overview figure
    nonzeroIdx    = find(noiseSigmasRel > 0);
    midNoiseIndex = nonzeroIdx(round((numel(nonzeroIdx)+1)/2));  % middle nonzero σ
    maxNoiseIndex = nonzeroIdx(end);                             % largest σ

    % Figure A: original cloud at zero, mid, and max noise levels
    plot_noise_overview_original(model_Data, ...
        noiseSigmasRel(midNoiseIndex), noiseSigmasRel(maxNoiseIndex), ...
        bboxDiag, outputFolder);

    % Pre-allocate metric arrays  [nTrials × nSigma]
    angNoise_CM    = zeros(nTrialsNoise, nSig);
    angNoise_V     = zeros(nTrialsNoise, nSig);
    rmsNoise_CM    = zeros(nTrialsNoise, nSig);
    rmsNoise_V     = zeros(nTrialsNoise, nSig);
    detNoise_CM    = zeros(nTrialsNoise, nSig);
    detNoise_V     = zeros(nTrialsNoise, nSig);
    rtNoise_CM     = zeros(nTrialsNoise, nSig);
    rtNoise_V      = zeros(nTrialsNoise, nSig);
    nullDimNoise_V = zeros(nTrialsNoise, nSig);   % Versor–SVD nullspace dim

    %% ---------- Main noise loop -----------------------------------------
    for sIdx = 1:nSig
        sigmaRel = noiseSigmasRel(sIdx);
        sigmaAbs = sigmaRel * bboxDiag;
        fprintf('  Noise level σ = %.3g (rel), %.3g (abs)\n', sigmaRel, sigmaAbs);

        for k = 1:nTrialsNoise
            % Random rigid transform
            x_angle     = randi([0, 360]);
            y_angle     = randi([0, 360]);
            z_angle     = randi([0, 360]);
            translation = [10*rand, 10*rand, 10*rand];

            % Apply transform to obtain a noise-free target
            [measured_Data_clean, ~, R_GT, ~] = transformation( ...
                model_Data, x_angle, y_angle, z_angle, translation);

            % Add isotropic Gaussian noise to the target cloud
            noise               = sigmaAbs * randn(size(measured_Data_clean));
            measured_Data_noisy = measured_Data_clean + noise;

            % Centre both sets
            c1 = mean(model_Data,          2);
            c2 = mean(measured_Data_noisy, 2);
            A  = model_Data          - c1;   % 3×N centred source (clean)
            B  = measured_Data_noisy - c2;   % 3×N centred target (noisy)

            % Cross- and auto-covariance for CM
            H  = A * B.';   % 3×3
            H2 = B * B.';   % 3×3

            % Solve AO with both methods
            res = run_single_trial_AO(A, B, R_GT, H, H2);

            % Store metrics
            angNoise_CM(k,sIdx)    = res.metrics_CM.ang_deg;
            angNoise_V(k,sIdx)     = res.metrics_V.ang_deg;
            rmsNoise_CM(k,sIdx)    = res.metrics_CM.rms;
            rmsNoise_V(k,sIdx)     = res.metrics_V.rms;
            detNoise_CM(k,sIdx)    = res.metrics_CM.detdev;
            detNoise_V(k,sIdx)     = res.metrics_V.detdev;
            rtNoise_CM(k,sIdx)     = res.rt_CM;
            rtNoise_V(k,sIdx)      = res.rt_V;
            nullDimNoise_V(k,sIdx) = res.info_VSVD.null_dim;
        end
    end

    %% ---------- Aggregate statistics over trials ------------------------
    meanAng_CM = mean(angNoise_CM, 1);
    meanAng_V  = mean(angNoise_V,  1);
    stdAng_CM  = std( angNoise_CM, 0, 1);
    stdAng_V   = std( angNoise_V,  0, 1);

    meanRMS_CM = mean(rmsNoise_CM, 1);
    meanRMS_V  = mean(rmsNoise_V,  1);
    stdRMS_CM  = std( rmsNoise_CM, 0, 1);
    stdRMS_V   = std( rmsNoise_V,  0, 1);

    meanRT_CM  = mean(rtNoise_CM,  1);
    meanRT_V   = mean(rtNoise_V,   1);
    stdRT_CM   = std( rtNoise_CM,  0, 1);
    stdRT_V    = std( rtNoise_V,   0, 1);

    xPercent = 100 * noiseSigmasRel;   % x-axis in percent

    %% ---------- Figure B: metrics vs noise level (2 stacked panels) -----
    hA = figure('Name', '(B) Noise_Metrics_vs_Sigma', ...
                'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);

    % Panel 1: angle error on log scale
    nexttile; hold on;
    errorbar(xPercent, meanAng_CM, stdAng_CM, 'b-o', ...
             'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'MarkerSize', 8);
    errorbar(xPercent, meanAng_V,  stdAng_V,  'k-s', ...
             'LineWidth', 1.5, 'MarkerFaceColor', 'k', 'MarkerSize', 8);
    set(gca, 'YScale', 'log');
    ylabel('Mean angle error [deg]'); grid on;
    legend('CM', 'Versor–SVD', 'Location', 'northwest');

    % Panel 2: RMS residual
    nexttile; hold on;
    errorbar(xPercent, meanRMS_CM, stdRMS_CM, 'b-o', ...
             'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'MarkerSize', 8);
    errorbar(xPercent, meanRMS_V,  stdRMS_V,  'k-s', ...
             'LineWidth', 1.5, 'MarkerFaceColor', 'k', 'MarkerSize', 8);
    xlabel('Noise level \sigma [% of bbox diag]');
    ylabel('Mean RMS residual (centered)'); grid on;
    legend('CM', 'Versor–SVD', 'Location', 'northwest');
    maxRMSplot = max([meanRMS_CM + stdRMS_CM, meanRMS_V + stdRMS_V]);
    ylim([0, maxRMSplot * 1.1]);

    print(hA, fullfile(outputFolder, 'B_Noise_Metrics_vs_Sigma.png'), '-dpng', '-r300');

    %% ---------- Save raw noise-experiment data --------------------------
    save(fullfile(outputFolder, 'B_noise_robustness.mat'), ...
        'noiseSigmasRel', 'nTrialsNoise', ...
        'angNoise_CM', 'angNoise_V', ...
        'rmsNoise_CM', 'rmsNoise_V', ...
        'detNoise_CM', 'detNoise_V', ...
        'rtNoise_CM',  'rtNoise_V', ...
        'nullDimNoise_V');

end % runNoiseExperiment


% =========================================================================
%% EXPERIMENT C — Conditioning: synthetic isotropic / planar / linear clouds
% =========================================================================
if runConditioningExperiment

    fprintf('\n=== (C) Conditioning experiment with synthetic clouds ===\n');

    % Shapes and anisotropy levels
    shapeList   = {'Isotropic', 'Planar', 'Linear'};
    epsList     = [1e-1, 1e-2, 1e-3];   % smaller eps => more degenerate cloud
    Nsynthetic  = 300;                  % points per synthetic cloud
    nTrialsCond = 100;

    nShapes = numel(shapeList);
    nEps    = numel(epsList);

    % Pre-allocate result struct array [nShapes × nEps]
    resultsCond = struct;
    for si = 1:nShapes
        for ei = 1:nEps
            resultsCond(si,ei).shape     = shapeList{si};
            resultsCond(si,ei).eps       = epsList(ei);
            resultsCond(si,ei).condH     = zeros(nTrialsCond, 1);
            resultsCond(si,ei).ang_CM    = zeros(nTrialsCond, 1);
            resultsCond(si,ei).ang_V     = zeros(nTrialsCond, 1);
            resultsCond(si,ei).nullDim_V = zeros(nTrialsCond, 1);
        end
    end

    %% ---------- Main conditioning loop ----------------------------------
    for si = 1:nShapes
        shape = shapeList{si};
        for ei = 1:nEps
            epsVal = epsList(ei);
            fprintf('  Shape = %-8s, epsilon = %.1e\n', shape, epsVal);

            for k = 1:nTrialsCond
                % Generate a synthetic cloud Y with controlled anisotropy
                Y = generate_synthetic_cloud(shape, Nsynthetic, epsVal);

                % Random rigid transform (smaller translation than Exp 1)
                x_angle     = randi([0, 360]);
                y_angle     = randi([0, 360]);
                z_angle     = randi([0, 360]);
                translation = [2*rand, 2*rand, 2*rand];

                % Apply transform: Y (source) -> X (target)
                [X, ~, R_GT, ~] = transformation(Y, x_angle, y_angle, z_angle, translation);

                % Centre both sets
                c1 = mean(Y, 2);
                c2 = mean(X, 2);
                A  = Y - c1;
                B  = X - c2;

                % Cross-covariance and condition number
                H    = A * B.';   % 3×3
                H2   = B * B.';   % 3×3
                condH = cond(H);

                % Solve AO with both methods
                res = run_single_trial_AO(A, B, R_GT, H, H2);

                % Store
                resultsCond(si,ei).condH(k)     = condH;
                resultsCond(si,ei).ang_CM(k)    = res.metrics_CM.ang_deg;
                resultsCond(si,ei).ang_V(k)     = res.metrics_V.ang_deg;
                resultsCond(si,ei).nullDim_V(k) = res.info_VSVD.null_dim;
            end
        end
    end

    %% ---------- Figure C: angle error vs log10(cond(H)) ----------------
    % Three vertically stacked tiles, one per cloud shape.
    hB = figure('Name', '(C) Conditioning_Angle_vs_CondH', ...
                'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    tl = tiledlayout(3, 1, 'Padding', 'compact', 'TileSpacing', 'loose'); 

    for si = 1:nShapes
        shape     = shapeList{si};
        allCond   = [];
        allAng_CM = [];
        allAng_V  = [];

        % Collect results across all epsilon values for this shape
        for ei = 1:nEps
            allCond   = [allCond;   resultsCond(si,ei).condH];
            allAng_CM = [allAng_CM; resultsCond(si,ei).ang_CM];
            allAng_V  = [allAng_V;  resultsCond(si,ei).ang_V];
        end

        % Filter out any pathological entries (NaN, Inf, negative)
        valid = isfinite(allCond) & (allCond > 0) & ...
                isfinite(allAng_CM) & isfinite(allAng_V) & ...
                (allAng_CM >= 0)    & (allAng_V >= 0);
        allCond   = allCond(valid);
        allAng_CM = allAng_CM(valid);
        allAng_V  = allAng_V(valid);

        logCond = log10(allCond);

        nexttile; hold on;
        semilogy(logCond, allAng_CM, 'b.', 'MarkerSize', 10);
        semilogy(logCond, allAng_V,  'k.', 'MarkerSize', 10);

        % Y-limits: avoid zero on log scale
        minAng = max(min([allAng_CM; allAng_V]), 1e-10);
        maxAng = max([allAng_CM; allAng_V]);
        ylim([minAng/2, maxAng*2]);

        % Nice integer ticks for log10(cond(H))
        xmin = floor(min(logCond));
        xmax = ceil(max(logCond));
        xticks(xmin:xmax);
        xlim([xmin-0.1, xmax+0.1]);

        grid on;
        title(sprintf('%s clouds', shape));
        ylabel('Angle error [deg]');

        if si == 1
            legend('CM', 'Versor–SVD', 'Location', 'southwest');
        end

        % Only put x-label on the bottom axes
        if si == nShapes
            xlabel('$\log_{10}(\mathrm{cond}(H))$', 'Interpreter', 'latex');
        else
            set(gca, 'XTickLabel', []);
        end
    end

    print(hB, fullfile(outputFolder, 'C_Cond_Angle_vs_CondH.png'), '-dpng', '-r300');

    %% ---------- Save conditioning data ----------------------------------
    save(fullfile(outputFolder, 'C_conditioning_synthetic.mat'), ...
        'shapeList', 'epsList', 'Nsynthetic', 'nTrialsCond', 'resultsCond');

end % runConditioningExperiment

fprintf('\nAll selected experiments finished.\n');


% =========================================================================
%% ========================= Helper functions =============================
% =========================================================================

% ------------------------------------------------------------------------
function res = run_single_trial_AO(A, B, R_GT, H, H2)
% Run CM and Versor–SVD on one AO instance.
%
% Inputs:
%   A,B   3×N centred point sets (A_i ↦ B_i)
%   R_GT  3×3 ground-truth rotation
%   H,H2  3×3 cross- and auto-covariance matrices (A*B', B*B')
%
% Output:
%   res  struct with fields:
%          R_CM, R_V
%          metrics_CM, metrics_V
%          rt_CM, rt_V
%          info_VSVD  (nullspace dimension, singular values, etc.)

    % CM method
    [F_clifford_up, G_clifford] = calculateReciprocals(3, H, H2);

    t = tic;
    [~, ~, R_CM] = CM_method(F_clifford_up, G_clifford);
    rt_CM = toc(t);

    % Versor–SVD
    t = tic;
    [R_V, info_VSVD] = versorSVD_from_pairs_GA(A, B);
    rt_V = toc(t);

    % Metrics
    metrics_CM = computeRotationMetrics(R_CM, R_GT, A, B);
    metrics_V  = computeRotationMetrics(R_V,  R_GT, A, B);

    res.R_CM       = R_CM;
    res.R_V        = R_V;
    res.metrics_CM = metrics_CM;
    res.metrics_V  = metrics_V;
    res.rt_CM      = rt_CM;
    res.rt_V       = rt_V;
    res.info_VSVD  = info_VSVD;
end

% ------------------------------------------------------------------------
function metrics = computeRotationMetrics(R_est, R_GT, A, B)
% Compute rotation accuracy metrics used in the experiments.

    metrics.fro     = norm(R_est - R_GT, 'fro');
    metrics.ang_deg = geodesicAngleDeg(R_est, R_GT);
    metrics.rms     = rmsResidualCentered(R_est, A, B);
    metrics.detdev  = detDeviation(R_est);
end

% ------------------------------------------------------------------------
function Y = generate_synthetic_cloud(shape, N, epsVal)
% Generate synthetic 3D point clouds with controlled anisotropy.
%
% Inputs:
%   shape  : 'isotropic', 'planar', or 'linear'
%   N      : number of points
%   epsVal : anisotropy parameter (small => more degenerate)
%
% Output:
%   Y      : 3×N point set

    shape = lower(shape);
    switch shape
        case 'isotropic'
            % Roughly covariance ~ I
            Y = randn(3, N);

        case 'planar'
            % Flat in z: eigenvalues ≈ (1,1,epsVal)
            stds = [1, 1, sqrt(epsVal)];
            Y    = diag(stds) * randn(3, N);

        case 'linear'
            % Line-like along x: eigenvalues ≈ (1,epsVal,epsVal)
            stds = [1, sqrt(epsVal), sqrt(epsVal)];
            Y    = diag(stds) * randn(3, N);

        otherwise
            error('Unknown synthetic shape "%s".', shape);
    end
end

% ------------------------------------------------------------------------
function plot_noise_overview_original(model_Data, sigmaMidRel, sigmaMaxRel, bboxDiag, outputFolder)
% 3-subplot overview: original clean cloud, plus additive Gaussian noise
% at two representative levels (mid and max). Noise is added directly to
% the original point cloud so the shape distortion is clear.
%
% All points are plotted in blue.

    maxPts = 3000;   % maximum number of points to plot
    N      = size(model_Data, 2);
    idx    = randperm(N, min(N, maxPts));

    Aclean = model_Data(:, idx);

    % Generate noisy versions (noise added directly to original cloud)
    Amid = Aclean + sigmaMidRel * bboxDiag * randn(size(Aclean));
    Amax = Aclean + sigmaMaxRel * bboxDiag * randn(size(Aclean));

    % Shared axis limits for fair visual comparison
    allXYZ = [Aclean, Amid, Amax];
    mins   = min(allXYZ, [], 2);
    maxs   = max(allXYZ, [], 2);

    h = figure('Name', 'NoiseOverviewOriginal', ...
               'Units', 'normalized', 'OuterPosition', [0 0 1 1]);

    % (1) Clean cloud (no noise)
    subplot(1,3,1);
    scatter3(Aclean(1,:), Aclean(2,:), Aclean(3,:), 8, 'b', 'filled');
    xlabel('X'); ylabel('Y'); zlabel('Z'); grid on; axis equal;
    xlim([mins(1) maxs(1)]); ylim([mins(2) maxs(2)]); zlim([mins(3) maxs(3)]);

    % (2) Mid noise level
    subplot(1,3,2);
    scatter3(Amid(1,:), Amid(2,:), Amid(3,:), 8, 'b', 'filled');
    xlabel('X'); ylabel('Y'); zlabel('Z'); grid on; axis equal;
    xlim([mins(1) maxs(1)]); ylim([mins(2) maxs(2)]); zlim([mins(3) maxs(3)]);

    % (3) Max noise level
    subplot(1,3,3);
    scatter3(Amax(1,:), Amax(2,:), Amax(3,:), 8, 'b', 'filled');
    xlabel('X'); ylabel('Y'); zlabel('Z'); grid on; axis equal;
    xlim([mins(1) maxs(1)]); ylim([mins(2) maxs(2)]); zlim([mins(3) maxs(3)]);

    if nargin > 4 && ~isempty(outputFolder)
        print(h, fullfile(outputFolder, 'A_Noise_Clouds_Overview.png'), '-dpng', '-r300');
    end
end

% ------------------------------------------------------------------------
function [transformed_data, transform_matrix, Rotation_GT, Translation_GT] = ...
    transformation(data, x, y, z, translation)
% Apply a rigid transform defined by per-axis rotations and a translation.
%
% Inputs:
%   data        3×N points (columns are points). If N×3 is given it is
%               transposed internally.
%   x,y,z       rotation angles in degrees about the x, y, z axes.
%   translation 1×3 translation vector
%
% Outputs:
%   transformed_data 3×N transformed points
%   transform_matrix 4×4 homogeneous transform
%   Rotation_GT      3×3 ground-truth rotation
%   Translation_GT   3×1 ground-truth translation
%
% The rotation matrix is constructed as:
%   Rotation_GT = Rx * Ry * Rz
% and applied on the left to column vectors:
%   p' = Rotation_GT * p + t

    if size(data, 1) ~= 3
        data = data.';
    end

    x = deg2rad(x);
    y = deg2rad(y);
    z = deg2rad(z);

    Rx = [1 0 0; 0 cos(x) -sin(x); 0 sin(x) cos(x)];
    Ry = [cos(y) 0 sin(y); 0 1 0; -sin(y) 0 cos(y)];
    Rz = [cos(z) -sin(z) 0; sin(z) cos(z) 0; 0 0 1];

    Rotation_GT      = Rx * Ry * Rz;
    transform_matrix = [Rotation_GT, translation(:); 0 0 0 1];
    Translation_GT   = transform_matrix(1:3, 4);

    N                = size(data, 2);
    homog_data       = [data; ones(1, N)];
    transformed      = transform_matrix * homog_data;
    transformed_data = transformed(1:3, :);
end

% ------------------------------------------------------------------------
function [F_clifford_up, G_clifford] = calculateReciprocals(n, F_matrix, G_matrix)
% Convert F_matrix (cross-covariance H) and G_matrix (auto-covariance H2)
% into Clifford vectors and compute the reciprocal frame of F.
%
% Inputs:
%   n         dimension (here 3)
%   F_matrix  3×3 matrix -> frame vectors F_i in GA
%   G_matrix  3×3 matrix -> vectors G_i in GA
%
% Outputs:
%   F_clifford_up  cell array {F1^, F2^, F3^}  reciprocal frame vectors
%   G_clifford     cell array {G1,  G2,  G3}

    F_matrix   = double(F_matrix);
    G_matrix   = double(G_matrix);
    F_clifford = cell(1, n);
    G_clifford = cell(1, n);

    % Basis {e1, e2, e3}
    basis = cell(1, n);
    for j = 1:n
        basis{j} = eval(sprintf('e%d', j));
    end

    % Matrix columns -> GA vectors
    for i = 1:n
        F_clifford{i} = 0;
        G_clifford{i} = 0;
        for j = 1:n
            F_clifford{i} = F_clifford{i} + F_matrix(j, i) * basis{j};
            G_clifford{i} = G_clifford{i} + G_matrix(j, i) * basis{j};
        end
    end

    % Pseudoscalar of the F frame: F1 ∧ F2 ∧ F3
    F_pseudo = F_clifford{1};
    for j = 2:n
        F_pseudo = wedge(F_pseudo, F_clifford{j});
    end

    % Basis pseudoscalar I = e1 ∧ e2 ∧ e3 (used in reciprocal construction)
    I_basis = basis{1};
    for j = 2:n
        I_basis = wedge(I_basis, basis{j});
    end

    % Reciprocal frame: F^i = (-1)^i * (F_j ∧ F_k) * I / |F_pseudo|
    F_clifford_up = cell(1, n);
    for i = 1:n
        wedge_excl = [];
        for j = 1:n
            if j ~= i
                if isempty(wedge_excl)
                    wedge_excl = F_clifford{j};
                else
                    wedge_excl = wedge(wedge_excl, F_clifford{j});
                end
            end
        end
        % Alternating sign and normalisation by |F_pseudo|;
        % the factor I_basis maps the bivector to the reciprocal direction.
        F_clifford_up{i} = (-1)^i * (1 / abs(F_pseudo)) * wedge_excl * I_basis;
    end
end

% ------------------------------------------------------------------------
function [New_Rotor, New_Rotor_Reverse, Rnew] = CM_method(F_clifford_up, G_clifford)
% Compute the CM rotor and extract the 3×3 rotation matrix.
%
% Inputs:
%   F_clifford_up : reciprocal frame {F1^, F2^, F3^}
%   G_clifford    : GA vectors {G1, G2, G3}
%
% Outputs:
%   New_Rotor          GA rotor V
%   New_Rotor_Reverse  reverse(V)
%   Rnew               3×3 rotation matrix

    % Unpack
    Fu1 = F_clifford_up{1};
    Fu2 = F_clifford_up{2};
    Fu3 = F_clifford_up{3};
    G1  = G_clifford{1};
    G2  = G_clifford{2};
    G3  = G_clifford{3};

    % Invariants (scalar + bivector + pseudoscalar parts)
    Inv1 = Fu1*G1 + Fu2*G2 + Fu3*G3;
    Inv2 = wedge(Fu2,Fu1) * wedge(G1,G2) ...
         + wedge(Fu3,Fu1) * wedge(G1,G3) ...
         + wedge(Fu3,Fu2) * wedge(G2,G3);
    Inv3 = wedge(Fu3,Fu2,Fu1) * wedge(G1,G2,G3);

    Inv_all = Inv1 + Inv2 + Inv3;

    % Rotor (reverse convention), then normalise to unit rotor
    Rotor_rev         = unit(1 + Inv_all);
    New_Rotor         = reverse(Rotor_rev);
    New_Rotor_Reverse = Rotor_rev;

    % Sandwich basis vectors to extract columns of R
    f1 = New_Rotor * e1 * New_Rotor_Reverse;
    f2 = New_Rotor * e2 * New_Rotor_Reverse;
    f3 = New_Rotor * e3 * New_Rotor_Reverse;

    c1 = padCoeffs(cell2mat(coefficients(f1)));
    c2 = padCoeffs(cell2mat(coefficients(f2)));
    c3 = padCoeffs(cell2mat(coefficients(f3)));

    Rnew = [c1'; c2'; c3']';
end

% ------------------------------------------------------------------------
function padded = padCoeffs(vec)
% Ensure a 3×1 vector of the first three coefficients (for e1, e2, e3).
    vec = vec(:);
    if numel(vec) < 3
        padded = [vec; zeros(3 - numel(vec), 1)];
    else
        padded = vec(1:3);
    end
end

% ------------------------------------------------------------------------
function [R, varargout] = versorSVD_from_pairs_GA(A, B)
% Versor–SVD solver for absolute orientation in G(3,0), as in §5.2.2.
%
% Inputs:
%   A,B  3×N centred point sets (columns are points), with A_i ↦ B_i
%        under an unknown rotation R (no translation; already centred).
%
% Outputs:
%   R    3×3 rotation matrix in SO(3), obtained by:
%          1) solving V A - B V = 0 in components
%          2) restricting to the even subalgebra G^+_3 (rotors) via M_+
%          3) stacking Q_+(a_i,b_i) for all pairs and applying SVD
%          4) normalising the resulting rotor V  (V ~V = 1)
%          5) sandwiching basis vectors: f_i = V e_i ~V, i = 1,2,3
%   info (optional) struct with SVD / nullspace diagnostics:
%          singular_values, tol, null_dim, case, rho, matrix_size, m

    if size(A,1)~=3 || size(B,1)~=3 || size(A,2)~=size(B,2)
        error('A,B must be 3×N with the same N.');
    end
    A = double(A);
    B = double(B);
    m = size(A, 2);   % number of point pairs

    % ----- Build Γ tensor for G(3,0) once (8×8×8) and cache persistently --
    persistent Gamma Mplus E1 E2 E3 rev_sign
    if isempty(Gamma)
        % Gamma(k,i,j) = Γ^k_{ij}  such that  e_i e_j = Γ^k_{ij} e_k
        [Gamma, rev_sign] = build_G3_gamma();

        % M_+ : 4×8 selection matrix for even subalgebra coefficients.
        % Ordering: v = [1 e1 e2 e3 e23 e31 e12 e123]^T
        %          v_+ = [1 e23 e31 e12]^T
        % By ignoring indices 2,3,4 (vectors) and 8 (pseudoscalar),
        % we strictly confine the search to G^+_3, resolving the
        % ambiguity between rotor R and its dual R*I.
        Mplus      = zeros(4, 8);
        Mplus(1,1) = 1;   % scalar part
        Mplus(2,5) = 1;   % e23
        Mplus(3,6) = 1;   % e31
        Mplus(4,7) = 1;   % e12

        % Embedded basis vectors e1, e2, e3 in 8D coefficient form
        E1 = [0;1;0;0;0;0;0;0];
        E2 = [0;0;1;0;0;0;0;0];
        E3 = [0;0;0;1;0;0;0;0];
    end

    % ----- Stack Q_+(a_i,b_i) over all pairs (each block is 8×4) ----------
    % For each pair (a_i,b_i):
    %   Q_+(a_i,b_i) v_+ = 0,
    % where Q_+ = (Γ_L(a) - Γ_R(b)) M_+^T
    Mstack = zeros(8*m, 4);
    for i = 1:m
        ai = embed_vec_G3(A(:,i));   % 8×1 embedding of vector A_i
        bi = embed_vec_G3(B(:,i));   % 8×1 embedding of vector B_i

        % Γ_L(a): left-multiplication matrix by a
        GL = zeros(8,8);
        for j = 1:8
            if ai(j) ~= 0
                GL = GL + ai(j) * Gamma(:,:,j);
            end
        end

        % Γ_R(b): right-multiplication matrix by b
        GR = zeros(8,8);
        for r = 1:8
            if bi(r) ~= 0
                GR = GR + bi(r) * squeeze(Gamma(:,r,:));
            end
        end

        Mstack(8*(i-1)+1:8*i, :) = (GL - GR) * Mplus.';
    end

    % ----- SVD: right null vector v_+ (4×1) --------------------------------
    % Use 'econ' for efficiency; we only need the 4×4 S and 4×4 V.
    [~, S, V4] = svd(Mstack, 'econ');
    svals    = diag(S);
    sigmaMax = svals(1);
    tol      = max(size(Mstack)) * eps(sigmaMax);   % similar to MATLAB rank/pinv
    nullDim  = sum(svals <= tol);

    % Decide case: unique null, LS solution, or degenerate / ambiguous
    if svals(end) <= tol && (numel(svals)==1 || svals(end-1) > tol)
        % Unique nullspace (dimension 1): exact solution
        vplus   = V4(:,end);
        caseTag = 'unique_null';
    elseif svals(end) > tol
        % No exact nullspace (LS solution): take smallest singular vector
        vplus   = V4(:,end);
        caseTag = 'least_squares';
    else
        % Nullspace dimension ≥ 2: solution not unique (degenerate data)
        warning('versorSVD:nonUnique', ...
                'Nullspace dimension >= 2; solution is not unique.');
        vplus   = V4(:,end);
        caseTag = 'multi_null';
    end

    if ~any(vplus)
        R = eye(3);
        if nargout > 1
            varargout{1} = struct('singular_values', svals, ...
                                  'tol',             tol, ...
                                  'null_dim',        nullDim, ...
                                  'case',            caseTag, ...
                                  'matrix_size',     size(Mstack), ...
                                  'm',               m);
        end
        return;
    end
    vplus = vplus / norm(vplus);   % unit vector in R^4

    % ----- Lift to full 8×1 rotor coefficients and normalise --------------
    % v_+ = M_+ v  =>  v = M_+^T v_+  with only even slots nonzero
    V  = Mplus.' * vplus;          % 8×1 rotor candidate in G^+_3
    Vt = reverse_mv(V, rev_sign);  % reverse(V)

    % ρ = <V ~V>_0 should be 1 in exact arithmetic; re-normalise numerically
    rho    = scalar_part(gp(V, Vt, Gamma));
    tolRho = 100 * eps(norm(V)^2);   % scale-aware threshold
    if rho <= tolRho
        if rho < -10*tolRho
            error('versorSVD:badRho', ...
                  '<V~V>_0 is negative beyond tolerance (degenerate data).');
        else
            rho = abs(rho);   % treat small negative as round-off
        end
    end
    s  = 1 / sqrt(rho);
    V  = V  * s;
    Vt = Vt * s;

    % ----- Build R by rotor sandwich of basis vectors ---------------------
    f1 = gp( gp(V, E1, Gamma), Vt, Gamma );   % f1 = V e1 ~V
    f2 = gp( gp(V, E2, Gamma), Vt, Gamma );   % f2 = V e2 ~V
    f3 = gp( gp(V, E3, Gamma), Vt, Gamma );   % f3 = V e3 ~V

    % Vector parts live in entries 2:4 corresponding to e1, e2, e3
    R = [ f1(2:4), f2(2:4), f3(2:4) ];

    if nargout > 1
        varargout{1} = struct('singular_values', svals, ...
                              'tol',             tol, ...
                              'null_dim',        nullDim, ...
                              'case',            caseTag, ...
                              'rho',             rho, ...
                              'matrix_size',     size(Mstack), ...
                              'm',               m);
    end
end

% ======================== Low-level GA helpers (numeric) =================

% ------------------------------------------------------------------------
function [Gamma, rev_sign] = build_G3_gamma()
% Construct structure constants Γ^k_{ij} for G(3,0) in the basis
% index <-> blade mapping:
%   1 -> 1
%   2 -> e1
%   3 -> e2
%   4 -> e3
%   5 -> e23 = e2∧e3
%   6 -> e31 = e3∧e1
%   7 -> e12 = e1∧e2
%   8 -> e123
%
% Gamma(k,i,j) is defined by:
%   e_i e_j = sum_k Gamma(k,i,j) e_k
%
% rev_sign encodes the sign change under reversion by grade:
%   (-1)^{r(r-1)/2} for grade r

    masks = [ ...
        0;   % 1
        1;   % e1
        2;   % e2
        4;   % e3
        6;   % e23
        5;   % e31
        3;   % e12
        7];  % e123

    Gamma = zeros(8, 8, 8);
    for i = 1:8
        for j = 1:8
            [k_mask, sgn] = gp_masks(masks(i), masks(j));
            k = find(masks == k_mask, 1);
            Gamma(k,i,j) = sgn;
        end
    end

    % Reverse signs per grade r: (-1)^{r(r-1)/2}
    grades   = [0 1 1 1 2 2 2 3];
    rev_sign = (-1) .^ (grades .* (grades-1) / 2);
end

% ------------------------------------------------------------------------
function [mask_out, sgn] = gp_masks(ma, mb)
% Geometric product of basis blades (Euclidean metric) represented by
% bitmasks ma, mb. Resulting blade has mask_out = ma XOR mb, and sign
% determined by the number of swaps needed to reorder basis vectors.

    mask_out = bitxor(ma, mb);

    % Count swaps: for each basis bit in A, count bits in B with lower index
    swaps = 0;
    for i = 1:3
        if bitget(ma, i)
            lower = bitand(mb, bitshift(1, i-1) - 1);
            swaps = swaps + bitcount(lower);
        end
    end
    sgn = 1 - 2*mod(swaps, 2);   % (-1)^swaps
end

% ------------------------------------------------------------------------
function c = bitcount(x)
% Popcount for up to 3 bits.
    c = (bitget(x,1)~=0) + (bitget(x,2)~=0) + (bitget(x,3)~=0);
end

% ------------------------------------------------------------------------
function v8 = embed_vec_G3(v3)
% Embed a Euclidean 3-vector v = (v1,v2,v3) into the 8D coefficient
% representation of G(3,0):
%   v8 = [0; v1; v2; v3; 0; 0; 0; 0]

    v8 = [0; v3(1); v3(2); v3(3); 0; 0; 0; 0];
end

% ------------------------------------------------------------------------
function Z = gp(X, Y, Gamma)
% Geometric product of two multivectors in component form:
%   Z_k = sum_{i,j} Γ^k_{ij} X_i Y_j

    Z = zeros(8, 1);
    for j = 1:8
        if Y(j) ~= 0
            Z = Z + (Gamma(:,:,j) * X) * Y(j);
        end
    end
end

% ------------------------------------------------------------------------
function Xr = reverse_mv(X, rev_sign)
% Reverse of a multivector in component form: sign flip per grade.
    Xr = X .* rev_sign(:);
end

% ------------------------------------------------------------------------
function s = scalar_part(X)
% Extract scalar component (basis index 1).
    s = X(1);
end

% ======================== Error / metric helpers =========================

% ------------------------------------------------------------------------
function ang = geodesicAngleDeg(Rhat, Rgt)
% Smallest rotation angle between Rhat and Rgt (in degrees).
% Uses:
%   cos(theta) = (trace(Rhat^T Rgt) - 1) / 2

    C   = (trace(Rhat.' * Rgt) - 1) / 2;
    C   = max(-1, min(1, C));   % clamp for numerical safety
    ang = rad2deg(acos(C));
end

% ------------------------------------------------------------------------
function d = detDeviation(R)
% Determinant deviation from SO(3): |det(R) - 1|
    d = abs(det(R) - 1);
end

% ------------------------------------------------------------------------
function rmsv = rmsResidualCentered(R, A, B)
% RMS residual after applying R to centred correspondences A -> B.
% A,B are 3×N (already centred).
    D    = R*A - B;                    % 3×N residuals
    rmsv = sqrt(mean(sum(D.^2, 1)));   % RMS over points
end