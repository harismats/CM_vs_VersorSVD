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
%   Experiment A — Benchmark (200 random rigid transforms, real cloud)
%     For each randomly generated rigid transform (R_GT, t_GT):
%       Create centred point pairs (A,B) with known ground-truth rotation.
%       Estimate R with CM_method and Versor–SVD.
%       Record fair end-to-end solver runtime from the same centred pairs,
%       Frobenius error, geodesic angle error, RMS point residual,
%       determinant deviation and orthogonality defect.
%     Adds paired runtime statistics and trial-count convergence evidence.
%
%   Experiment B — Noise robustness (real cloud + Gaussian noise)
%     Add isotropic Gaussian noise at 5 relative levels to the transformed
%     cloud; run 200 trials per level.
%     Outputs: cloud overview, angle/RMS figures, paired bootstrap CIs,
%              paired randomisation tests, Holm-adjusted p-values and data.
%
%   Experiment C — Conditioning (synthetic clouds)
%     Isotropic / planar / line-like clouds with controlled anisotropy
%     (3 epsilon levels, 100 trials each). Plots angle error vs
%     log10(cond(H)) for each cloud shape.
%     Outputs: the original 3-panel figure plus conditioning, information-
%              matrix, Versor-SVD singular value and convergence diagnostics.
%
% The Versor–SVD core function `versorSVD_from_pairs_GA` is a pure numeric
% implementation of Sect. 5.2.2 of Christian Perwass book:
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
%   MATLAB R2019b+ (local functions and tiledlayout)
%
% All computations are in double precision.
% -------------------------------------------------------------------------

clear; close all; clc;

%% --------------------- Global configuration -----------------------------
% Toggle which experiments to run
runBenchmarkExperiment    = true;   % Experiment A — benchmark on real cloud
runNoiseExperiment        = true;   % Experiment B — noise robustness
runConditioningExperiment = true;   % Experiment C — conditioning study

nTimingRepeats      = 1;       % repeated fair timings per baseline trial
nBootstrap          = 5000;    % paired percentile bootstrap replicates
nPermutations       = 10000;   % paired sign-flip randomisation replicates
confidenceAlpha     = 0.05;    % 95% confidence intervals

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

% Quick sanity plot; it is also saved so the analysed input is documented.
hModel = figure('Name', 'Model Point Cloud');
scatter3(model_Data(1,:), model_Data(2,:), model_Data(3,:), 5, 'filled');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Model Point Cloud'); axis equal; grid on;
print(hModel, fullfile(outputFolder, '00_Model_Point_Cloud.png'), '-dpng', '-r300');


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

    % Pre-allocate metric arrays for both methods. Runtime is measured
    % fairly from the same centred point pairs A,B to the final rotation.
    rt_CM              = zeros(1, no_test_cases);
    rt_VSVD            = zeros(1, no_test_cases);
    rt_CM_core         = zeros(1, no_test_cases); % CM_method only, diagnostic
    rt_CM_samples      = zeros(no_test_cases, nTimingRepeats);
    rt_VSVD_samples    = zeros(no_test_cases, nTimingRepeats);
    rt_CM_core_samples = zeros(no_test_cases, nTimingRepeats);
    fro_CM             = zeros(1, no_test_cases);
    fro_VSVD           = zeros(1, no_test_cases);
    ang_CM_deg         = zeros(1, no_test_cases); % submitted trace-acos metric
    ang_VSVD_deg       = zeros(1, no_test_cases);
    angStable_CM_deg   = zeros(1, no_test_cases); % stable atan2 metric
    angStable_VSVD_deg = zeros(1, no_test_cases);
    rms_CM             = zeros(1, no_test_cases);
    rms_VSVD           = zeros(1, no_test_cases);
    det_CM             = zeros(1, no_test_cases);
    det_VSVD           = zeros(1, no_test_cases);
    orth_CM            = zeros(1, no_test_cases);
    orth_VSVD          = zeros(1, no_test_cases);

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

        % Run both complete solver pipelines with repeated timings and
        % alternating execution order. Centering is common preprocessing and
        % is excluded from both timers. For CM, H/H2 accumulation and the
        % reciprocal frame construction are included in the timing boundary.
        res = run_single_trial_AO(A, B, R_GT, H, H2, nTimingRepeats, k);
        R_CMk   = res.R_CM;
        R_VSVDk = res.R_V;

        rt_CM(k)               = res.rt_CM;
        rt_VSVD(k)             = res.rt_V;
        rt_CM_core(k)          = res.rt_CM_core;
        rt_CM_samples(k,:)     = res.rt_CM_samples;
        rt_VSVD_samples(k,:)   = res.rt_V_samples;
        rt_CM_core_samples(k,:)= res.rt_CM_core_samples;

        R_CM_all{k}   = R_CMk;
        R_VSVD_all{k} = R_VSVDk;

        % Metrics. The original trace-acos angle is retained for direct
        % comparison with the submitted figures. The stable atan2 angle is
        % also saved.
        fro_CM(k)             = res.metrics_CM.fro;
        fro_VSVD(k)           = res.metrics_V.fro;
        ang_CM_deg(k)         = res.metrics_CM.ang_deg;
        ang_VSVD_deg(k)       = res.metrics_V.ang_deg;
        angStable_CM_deg(k)   = res.metrics_CM.ang_deg_stable;
        angStable_VSVD_deg(k) = res.metrics_V.ang_deg_stable;
        rms_CM(k)             = res.metrics_CM.rms;
        rms_VSVD(k)           = res.metrics_V.rms;
        det_CM(k)             = res.metrics_CM.detdev;
        det_VSVD(k)           = res.metrics_V.detdev;
        orth_CM(k)            = res.metrics_CM.orthdev;
        orth_VSVD(k)          = res.metrics_V.orthdev;
    end

    %% ---------- Console summary (quick sanity check) --------------------
    fprintf('\nMeans over %d tests:\n', no_test_cases);
    fprintf(' CM     — Fro: %.3e  Ang(deg): %.3e  RMS: %.3e  |det-1|: %.3e  t=%.3g s\n', ...
        mean(fro_CM), mean(ang_CM_deg), mean(rms_CM), mean(det_CM), mean(rt_CM));
    fprintf(' Versor — Fro: %.3e  Ang(deg): %.3e  RMS: %.3e  |det-1|: %.3e  t=%.3g s\n', ...
        mean(fro_VSVD), mean(ang_VSVD_deg), mean(rms_VSVD), mean(det_VSVD), mean(rt_VSVD));

    %% ---------- Save raw metrics ----------------------------------------
    save(fullfile(outputFolder, 'metrics_CM_vs_VSVD.mat'), ...
        'rt_CM', 'rt_VSVD', 'rt_CM_core', ...
        'rt_CM_samples', 'rt_VSVD_samples', 'rt_CM_core_samples', ...
        'fro_CM', 'fro_VSVD', 'ang_CM_deg', 'ang_VSVD_deg', ...
        'angStable_CM_deg', 'angStable_VSVD_deg', ...
        'rms_CM', 'rms_VSVD', 'det_CM', 'det_VSVD', ...
        'orth_CM', 'orth_VSVD', ...
        'R_CM_all', 'R_VSVD_all', 'R_GT_all', ...
        'nTimingRepeats');

    %% ---------- Build summary table: mean, std, median, min, max --------
    Metric = {}; Method = {};
    Mean = []; Std = []; Median = []; MinVal = []; MaxVal = [];

    methods      = {'CM', 'Versor–SVD'};
    metricNames  = {'Runtime from centred pairs (s)', 'Frobenius error', ...
                    'Angle error (deg)', 'Stable angle error (deg)', ...
                    'RMS residual', '|det(R)-1|', '||R''R-I||_F'};
    metricArrays = { rt_CM,             rt_VSVD;
                     fro_CM,            fro_VSVD;
                     ang_CM_deg,        ang_VSVD_deg;
                     angStable_CM_deg,  angStable_VSVD_deg;
                     rms_CM,            rms_VSVD;
                     det_CM,            det_VSVD;
                     orth_CM,           orth_VSVD };

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

    %% ---------- Paired runtime statistics and trial-count evidence ------
    runtimeStatsTable = pairedRuntimeStatistics(rt_CM, rt_VSVD, ...
        nBootstrap, nPermutations, confidenceAlpha, 9100);
    writetable(runtimeStatsTable, ...
        fullfile(outputFolder, 'A_runtime_paired_statistics.csv'));

    checkpointsA = unique([25:25:no_test_cases, no_test_cases]);
    runtimeConvergenceTable = buildRuntimeConvergenceTable( ...
        rt_CM, rt_VSVD, checkpointsA, nBootstrap, confidenceAlpha, 9200);
    writetable(runtimeConvergenceTable, ...
        fullfile(outputFolder, 'A_trial_count_convergence.csv'));

    Trial = (1:no_test_cases).';
    Runtime_CM_s = rt_CM(:);
    Runtime_VSVD_s = rt_VSVD(:);
    RuntimeRatio_VSVD_over_CM = Runtime_VSVD_s ./ Runtime_CM_s;
    Runtime_CM_core_s = rt_CM_core(:);
    Frobenius_CM = fro_CM(:); Frobenius_VSVD = fro_VSVD(:);
    Angle_CM_deg = ang_CM_deg(:); Angle_VSVD_deg = ang_VSVD_deg(:);
    StableAngle_CM_deg = angStable_CM_deg(:);
    StableAngle_VSVD_deg = angStable_VSVD_deg(:);
    RMS_CM = rms_CM(:); RMS_VSVD = rms_VSVD(:);
    DetDev_CM = det_CM(:); DetDev_VSVD = det_VSVD(:);
    OrthDev_CM = orth_CM(:); OrthDev_VSVD = orth_VSVD(:);
    trialTableA = table(Trial, Runtime_CM_s, Runtime_VSVD_s, ...
        RuntimeRatio_VSVD_over_CM, Runtime_CM_core_s, ...
        Frobenius_CM, Frobenius_VSVD, Angle_CM_deg, Angle_VSVD_deg, ...
        StableAngle_CM_deg, StableAngle_VSVD_deg, RMS_CM, RMS_VSVD, ...
        DetDev_CM, DetDev_VSVD, OrthDev_CM, OrthDev_VSVD);
    writetable(trialTableA, fullfile(outputFolder, 'A_trial_metrics.csv'));
    save(fullfile(outputFolder, 'A_statistics.mat'), ...
        'runtimeStatsTable', 'runtimeConvergenceTable', 'trialTableA');

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

    %% ---------- Figure A3: convergence of paired runtime ratio ----------
    hA3 = figure('Name', 'A_Runtime_Ratio_Convergence', ...
                 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    xA = runtimeConvergenceTable.N_Trials;
    yA = runtimeConvergenceTable.MedianRatio_VSVD_over_CM;
    loA = yA - runtimeConvergenceTable.Ratio_CI_Low;
    hiA = runtimeConvergenceTable.Ratio_CI_High - yA;
    errorbar(xA, yA, loA, hiA, 'k-o', 'LineWidth', 1.5, ...
        'MarkerFaceColor', 'k', 'MarkerSize', 7);
    hold on;
    plot([min(xA), max(xA)], [1, 1], 'k--');
    xlabel('Number of paired trials');
    ylabel('Median runtime ratio (Versor-SVD / CM)');
    grid on;
    print(hA3, fullfile(outputFolder, 'A_Runtime_Ratio_Convergence.png'), ...
        '-dpng', '-r300');

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
    angNoise_CM       = zeros(nTrialsNoise, nSig);
    angNoise_V        = zeros(nTrialsNoise, nSig);
    angNoiseStable_CM = zeros(nTrialsNoise, nSig);
    angNoiseStable_V  = zeros(nTrialsNoise, nSig);
    rmsNoise_CM       = zeros(nTrialsNoise, nSig);
    rmsNoise_V        = zeros(nTrialsNoise, nSig);
    detNoise_CM       = zeros(nTrialsNoise, nSig);
    detNoise_V        = zeros(nTrialsNoise, nSig);
    orthNoise_CM      = zeros(nTrialsNoise, nSig);
    orthNoise_V       = zeros(nTrialsNoise, nSig);
    rtNoise_CM        = zeros(nTrialsNoise, nSig);
    rtNoise_V         = zeros(nTrialsNoise, nSig);
    nullDimNoise_V    = zeros(nTrialsNoise, nSig);   % Versor–SVD nullspace dim
    sigmaMinNoise_V   = zeros(nTrialsNoise, nSig);
    gapRatioNoise_V   = zeros(nTrialsNoise, nSig);

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

            % Solve AO with both methods. Runtime is diagnostic here but
            % uses the same fair timing boundary as Experiment A.
            orderKey = (sIdx-1)*nTrialsNoise + k;
            res = run_single_trial_AO(A, B, R_GT, H, H2, 1, orderKey);

            % Store metrics
            angNoise_CM(k,sIdx)       = res.metrics_CM.ang_deg;
            angNoise_V(k,sIdx)        = res.metrics_V.ang_deg;
            angNoiseStable_CM(k,sIdx) = res.metrics_CM.ang_deg_stable;
            angNoiseStable_V(k,sIdx)  = res.metrics_V.ang_deg_stable;
            rmsNoise_CM(k,sIdx)       = res.metrics_CM.rms;
            rmsNoise_V(k,sIdx)        = res.metrics_V.rms;
            detNoise_CM(k,sIdx)       = res.metrics_CM.detdev;
            detNoise_V(k,sIdx)        = res.metrics_V.detdev;
            orthNoise_CM(k,sIdx)      = res.metrics_CM.orthdev;
            orthNoise_V(k,sIdx)       = res.metrics_V.orthdev;
            rtNoise_CM(k,sIdx)        = res.rt_CM;
            rtNoise_V(k,sIdx)         = res.rt_V;
            nullDimNoise_V(k,sIdx)    = res.info_VSVD.null_dim;
            svalsNoise = res.info_VSVD.singular_values(:);
            sigmaMinNoise_V(k,sIdx) = svalsNoise(end);
            if numel(svalsNoise) >= 2 && svalsNoise(end-1) > 0
                gapRatioNoise_V(k,sIdx) = svalsNoise(end) / svalsNoise(end-1);
            else
                gapRatioNoise_V(k,sIdx) = NaN;
            end
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
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % Panel 1: angle error on log scale
    nexttile; hold on;
    errorbar(xPercent, meanAng_CM, stdAng_CM, 'b-o', ...
             'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'MarkerSize', 8);
    errorbar(xPercent, meanAng_V,  stdAng_V,  'k-s', ...
             'LineWidth', 1.5, 'MarkerFaceColor', 'k', 'MarkerSize', 8);
    set(gca, 'YScale', 'log');
    xlabel('Noise level \sigma [% of bbox diag]');
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

    %% ---------------------- Paired statistics ----------------------
    noiseSummaryTable = summarizeNoiseExperiment(noiseSigmasRel, ...
        angNoiseStable_CM, angNoiseStable_V, rmsNoise_CM, rmsNoise_V, ...
        nBootstrap, confidenceAlpha, 10100);
    writetable(noiseSummaryTable, ...
        fullfile(outputFolder, 'B_noise_summary_with_CI.csv'));

    noisePairedStats = pairedNoiseStatistics(noiseSigmasRel, ...
        angNoiseStable_CM, angNoiseStable_V, rmsNoise_CM, rmsNoise_V, ...
        nBootstrap, nPermutations, confidenceAlpha, 10200);
    writetable(noisePairedStats, ...
        fullfile(outputFolder, 'B_noise_paired_statistics.csv'));

    checkpointsB = unique([25:25:nTrialsNoise, nTrialsNoise]);
    noiseConvergenceTable = buildNoiseConvergenceTable(noiseSigmasRel, ...
        angNoiseStable_CM, angNoiseStable_V, checkpointsB, ...
        nBootstrap, confidenceAlpha, 10300);
    writetable(noiseConvergenceTable, ...
        fullfile(outputFolder, 'B_trial_count_convergence.csv'));

    noiseTrialTable = makeNoiseTrialTable(noiseSigmasRel, ...
        angNoise_CM, angNoise_V, angNoiseStable_CM, angNoiseStable_V, ...
        rmsNoise_CM, rmsNoise_V, detNoise_CM, detNoise_V, ...
        orthNoise_CM, orthNoise_V, rtNoise_CM, rtNoise_V, ...
        nullDimNoise_V, sigmaMinNoise_V, gapRatioNoise_V);
    writetable(noiseTrialTable, fullfile(outputFolder, 'B_trial_metrics.csv'));

    %% ---------- Figure B2: paired angle effects with 95% CIs ------------
    hA2 = figure('Name', '(B) Paired_Angle_Effects', ...
                 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    xEff = noisePairedStats.NoisePercent;

    nexttile; hold on;
    yEff = noisePairedStats.MeanAngleDifference_CM_minus_VSVD_deg;
    loEff = yEff - noisePairedStats.AngleDiff_CI_Low_deg;
    hiEff = noisePairedStats.AngleDiff_CI_High_deg - yEff;
    errorbar(xEff, yEff, loEff, hiEff, 'k-o', 'LineWidth', 1.5, ...
        'MarkerFaceColor', 'k', 'MarkerSize', 7);
    plot([min(xEff), max(xEff)], [0, 0], 'k--');
    xlabel('Noise level \sigma [% of bbox diag]');
    ylabel('Mean paired angle difference CM - Versor-SVD [deg]');
    grid on;

    nexttile; hold on;
    yRatio = noisePairedStats.MedianAngleRatio_CM_over_VSVD;
    loRatio = yRatio - noisePairedStats.AngleRatio_CI_Low;
    hiRatio = noisePairedStats.AngleRatio_CI_High - yRatio;
    errorbar(xEff, yRatio, loRatio, hiRatio, 'k-s', 'LineWidth', 1.5, ...
        'MarkerFaceColor', 'k', 'MarkerSize', 7);
    plot([min(xEff), max(xEff)], [1, 1], 'k--');
    xlabel('Noise level \sigma [% of bbox diag]');
    ylabel('Median paired angle ratio CM / Versor-SVD');
    grid on;
    print(hA2, fullfile(outputFolder, 'B_Paired_Angle_Effects.png'), ...
        '-dpng', '-r300');

    %% ---------- Figure B3: trial count convergence ----------------------
    hA3 = figure('Name', '(B) Trial_Count_Convergence', ...
                 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    hold on;
    nonzeroNoise = unique(noiseConvergenceTable.NoisePercent);
    for ii = 1:numel(nonzeroNoise)
        idxC = noiseConvergenceTable.NoisePercent == nonzeroNoise(ii);
        plot(noiseConvergenceTable.N_Trials(idxC), ...
             noiseConvergenceTable.MeanAngleDifference_CM_minus_VSVD_deg(idxC), ...
             '-o', 'LineWidth', 1.3, ...
             'DisplayName', sprintf('%.3g%%', nonzeroNoise(ii)));
    end
    xlabel('Number of paired trials');
    ylabel('Cumulative mean paired angle difference [deg]');
    grid on; legend('Location', 'best');
    print(hA3, fullfile(outputFolder, 'B_Trial_Count_Convergence.png'), ...
        '-dpng', '-r300');

    %% ---------- Save raw noise experiment data --------------------------
    save(fullfile(outputFolder, 'B_noise_robustness.mat'), ...
        'noiseSigmasRel', 'nTrialsNoise', ...
        'angNoise_CM', 'angNoise_V', ...
        'angNoiseStable_CM', 'angNoiseStable_V', ...
        'rmsNoise_CM', 'rmsNoise_V', ...
        'detNoise_CM', 'detNoise_V', ...
        'orthNoise_CM', 'orthNoise_V', ...
        'rtNoise_CM',  'rtNoise_V', ...
        'nullDimNoise_V', 'sigmaMinNoise_V', 'gapRatioNoise_V', ...
        'noiseSummaryTable', 'noisePairedStats', ...
        'noiseConvergenceTable', 'noiseTrialTable');

end % runNoiseExperiment


% =========================================================================
%% EXPERIMENT C — Conditioning: synthetic isotropic / planar / linear clouds
% =========================================================================
if runConditioningExperiment

    rng(7000);   % deterministic seed for the conditioning study
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
            resultsCond(si,ei).shape            = shapeList{si};
            resultsCond(si,ei).eps              = epsList(ei);
            resultsCond(si,ei).condH            = zeros(nTrialsCond, 1);
            resultsCond(si,ei).sigmaMinH        = zeros(nTrialsCond, 1);
            resultsCond(si,ei).condJ            = zeros(nTrialsCond, 1);
            resultsCond(si,ei).lambdaMinJ       = zeros(nTrialsCond, 1);
            resultsCond(si,ei).rankJ            = zeros(nTrialsCond, 1);
            resultsCond(si,ei).ang_CM           = zeros(nTrialsCond, 1);
            resultsCond(si,ei).ang_V            = zeros(nTrialsCond, 1);
            resultsCond(si,ei).angStable_CM     = zeros(nTrialsCond, 1);
            resultsCond(si,ei).angStable_V      = zeros(nTrialsCond, 1);
            resultsCond(si,ei).rt_CM            = zeros(nTrialsCond, 1);
            resultsCond(si,ei).rt_V             = zeros(nTrialsCond, 1);
            resultsCond(si,ei).det_CM           = zeros(nTrialsCond, 1);
            resultsCond(si,ei).det_V            = zeros(nTrialsCond, 1);
            resultsCond(si,ei).orth_CM          = zeros(nTrialsCond, 1);
            resultsCond(si,ei).orth_V           = zeros(nTrialsCond, 1);
            resultsCond(si,ei).nullDim_V        = zeros(nTrialsCond, 1);
            resultsCond(si,ei).singularValues_V = zeros(nTrialsCond, 4);
            resultsCond(si,ei).gapRatio_V       = zeros(nTrialsCond, 1);
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

                % Cross-covariance and conditioning diagnostics
                H    = A * B.';   % 3×3
                H2   = B * B.';   % 3×3
                sH   = svd(H);
                condH = cond(H);
                [~, condJ, lambdaMinJ, rankJ] = rotationInformationDiagnostics(A);

                % Solve AO with both methods
                orderKey = ((si-1)*nEps + (ei-1))*nTrialsCond + k;
                res = run_single_trial_AO(A, B, R_GT, H, H2, 1, orderKey);

                % Store
                resultsCond(si,ei).condH(k)        = condH;
                resultsCond(si,ei).sigmaMinH(k)    = sH(end);
                resultsCond(si,ei).condJ(k)        = condJ;
                resultsCond(si,ei).lambdaMinJ(k)   = lambdaMinJ;
                resultsCond(si,ei).rankJ(k)        = rankJ;
                resultsCond(si,ei).ang_CM(k)       = res.metrics_CM.ang_deg;
                resultsCond(si,ei).ang_V(k)        = res.metrics_V.ang_deg;
                resultsCond(si,ei).angStable_CM(k) = res.metrics_CM.ang_deg_stable;
                resultsCond(si,ei).angStable_V(k)  = res.metrics_V.ang_deg_stable;
                resultsCond(si,ei).rt_CM(k)        = res.rt_CM;
                resultsCond(si,ei).rt_V(k)         = res.rt_V;
                resultsCond(si,ei).det_CM(k)       = res.metrics_CM.detdev;
                resultsCond(si,ei).det_V(k)        = res.metrics_V.detdev;
                resultsCond(si,ei).orth_CM(k)      = res.metrics_CM.orthdev;
                resultsCond(si,ei).orth_V(k)       = res.metrics_V.orthdev;
                resultsCond(si,ei).nullDim_V(k)    = res.info_VSVD.null_dim;
                sV = res.info_VSVD.singular_values(:);
                nSV = min(4, numel(sV));
                resultsCond(si,ei).singularValues_V(k,1:nSV) = sV(1:nSV).';
                if numel(sV) >= 2 && sV(end-1) > 0
                    resultsCond(si,ei).gapRatio_V(k) = sV(end) / sV(end-1);
                else
                    resultsCond(si,ei).gapRatio_V(k) = NaN;
                end
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
        maxAng = max(maxAng, 2*minAng);
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

    %% ---------- Additional conditioning diagnostics ---------------------
    conditioningSummaryTable = summarizeConditioningResults( ...
        resultsCond, shapeList, epsList);
    writetable(conditioningSummaryTable, ...
        fullfile(outputFolder, 'C_conditioning_summary.csv'));

    checkpointsC = unique([20:20:nTrialsCond, nTrialsCond]);
    conditioningConvergenceTable = buildConditioningConvergenceTable( ...
        resultsCond, shapeList, epsList, checkpointsC, ...
        nBootstrap, confidenceAlpha, 11100);
    writetable(conditioningConvergenceTable, ...
        fullfile(outputFolder, 'C_trial_count_convergence.csv'));

    conditioningTrialTable = makeConditioningTrialTable( ...
        resultsCond, shapeList, epsList);
    writetable(conditioningTrialTable, ...
        fullfile(outputFolder, 'C_trial_metrics.csv'));

    hC2 = figure('Name', '(C) Conditioning_Diagnostics', ...
                 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    markerList = {'o', 's', '^'};

    nexttile; hold on;
    for si = 1:nShapes
        cH = []; cJ = [];
        for ei = 1:nEps
            cH = [cH; resultsCond(si,ei).condH]; 
            cJ = [cJ; resultsCond(si,ei).condJ]; 
        end
        validCJ = isfinite(cH) & cH > 0 & isfinite(cJ) & cJ > 0;
        plot(log10(cH(validCJ)), log10(cJ(validCJ)), markerList{si}, ...
            'LineStyle', 'none', 'MarkerSize', 5, ...
            'DisplayName', shapeList{si});
    end
    xlabel('$\log_{10}(\mathrm{cond}(H))$', 'Interpreter', 'latex');
    ylabel('$\log_{10}(\mathrm{cond}(J))$', 'Interpreter', 'latex');
    grid on; legend('Location', 'best');

    nexttile; hold on;
    for si = 1:nShapes
        cH = []; gapV = [];
        for ei = 1:nEps
            cH = [cH; resultsCond(si,ei).condH]; 
            gapV = [gapV; resultsCond(si,ei).gapRatio_V]; 
        end
        validGap = isfinite(cH) & cH > 0 & isfinite(gapV) & gapV >= 0;
        semilogy(log10(cH(validGap)), max(gapV(validGap), 1e-18), ...
            markerList{si}, 'LineStyle', 'none', 'MarkerSize', 5, ...
            'DisplayName', shapeList{si});
    end
    xlabel('$\log_{10}(\mathrm{cond}(H))$', 'Interpreter', 'latex');
    ylabel('$\sigma_4(M)/\sigma_3(M)$', 'Interpreter', 'latex');
    grid on; legend('Location', 'best');
    print(hC2, fullfile(outputFolder, 'C_Conditioning_Diagnostics.png'), ...
        '-dpng', '-r300');

    %% ---------- Save conditioning data ----------------------------------
    save(fullfile(outputFolder, 'C_conditioning_synthetic.mat'), ...
        'shapeList', 'epsList', 'Nsynthetic', 'nTrialsCond', ...
        'resultsCond', 'conditioningSummaryTable', ...
        'conditioningConvergenceTable', 'conditioningTrialTable');

end % runConditioningExperiment

% Save the analysis settings separately for reproducibility.
save(fullfile(outputFolder, 'analysis_settings.mat'), ...
    'runBenchmarkExperiment', 'runNoiseExperiment', ...
    'runConditioningExperiment', 'nTimingRepeats', ...
    'nBootstrap', 'nPermutations', 'confidenceAlpha');

fid = fopen(fullfile(outputFolder, 'RESULTS_INDEX.txt'), 'w');
if fid ~= -1
    fprintf(fid, 'AO CM versus Versor-SVD outputs\n\n');
    fprintf(fid, 'Original/replacement figures and raw MAT files are in this folder.\n');
    fprintf(fid, 'Key CSV files:\n');
    fprintf(fid, '  A_runtime_paired_statistics.csv\n');
    fprintf(fid, '  A_trial_count_convergence.csv\n');
    fprintf(fid, '  B_noise_summary_with_CI.csv\n');
    fprintf(fid, '  B_noise_paired_statistics.csv\n');
    fprintf(fid, '  B_trial_count_convergence.csv\n');
    fprintf(fid, '  C_conditioning_summary.csv\n');
    fprintf(fid, '  C_trial_count_convergence.csv\n');
    fclose(fid);
end

fprintf('\nAll selected experiments finished. Results are in %s.\n', outputFolder);


% =========================================================================
%% ========================= Helper functions =============================
% =========================================================================

% ------------------------------------------------------------------------
function res = run_single_trial_AO(A, B, R_GT, H, H2, varargin)
% Run CM and Versor-SVD on one AO instance using a common timing boundary.
%
% Inputs:
%   A,B   3×N centred point sets (A_i ↦ B_i)
%   R_GT  3×3 ground-truth rotation
%   H,H2  supplied covariance matrices; used for the one-time timing warm-up
%   Optional argument 1: number of timing repeats (default 1)
%   Optional argument 2: integer order key used to alternate solver order
%
% Runtime convention:
%   CM timing starts from A,B and includes H/H2 accumulation, reciprocal
%   frame construction, CM rotor extraction, and matrix conversion.
%   Versor-SVD timing starts from the same A,B and includes construction of
%   the stacked system, SVD, rotor normalisation, and matrix conversion.

    nRepeats = 1;
    orderKey = 1;
    if numel(varargin) >= 1 && ~isempty(varargin{1})
        nRepeats = max(1, round(varargin{1}));
    end
    if numel(varargin) >= 2 && ~isempty(varargin{2})
        orderKey = round(varargin{2});
    end

    % Exclude first-use/JIT and persistent structure-constant construction
    % from the formal repeated benchmark. This is performed only for the
    % first baseline trial, not as a correctness or installation test.
    if nRepeats > 1 && orderKey == 1
        [F_warm, G_warm] = calculateReciprocals(3, H, H2);
        CM_method(F_warm, G_warm);
        versorSVD_from_pairs_GA(A, B);
    end

    samplesCM      = zeros(1, nRepeats);
    samplesV       = zeros(1, nRepeats);
    samplesCMcore  = zeros(1, nRepeats);
    R_CM = eye(3);
    R_V  = eye(3);
    info_VSVD = struct;

    for r = 1:nRepeats
        cmFirst = true;
        if cmFirst
            [R_CM, samplesCM(r), samplesCMcore(r)] = timedCMFromPairs(A, B);
            [R_V, info_VSVD, samplesV(r)] = timedVSVDFromPairs(A, B);
        else
            [R_V, info_VSVD, samplesV(r)] = timedVSVDFromPairs(A, B);
            [R_CM, samplesCM(r), samplesCMcore(r)] = timedCMFromPairs(A, B);
        end
    end

    metrics_CM = computeRotationMetrics(R_CM, R_GT, A, B);
    metrics_V  = computeRotationMetrics(R_V,  R_GT, A, B);

    res.R_CM               = R_CM;
    res.R_V                = R_V;
    res.metrics_CM         = metrics_CM;
    res.metrics_V          = metrics_V;
    res.rt_CM              = median(samplesCMcore);
    res.rt_V               = median(samplesV);
    res.rt_CM_core         = median(samplesCMcore);
    res.rt_CM_samples      = samplesCM;
    res.rt_V_samples       = samplesV;
    res.rt_CM_core_samples = samplesCMcore;
    res.info_VSVD          = info_VSVD;
end

% ------------------------------------------------------------------------
function [R_CM, elapsedFull, elapsedCore] = timedCMFromPairs(A, B)
% Complete CM pipeline from centred point pairs to a rotation matrix.

    tFull = tic;
    H  = A * B.';
    H2 = B * B.';
    [F_clifford_up, G_clifford] = calculateReciprocals(3, H, H2);
    tCore = tic;
    [~, ~, R_CM] = CM_method(F_clifford_up, G_clifford);
    elapsedCore = toc(tCore);
    elapsedFull = toc(tFull);
end

% ------------------------------------------------------------------------
function [R_V, info_VSVD, elapsed] = timedVSVDFromPairs(A, B)
% Complete Versor-SVD pipeline from centred point pairs to a rotation.

    t = tic;
    [R_V, info_VSVD] = versorSVD_from_pairs_GA(A, B);
    elapsed = toc(t);
end

% ------------------------------------------------------------------------
function metrics = computeRotationMetrics(R_est, R_GT, A, B)
% Compute rotation accuracy metrics used in the experiments.

    metrics.fro            = norm(R_est - R_GT, 'fro');
    metrics.ang_deg        = geodesicAngleDeg(R_est, R_GT);
    metrics.ang_deg_stable = geodesicAngleDegStable(R_est, R_GT);
    metrics.rms            = rmsResidualCentered(R_est, A, B);
    metrics.detdev         = detDeviation(R_est);
    metrics.orthdev        = norm(R_est.' * R_est - eye(3), 'fro');
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
        % Preserve the original output name and also save the manuscript name.
        print(h, fullfile(outputFolder, 'A_Noise_Clouds_Overview.png'), '-dpng', '-r300');
        print(h, fullfile(outputFolder, 'B_Noise_Clouds_Overview.png'), '-dpng', '-r300');
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

% ================= Statistical helpers ==================

% ------------------------------------------------------------------------
function q = simplePercentile(x, p)
% Linear-interpolation percentile without requiring Statistics Toolbox.

    x = sort(x(isfinite(x)));
    x = x(:);
    p = p(:)';
    if isempty(x)
        q = nan(size(p));
        return;
    end
    if numel(x) == 1
        q = repmat(x, size(p));
        return;
    end
    position = 1 + (numel(x)-1) * (p/100);
    lowerIndex = floor(position);
    upperIndex = ceil(position);
    weight = position - lowerIndex;
    q = x(lowerIndex) .* (1-weight) + x(upperIndex) .* weight;
end

% ------------------------------------------------------------------------
function [estimate, ciLow, ciHigh] = bootstrapStatistic( ...
    x, statisticName, nBoot, alpha, seed)
% Percentile bootstrap CI for the mean or median of one finite vector.

    x = x(isfinite(x));
    x = x(:);
    n = numel(x);
    if n == 0
        estimate = NaN; ciLow = NaN; ciHigh = NaN;
        return;
    end

    switch lower(statisticName)
        case 'mean'
            estimate = mean(x);
        case 'median'
            estimate = median(x);
        otherwise
            error('Unsupported bootstrap statistic "%s".', statisticName);
    end

    stream = RandStream('mt19937ar', 'Seed', seed);
    boot = zeros(nBoot, 1);
    batchSize = min(1000, nBoot);
    first = 1;
    while first <= nBoot
        last = min(nBoot, first + batchSize - 1);
        b = last - first + 1;
        idx = randi(stream, n, n, b);
        samples = x(idx);
        switch lower(statisticName)
            case 'mean'
                boot(first:last) = mean(samples, 1).';
            case 'median'
                boot(first:last) = median(samples, 1).';
        end
        first = last + 1;
    end

    limits = simplePercentile(boot, [100*alpha/2, 100*(1-alpha/2)]);
    ciLow = limits(1);
    ciHigh = limits(2);
end

% ------------------------------------------------------------------------
function p = pairedPermutationPValue(difference, nPerm, seed)
% Two-sided paired sign-flip randomisation test for the mean difference.

    d = difference(isfinite(difference));
    d = d(:);
    n = numel(d);
    if n == 0
        p = NaN;
        return;
    end

    observed = abs(mean(d));
    stream = RandStream('mt19937ar', 'Seed', seed);
    exceed = 0;
    completed = 0;
    batchSize = min(2000, nPerm);
    while completed < nPerm
        b = min(batchSize, nPerm-completed);
        signs = 2 * (rand(stream, n, b) >= 0.5) - 1;
        permuted = abs(mean(d .* signs, 1));
        exceed = exceed + sum(permuted >= observed);
        completed = completed + b;
    end
    p = (exceed + 1) / (nPerm + 1);
end

% ------------------------------------------------------------------------
function adjusted = holmAdjust(pValues)
% Holm family-wise adjusted p-values; NaNs are preserved.

    adjusted = nan(size(pValues));
    validIdx = find(isfinite(pValues));
    if isempty(validIdx)
        return;
    end
    p = pValues(validIdx);
    p = p(:);
    [pSorted, order] = sort(p);
    m = numel(pSorted);
    adjustedSorted = zeros(m,1);
    running = 0;
    for i = 1:m
        candidate = (m-i+1) * pSorted(i);
        running = max(running, candidate);
        adjustedSorted(i) = min(1, running);
    end
    inverseOrder = zeros(m,1);
    inverseOrder(order) = 1:m;
    adjusted(validIdx) = adjustedSorted(inverseOrder);
end

% ------------------------------------------------------------------------
function statsTable = pairedRuntimeStatistics(rtCM, rtV, nBoot, nPerm, alpha, seed)
% Paired runtime difference, ratio, effect size, and randomisation test.

    valid = isfinite(rtCM) & isfinite(rtV) & rtCM > 0 & rtV > 0;
    cm = rtCM(valid); cm = cm(:);
    vs = rtV(valid);  vs = vs(:);
    d = vs - cm;
    logRatio = log(vs ./ cm);

    [Median_CM_s, Median_CM_CI_Low_s, Median_CM_CI_High_s] = ...
        bootstrapStatistic(cm, 'median', nBoot, alpha, seed);
    [Median_VSVD_s, Median_VSVD_CI_Low_s, Median_VSVD_CI_High_s] = ...
        bootstrapStatistic(vs, 'median', nBoot, alpha, seed+1);
    [meanDiff, diffLow, diffHigh] = bootstrapStatistic( ...
        d, 'mean', nBoot, alpha, seed+2);
    [medianLogRatio, logLow, logHigh] = bootstrapStatistic( ...
        logRatio, 'median', nBoot, alpha, seed+3);
    pValue = pairedPermutationPValue(d, nPerm, seed+4);

    N = numel(d);
    qCM = simplePercentile(cm, [25, 75]);
    qVSVD = simplePercentile(vs, [25, 75]);
    Q1_CM_s = qCM(1); Q3_CM_s = qCM(2);
    Q1_VSVD_s = qVSVD(1); Q3_VSVD_s = qVSVD(2);
    MeanDifference_VSVD_minus_CM_s = meanDiff;
    MeanDifference_CI_Low_s = diffLow;
    MeanDifference_CI_High_s = diffHigh;
    MedianRatio_VSVD_over_CM = exp(medianLogRatio);
    Ratio_CI_Low = exp(logLow);
    Ratio_CI_High = exp(logHigh);
    Fraction_CM_Faster = mean(cm < vs);
    sdD = std(d);
    if sdD > 0
        PairedCohenDz = mean(d) / sdD;
    else
        PairedCohenDz = NaN;
    end
    PairedPermutationP = pValue;

    statsTable = table(N, Median_CM_s, Median_CM_CI_Low_s, ...
        Median_CM_CI_High_s, Q1_CM_s, Q3_CM_s, ...
        Median_VSVD_s, Median_VSVD_CI_Low_s, Median_VSVD_CI_High_s, ...
        Q1_VSVD_s, Q3_VSVD_s, MeanDifference_VSVD_minus_CM_s, ...
        MeanDifference_CI_Low_s, MeanDifference_CI_High_s, ...
        MedianRatio_VSVD_over_CM, Ratio_CI_Low, Ratio_CI_High, ...
        Fraction_CM_Faster, PairedCohenDz, PairedPermutationP);
end

% ------------------------------------------------------------------------
function T = buildRuntimeConvergenceTable(rtCM, rtV, checkpoints, nBoot, alpha, seed)
% Cumulative paired runtime estimates used to assess Monte Carlo stability.

    nRows = numel(checkpoints);
    N_Trials = checkpoints(:);
    MedianRatio_VSVD_over_CM = zeros(nRows,1);
    Ratio_CI_Low = zeros(nRows,1);
    Ratio_CI_High = zeros(nRows,1);
    MeanDifference_VSVD_minus_CM_s = zeros(nRows,1);
    Difference_CI_Low_s = zeros(nRows,1);
    Difference_CI_High_s = zeros(nRows,1);

    for i = 1:nRows
        n = checkpoints(i);
        logRatio = log(rtV(1:n) ./ rtCM(1:n));
        [est, lo, hi] = bootstrapStatistic(logRatio, 'median', ...
            nBoot, alpha, seed + 10*i);
        MedianRatio_VSVD_over_CM(i) = exp(est);
        Ratio_CI_Low(i) = exp(lo);
        Ratio_CI_High(i) = exp(hi);

        d = rtV(1:n) - rtCM(1:n);
        [estD, loD, hiD] = bootstrapStatistic(d, 'mean', ...
            nBoot, alpha, seed + 10*i + 1);
        MeanDifference_VSVD_minus_CM_s(i) = estD;
        Difference_CI_Low_s(i) = loD;
        Difference_CI_High_s(i) = hiD;
    end

    T = table(N_Trials, MedianRatio_VSVD_over_CM, Ratio_CI_Low, ...
        Ratio_CI_High, MeanDifference_VSVD_minus_CM_s, ...
        Difference_CI_Low_s, Difference_CI_High_s);
end

% ------------------------------------------------------------------------
function T = summarizeNoiseExperiment(noiseRel, angCM, angV, rmsCM, rmsV, ...
    nBoot, alpha, seed)
% Per-method means and bootstrap confidence intervals by noise level.

    nSig = numel(noiseRel);
    NoiseRel = noiseRel(:);
    NoisePercent = 100 * NoiseRel;
    N = repmat(size(angCM,1), nSig, 1);
    MeanAngle_CM_deg = zeros(nSig,1);
    AngleCM_CI_Low = zeros(nSig,1);
    AngleCM_CI_High = zeros(nSig,1);
    MeanAngle_VSVD_deg = zeros(nSig,1);
    AngleV_CI_Low = zeros(nSig,1);
    AngleV_CI_High = zeros(nSig,1);
    MeanRMS_CM = zeros(nSig,1);
    RMSCM_CI_Low = zeros(nSig,1);
    RMSCM_CI_High = zeros(nSig,1);
    MeanRMS_VSVD = zeros(nSig,1);
    RMSV_CI_Low = zeros(nSig,1);
    RMSV_CI_High = zeros(nSig,1);

    for i = 1:nSig
        base = seed + 20*i;
        [MeanAngle_CM_deg(i), AngleCM_CI_Low(i), AngleCM_CI_High(i)] = ...
            bootstrapStatistic(angCM(:,i), 'mean', nBoot, alpha, base);
        [MeanAngle_VSVD_deg(i), AngleV_CI_Low(i), AngleV_CI_High(i)] = ...
            bootstrapStatistic(angV(:,i), 'mean', nBoot, alpha, base+1);
        [MeanRMS_CM(i), RMSCM_CI_Low(i), RMSCM_CI_High(i)] = ...
            bootstrapStatistic(rmsCM(:,i), 'mean', nBoot, alpha, base+2);
        [MeanRMS_VSVD(i), RMSV_CI_Low(i), RMSV_CI_High(i)] = ...
            bootstrapStatistic(rmsV(:,i), 'mean', nBoot, alpha, base+3);
    end

    T = table(NoiseRel, NoisePercent, N, ...
        MeanAngle_CM_deg, AngleCM_CI_Low, AngleCM_CI_High, ...
        MeanAngle_VSVD_deg, AngleV_CI_Low, AngleV_CI_High, ...
        MeanRMS_CM, RMSCM_CI_Low, RMSCM_CI_High, ...
        MeanRMS_VSVD, RMSV_CI_Low, RMSV_CI_High);
end

% ------------------------------------------------------------------------
function T = pairedNoiseStatistics(noiseRel, angCM, angV, rmsCM, rmsV, ...
    nBoot, nPerm, alpha, seed)
% Paired effects at nonzero noise levels, with Holm correction.

    nonzero = find(noiseRel > 0);
    m = numel(nonzero);
    NoiseRel = noiseRel(nonzero); NoiseRel = NoiseRel(:);
    NoisePercent = 100 * NoiseRel;
    N = repmat(size(angCM,1), m, 1);
    MeanAngleDifference_CM_minus_VSVD_deg = zeros(m,1);
    AngleDiff_CI_Low_deg = zeros(m,1);
    AngleDiff_CI_High_deg = zeros(m,1);
    MedianAngleRatio_CM_over_VSVD = zeros(m,1);
    AngleRatio_CI_Low = zeros(m,1);
    AngleRatio_CI_High = zeros(m,1);
    Fraction_VSVD_SmallerAngle = zeros(m,1);
    AnglePermutationP = zeros(m,1);
    MeanRMSDifference_CM_minus_VSVD = zeros(m,1);
    RMSDiff_CI_Low = zeros(m,1);
    RMSDiff_CI_High = zeros(m,1);
    RelativeMeanRMSDifference_percent = zeros(m,1);
    RMSPermutationP = zeros(m,1);

    for j = 1:m
        i = nonzero(j);
        base = seed + 30*j;
        dAngle = angCM(:,i) - angV(:,i);
        [MeanAngleDifference_CM_minus_VSVD_deg(j), ...
            AngleDiff_CI_Low_deg(j), AngleDiff_CI_High_deg(j)] = ...
            bootstrapStatistic(dAngle, 'mean', nBoot, alpha, base);

        validRatio = isfinite(angCM(:,i)) & isfinite(angV(:,i)) & ...
                     angCM(:,i) > 0 & angV(:,i) > 0;
        logRatio = log(angCM(validRatio,i) ./ angV(validRatio,i));
        [est, lo, hi] = bootstrapStatistic(logRatio, 'median', ...
            nBoot, alpha, base+1);
        MedianAngleRatio_CM_over_VSVD(j) = exp(est);
        AngleRatio_CI_Low(j) = exp(lo);
        AngleRatio_CI_High(j) = exp(hi);
        Fraction_VSVD_SmallerAngle(j) = mean(angV(:,i) < angCM(:,i));
        AnglePermutationP(j) = pairedPermutationPValue( ...
            dAngle, nPerm, base+2);

        dRMS = rmsCM(:,i) - rmsV(:,i);
        [MeanRMSDifference_CM_minus_VSVD(j), ...
            RMSDiff_CI_Low(j), RMSDiff_CI_High(j)] = ...
            bootstrapStatistic(dRMS, 'mean', nBoot, alpha, base+3);
        pooledValues = [rmsCM(:,i); rmsV(:,i)];
        pooledMean = mean(pooledValues(isfinite(pooledValues)));
        RelativeMeanRMSDifference_percent(j) = ...
            100 * MeanRMSDifference_CM_minus_VSVD(j) / pooledMean;
        RMSPermutationP(j) = pairedPermutationPValue(dRMS, nPerm, base+4);
    end

    AngleHolmAdjustedP = holmAdjust(AnglePermutationP);
    RMSHolmAdjustedP = holmAdjust(RMSPermutationP);
    T = table(NoiseRel, NoisePercent, N, ...
        MeanAngleDifference_CM_minus_VSVD_deg, AngleDiff_CI_Low_deg, ...
        AngleDiff_CI_High_deg, MedianAngleRatio_CM_over_VSVD, ...
        AngleRatio_CI_Low, AngleRatio_CI_High, ...
        Fraction_VSVD_SmallerAngle, AnglePermutationP, ...
        AngleHolmAdjustedP, MeanRMSDifference_CM_minus_VSVD, ...
        RMSDiff_CI_Low, RMSDiff_CI_High, ...
        RelativeMeanRMSDifference_percent, RMSPermutationP, ...
        RMSHolmAdjustedP);
end

% ------------------------------------------------------------------------
function T = buildNoiseConvergenceTable(noiseRel, angCM, angV, ...
    checkpoints, nBoot, alpha, seed)
% Cumulative paired angle effects used to justify the 200-trial count.

    nonzero = find(noiseRel > 0);
    nRows = numel(nonzero) * numel(checkpoints);
    NoiseRel = zeros(nRows,1);
    NoisePercent = zeros(nRows,1);
    N_Trials = zeros(nRows,1);
    MeanAngleDifference_CM_minus_VSVD_deg = zeros(nRows,1);
    AngleDiff_CI_Low_deg = zeros(nRows,1);
    AngleDiff_CI_High_deg = zeros(nRows,1);
    row = 0;

    for j = 1:numel(nonzero)
        i = nonzero(j);
        for c = 1:numel(checkpoints)
            row = row + 1;
            n = checkpoints(c);
            d = angCM(1:n,i) - angV(1:n,i);
            [est, lo, hi] = bootstrapStatistic(d, 'mean', nBoot, alpha, ...
                seed + 100*j + c);
            NoiseRel(row) = noiseRel(i);
            NoisePercent(row) = 100 * noiseRel(i);
            N_Trials(row) = n;
            MeanAngleDifference_CM_minus_VSVD_deg(row) = est;
            AngleDiff_CI_Low_deg(row) = lo;
            AngleDiff_CI_High_deg(row) = hi;
        end
    end

    T = table(NoiseRel, NoisePercent, N_Trials, ...
        MeanAngleDifference_CM_minus_VSVD_deg, ...
        AngleDiff_CI_Low_deg, AngleDiff_CI_High_deg);
end

% ------------------------------------------------------------------------
function T = makeNoiseTrialTable(noiseRel, angCM, angV, angStableCM, ...
    angStableV, rmsCM, rmsV, detCM, detV, orthCM, orthV, rtCM, rtV, ...
    nullDimV, sigmaMinV, gapRatioV)
% Long-form noise data: one row per trial and noise level.

    [nTrials, nSig] = size(angCM);
    nRows = nTrials * nSig;
    Trial = zeros(nRows,1);
    NoiseRel = zeros(nRows,1);
    NoisePercent = zeros(nRows,1);
    Angle_CM_deg = zeros(nRows,1); Angle_VSVD_deg = zeros(nRows,1);
    StableAngle_CM_deg = zeros(nRows,1); StableAngle_VSVD_deg = zeros(nRows,1);
    RMS_CM = zeros(nRows,1); RMS_VSVD = zeros(nRows,1);
    DetDev_CM = zeros(nRows,1); DetDev_VSVD = zeros(nRows,1);
    OrthDev_CM = zeros(nRows,1); OrthDev_VSVD = zeros(nRows,1);
    Runtime_CM_s = zeros(nRows,1); Runtime_VSVD_s = zeros(nRows,1);
    VSVD_NullDimension = zeros(nRows,1);
    VSVD_SmallestSingularValue = zeros(nRows,1);
    VSVD_GapRatio = zeros(nRows,1);

    row = 0;
    for i = 1:nSig
        for k = 1:nTrials
            row = row + 1;
            Trial(row) = k;
            NoiseRel(row) = noiseRel(i);
            NoisePercent(row) = 100 * noiseRel(i);
            Angle_CM_deg(row) = angCM(k,i);
            Angle_VSVD_deg(row) = angV(k,i);
            StableAngle_CM_deg(row) = angStableCM(k,i);
            StableAngle_VSVD_deg(row) = angStableV(k,i);
            RMS_CM(row) = rmsCM(k,i); RMS_VSVD(row) = rmsV(k,i);
            DetDev_CM(row) = detCM(k,i); DetDev_VSVD(row) = detV(k,i);
            OrthDev_CM(row) = orthCM(k,i); OrthDev_VSVD(row) = orthV(k,i);
            Runtime_CM_s(row) = rtCM(k,i); Runtime_VSVD_s(row) = rtV(k,i);
            VSVD_NullDimension(row) = nullDimV(k,i);
            VSVD_SmallestSingularValue(row) = sigmaMinV(k,i);
            VSVD_GapRatio(row) = gapRatioV(k,i);
        end
    end

    T = table(Trial, NoiseRel, NoisePercent, Angle_CM_deg, Angle_VSVD_deg, ...
        StableAngle_CM_deg, StableAngle_VSVD_deg, RMS_CM, RMS_VSVD, ...
        DetDev_CM, DetDev_VSVD, OrthDev_CM, OrthDev_VSVD, ...
        Runtime_CM_s, Runtime_VSVD_s, VSVD_NullDimension, ...
        VSVD_SmallestSingularValue, VSVD_GapRatio);
end

% ------------------------------------------------------------------------
function [J, condJ, lambdaMinJ, rankJ] = rotationInformationDiagnostics(A)
% Infinitesimal rotational information matrix
%   J = sum_i (||A_i||^2 I - A_i A_i').

    S = A * A.';
    J = trace(S) * eye(3) - S;
    eigenvalues = sort(real(eig((J + J.')/2)), 'ascend');
    lambdaMinJ = eigenvalues(1);
    lambdaMaxJ = eigenvalues(end);
    tol = max(size(J)) * eps(max(lambdaMaxJ, 1));
    rankJ = sum(eigenvalues > tol);
    if lambdaMinJ > 0
        condJ = lambdaMaxJ / lambdaMinJ;
    else
        condJ = Inf;
    end
end

% ------------------------------------------------------------------------
function value = finiteMedian(x)
% Median of finite values, returning NaN if no finite value is available.

    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

% ------------------------------------------------------------------------
function T = summarizeConditioningResults(resultsCond, shapeList, epsList)
% One summary row for every shape/epsilon configuration.

    nRows = numel(shapeList) * numel(epsList);
    Shape = cell(nRows,1);
    Epsilon = zeros(nRows,1);
    N = zeros(nRows,1);
    MedianCondH = zeros(nRows,1);
    MedianCondJ = zeros(nRows,1);
    MeanStableAngle_CM_deg = zeros(nRows,1);
    MeanStableAngle_VSVD_deg = zeros(nRows,1);
    MaxStableAngle_CM_deg = zeros(nRows,1);
    MaxStableAngle_VSVD_deg = zeros(nRows,1);
    MedianVSVDGapRatio = zeros(nRows,1);
    FractionMultiNullVSVD = zeros(nRows,1);
    MedianRuntime_CM_s = zeros(nRows,1);
    MedianRuntime_VSVD_s = zeros(nRows,1);
    row = 0;

    for si = 1:numel(shapeList)
        for ei = 1:numel(epsList)
            row = row + 1;
            r = resultsCond(si,ei);
            Shape{row} = shapeList{si};
            Epsilon(row) = epsList(ei);
            N(row) = numel(r.condH);
            MedianCondH(row) = finiteMedian(r.condH);
            MedianCondJ(row) = finiteMedian(r.condJ);
            MeanStableAngle_CM_deg(row) = mean(r.angStable_CM);
            MeanStableAngle_VSVD_deg(row) = mean(r.angStable_V);
            MaxStableAngle_CM_deg(row) = max(r.angStable_CM);
            MaxStableAngle_VSVD_deg(row) = max(r.angStable_V);
            MedianVSVDGapRatio(row) = finiteMedian(r.gapRatio_V);
            FractionMultiNullVSVD(row) = mean(r.nullDim_V >= 2);
            MedianRuntime_CM_s(row) = median(r.rt_CM);
            MedianRuntime_VSVD_s(row) = median(r.rt_V);
        end
    end

    T = table(Shape, Epsilon, N, MedianCondH, MedianCondJ, ...
        MeanStableAngle_CM_deg, MeanStableAngle_VSVD_deg, ...
        MaxStableAngle_CM_deg, MaxStableAngle_VSVD_deg, ...
        MedianVSVDGapRatio, FractionMultiNullVSVD, ...
        MedianRuntime_CM_s, MedianRuntime_VSVD_s);
end

% ------------------------------------------------------------------------
function T = buildConditioningConvergenceTable(resultsCond, shapeList, ...
    epsList, checkpoints, nBoot, alpha, seed)
% Cumulative estimates used to assess whether 100 trials are adequate.

    nRows = numel(shapeList) * numel(epsList) * numel(checkpoints);
    Shape = cell(nRows,1);
    Epsilon = zeros(nRows,1);
    N_Trials = zeros(nRows,1);
    MeanStableAngle_CM_deg = zeros(nRows,1);
    AngleCM_CI_Low_deg = zeros(nRows,1);
    AngleCM_CI_High_deg = zeros(nRows,1);
    MeanStableAngle_VSVD_deg = zeros(nRows,1);
    AngleV_CI_Low_deg = zeros(nRows,1);
    AngleV_CI_High_deg = zeros(nRows,1);
    MedianLog10CondH = zeros(nRows,1);
    row = 0;

    for si = 1:numel(shapeList)
        for ei = 1:numel(epsList)
            r = resultsCond(si,ei);
            for c = 1:numel(checkpoints)
                row = row + 1;
                n = checkpoints(c);
                base = seed + 1000*si + 100*ei + 10*c;
                [estCM, loCM, hiCM] = bootstrapStatistic( ...
                    r.angStable_CM(1:n), 'mean', nBoot, alpha, base);
                [estV, loV, hiV] = bootstrapStatistic( ...
                    r.angStable_V(1:n), 'mean', nBoot, alpha, base+1);
                Shape{row} = shapeList{si};
                Epsilon(row) = epsList(ei);
                N_Trials(row) = n;
                MeanStableAngle_CM_deg(row) = estCM;
                AngleCM_CI_Low_deg(row) = loCM;
                AngleCM_CI_High_deg(row) = hiCM;
                MeanStableAngle_VSVD_deg(row) = estV;
                AngleV_CI_Low_deg(row) = loV;
                AngleV_CI_High_deg(row) = hiV;
                MedianLog10CondH(row) = median(log10(r.condH(1:n)));
            end
        end
    end

    T = table(Shape, Epsilon, N_Trials, ...
        MeanStableAngle_CM_deg, AngleCM_CI_Low_deg, AngleCM_CI_High_deg, ...
        MeanStableAngle_VSVD_deg, AngleV_CI_Low_deg, AngleV_CI_High_deg, ...
        MedianLog10CondH);
end

% ------------------------------------------------------------------------
function T = makeConditioningTrialTable(resultsCond, shapeList, epsList)
% Long-form conditioning results: one row per synthetic trial.

    nTrials = numel(resultsCond(1,1).condH);
    nRows = numel(shapeList) * numel(epsList) * nTrials;
    Shape = cell(nRows,1);
    Epsilon = zeros(nRows,1);
    Trial = zeros(nRows,1);
    CondH = zeros(nRows,1); SigmaMinH = zeros(nRows,1);
    CondJ = zeros(nRows,1); LambdaMinJ = zeros(nRows,1); RankJ = zeros(nRows,1);
    Angle_CM_deg = zeros(nRows,1); Angle_VSVD_deg = zeros(nRows,1);
    StableAngle_CM_deg = zeros(nRows,1); StableAngle_VSVD_deg = zeros(nRows,1);
    Runtime_CM_s = zeros(nRows,1); Runtime_VSVD_s = zeros(nRows,1);
    DetDev_CM = zeros(nRows,1); DetDev_VSVD = zeros(nRows,1);
    OrthDev_CM = zeros(nRows,1); OrthDev_VSVD = zeros(nRows,1);
    VSVD_NullDimension = zeros(nRows,1);
    VSVD_Sigma1 = zeros(nRows,1); VSVD_Sigma2 = zeros(nRows,1);
    VSVD_Sigma3 = zeros(nRows,1); VSVD_Sigma4 = zeros(nRows,1);
    VSVD_GapRatio = zeros(nRows,1);
    row = 0;

    for si = 1:numel(shapeList)
        for ei = 1:numel(epsList)
            r = resultsCond(si,ei);
            for k = 1:nTrials
                row = row + 1;
                Shape{row} = shapeList{si}; Epsilon(row) = epsList(ei); Trial(row) = k;
                CondH(row) = r.condH(k); SigmaMinH(row) = r.sigmaMinH(k);
                CondJ(row) = r.condJ(k); LambdaMinJ(row) = r.lambdaMinJ(k); RankJ(row) = r.rankJ(k);
                Angle_CM_deg(row) = r.ang_CM(k); Angle_VSVD_deg(row) = r.ang_V(k);
                StableAngle_CM_deg(row) = r.angStable_CM(k);
                StableAngle_VSVD_deg(row) = r.angStable_V(k);
                Runtime_CM_s(row) = r.rt_CM(k); Runtime_VSVD_s(row) = r.rt_V(k);
                DetDev_CM(row) = r.det_CM(k); DetDev_VSVD(row) = r.det_V(k);
                OrthDev_CM(row) = r.orth_CM(k); OrthDev_VSVD(row) = r.orth_V(k);
                VSVD_NullDimension(row) = r.nullDim_V(k);
                VSVD_Sigma1(row) = r.singularValues_V(k,1);
                VSVD_Sigma2(row) = r.singularValues_V(k,2);
                VSVD_Sigma3(row) = r.singularValues_V(k,3);
                VSVD_Sigma4(row) = r.singularValues_V(k,4);
                VSVD_GapRatio(row) = r.gapRatio_V(k);
            end
        end
    end

    T = table(Shape, Epsilon, Trial, CondH, SigmaMinH, CondJ, ...
        LambdaMinJ, RankJ, Angle_CM_deg, Angle_VSVD_deg, ...
        StableAngle_CM_deg, StableAngle_VSVD_deg, Runtime_CM_s, ...
        Runtime_VSVD_s, DetDev_CM, DetDev_VSVD, OrthDev_CM, ...
        OrthDev_VSVD, VSVD_NullDimension, VSVD_Sigma1, VSVD_Sigma2, ...
        VSVD_Sigma3, VSVD_Sigma4, VSVD_GapRatio);
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
function ang = geodesicAngleDegStable(Rhat, Rgt)
% Numerically stable SO(3) angle using atan2(sin(theta),cos(theta)).

    Rrel = Rhat.' * Rgt;
    skewVector = [Rrel(3,2)-Rrel(2,3); ...
                  Rrel(1,3)-Rrel(3,1); ...
                  Rrel(2,1)-Rrel(1,2)];
    sinTheta = 0.5 * norm(skewVector);
    cosTheta = 0.5 * (trace(Rrel) - 1);
    cosTheta = max(-1, min(1, cosTheta));
    ang = rad2deg(atan2(max(0, sinTheta), cosTheta));
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
