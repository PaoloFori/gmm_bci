clc; clear all; close all;
addpath(genpath('/home/paolo/Local/Matlab/yamlmatlab'));

%% load data and gmm
% not modification needed for these informations
datapath = '/home/paolo/cvsa/ic_cvsa_ws/src/gmm_cvsa/';
filein = [datapath ,'test/processed_data.csv'];
data = readmatrix(filein);
disp(['[info] Data loaded. Matrix of size: ' num2str(size(data,1)) 'x' num2str(size(data,2))])

path_yaml = [datapath ,'cfg/test.yaml'];
disp(['Loading model from: ', path_yaml]);
try
    modelData = ReadYaml(path_yaml);
    params = modelData.GmmModelCfg.params;
    model_params = modelData.GmmModelCfg.model_params;
catch ME
    disp('Error in the loading of the YAML file. Is the file path correct? Do you have YAMLMatlab installed? Is the path correct?.');
    disp(ME.message);
    return;
end
mu_datal = cellfun(@(x) x(1), params.mu);
sigma_data = cellfun(@(x) x(1), params.sigma);
nfeatures = params.nfeatures;

% model params
K = model_params.K;
weights = cell2mat(model_params.weights);
means_cell = model_params.means;
means = cell2mat(means_cell);
cov_cell = model_params.covariances; 
covariances = zeros(nfeatures, nfeatures, K);
for k = 1:K
    % Extract the k-th matrix
    matrix_cell = cov_cell{k}; 
    
    matrix_k = cell2mat(matrix_cell);
    covariances(:, :, k) = matrix_k;


end
gmm_model = gmdistribution(means, covariances, weights);

fprintf('GMM model (K=%d, NFeatures=%d) loaded.\n', K, nfeatures);

%% features extraction and classification
sparsity = nan(size(data, 1), nfeatures);
for idx_sample = 1:size(data,1)
    c_signal = data(idx_sample,:);
    [sparsity(idx_sample,:), ~, ~, ~, ~, ~, ~, ~] = compute_features_gmm(c_signal);
end

data_standardized = (sparsity - mu_datal) ./ sigma_data;

probabilities = posterior(gmm_model, data_standardized);

disp('Data classified.');

%% Load file of rosneuro
classID = 1;
SampleRate = 16;
start = 1;

files{1} = [datapath 'test/node/classified.csv'];

for i=1:length(files)
    file = files{i};
    disp(['Loading file: ' file])
    ros_data = readmatrix(file);
    matlab_data = probabilities;

    c_title = "processed with ros node simulation";
    nsamples = size(ros_data,1);
    t = 0:1/SampleRate:nsamples/SampleRate - 1/SampleRate;


    figure;
    subplot(2, 1, 1);
    hold on;
    plot(t(start:end), ros_data(start:size(t,2), classID), 'b', 'LineWidth', 1);
    plot(t(start:end), matlab_data(start:size(t,2), classID), 'r');
    legend('rosneuro', 'matlab');
    hold off;
    grid on;

    subplot(2,1,2)
    bar(t(start:end), abs(ros_data(start:size(t,2), classID)- matlab_data(start:size(t,2), classID)));
    grid on;
    xlabel('time [s]');
    ylabel('amplitude [uV]');
    title('Difference')

    sgtitle(['Evaluation' c_title])
end

function [sparsity, label_sparsity, o_l, o_r, frontal, c_l, c_r, excluded_chs] = compute_features_gmm(c_signal) %% in the features must be update for subbands
    % take the one which contribute at the 95% of the energy
    sparsity = nan(3,1);
    label_sparsity = [{'LI'},{'GI'},{'GB'}];
    o_l = sort([29 13 30 37 33 34 17]); o_r = sort([31 15 32 35 36 38 18]);
    frontal = sort([3 4 5 20 21]); c_l = sort([6 22 25 8 11 27]); c_r = sort([7 24 26 10 12 28]);
    excluded_chs = [1,2,19];

    % --- LAP --- Calcola il LAP per tutti i punti nella finestra passata (window_signal)
    % show the occipital lateralization that is strongand present during the CVSA
    P_left_window  = mean(c_signal(o_l));
    P_right_window = mean(c_signal(o_r));
    LAP_history = (P_right_window - P_left_window) ./ (P_right_window + P_left_window + eps);
    sparsity(1) = abs(LAP_history); % LAP_Mean

    % --- Gini Index + Occipital Power ---  -> when high there is a zone stronger, so IC
    % show the focusing is weighted in with the power in the occipital part, in this way ig CVSA then strong value 
    non_zeros_chs = setdiff(1:size(c_signal,2), excluded_chs);
    global_mean = mean(c_signal(non_zeros_chs)); % car filter
    current_signal_normalized = c_signal - global_mean; % remove the global energy
    mean_roi = [mean(current_signal_normalized(frontal)), mean(current_signal_normalized(c_l)), ...
        mean(current_signal_normalized(c_r)), mean(current_signal_normalized(o_l)), ...
        mean(current_signal_normalized(o_r))];
    mean_roi = abs(mean_roi); % make sure the energy is positive--> we are using peak and valli with same significance
    mean_roi_ordered = sort(mean_roi);
    n = length(mean_roi_ordered);
    sum_roi_p = 0;
    for i = 1:n
        sum_roi_p = sum_roi_p + (n+1-i) * mean_roi_ordered(i);
    end
    total_sum = sum(mean_roi_ordered);
    if total_sum > 0
        gi = (1/n) * (n+1-2*sum_roi_p/total_sum);
    else
        gi = 0;
    end
    % compute the weight factor
    pot_occipital = mean_roi(4) + mean_roi(5);
    pot_total_roi = sum(mean_roi); % Somma di F, CL, CR, OL, OR
    if pot_total_roi > 0
        occipital_power = pot_occipital / pot_total_roi;
    else
        occipital_power = 0; 
    end
    sparsity(2) = occipital_power * gi;

    % --- GB ---
    % return the global power mean, 
    sparsity(3) = global_mean;
end



