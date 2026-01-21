clear all; % close all;

addpath('/home/paolo/cvsa/ic_cvsa_ws/src/analysis_bci/equal_ros')
addpath('/home/paolo/cvsa/ic_cvsa_ws/src/analysis_bci/utils')

%% Initialization
DATAPAH = '/home/paolo/cvsa/ic_cvsa_ws/src/';
classes = [769 770];
nchannels = 16;
nclasses = length(classes);
filterOrder = 4;
avg = 1;
threshold_gmm_ic = 0.7;
channels_label = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'Fp1', 'CP1', 'CPz', 'CP2', 'Fp2'};


%% Load file
[filenames, pathname] = uigetfile('*.gdf', 'Select GDF Files', 'MultiSelect', 'on');
if ischar(filenames)
    filenames = {filenames};
end
subject = filenames{1}(1:2);
time_str = datestr(now, 'ddmmyyyy_HHMMSS');
gmm_file = ['gmm_' subject '_' time_str '_mi.yaml'];
save_path_gmm = [DATAPAH, 'gmm_bci/cfg/' gmm_file];
extra = '_dummy'; % '' or '_dummy'
save_path_qda_dataset = [DATAPAH 'qda_bci/create_qda/datasets/gmm/data_' subject '_' time_str '_mi' extra '.mat'];

%% understand the band
nFiles = length(filenames);
peaks = zeros(1, nFiles);
for idx_file = 1:nFiles
    fullpath_file = fullfile(pathname, filenames{idx_file});
    peaks(idx_file) = analyze_alpha_peak(fullpath_file, 'RestTrigger', 786, 'band', [8 14], 'target_regions', {'C1', 'C3', 'C2', 'C4'});
end

%% start processing data
bands = [{[8 13]} {[18 24]}];
bands_str = cellfun(@(x) sprintf('%d-%d', x(1), x(2)), bands, 'UniformOutput', false);
nbands = length(bands);
signals = cell(1, nbands);
artifacts = [];
headers = cell(1, nbands);
for idx_band = 1:nbands
    headers{idx_band}.TYP = [];
    headers{idx_band}.DUR = [];
    headers{idx_band}.POS = [];
    signals{idx_band} = [];
end

for idx_file= 1: nFiles
    fullpath_file = fullfile(pathname, filenames{idx_file});
    disp(['file (' num2str(idx_file) '/' num2str(nFiles)  '): ', filenames{idx_file}]);
    [c_signal,header] = sload(fullpath_file);
    c_signal = c_signal(:,1:nchannels);
    sampleRate = header.SampleRate;

    excl_ch = {'Fp1', 'Fp2', 'EOG'};
    [found, indices] = ismember(excl_ch, channels_label);
    excl_chs = indices(found);

    % for power band using hilbert transformation and artefact remotion -----------------------------------------------
    bufferSize = floor(avg*sampleRate);
    chunkSize = 32;
    eog.filterOrder = 4;
    eog.band = [1 10];
    eog.label = {'Fp1', 'Fp2'};
    eog.h_threshold = 60;
    eog.v_threshold = 60;
    picks.filterOrder = 4;
    picks.freq = 1; % remove antneuro problems
    picks.threshold = 100;
    artifact = artifact_rejection(c_signal, header, nchannels, bufferSize, chunkSize, eog, picks);
    artifacts = cat(1, artifacts, artifact(:,:));

    disp('   [proc] power band');
    for idx_band = 1:nbands
        band = bands{idx_band};

        [signal_processed, header_processed] = processing_onlineROS_CAR_hilbert(c_signal, header, nchannels, bufferSize, filterOrder, band, chunkSize, excl_chs);
        
        c_header = headers{1, idx_band};
        c_header.sampleRate = header_processed.SampleRate/chunkSize;
        c_header.channels_labels = header_processed.Label;
        if isempty(find(header_processed.EVENT.TYP == 2, 1)) % no eye calibration
            c_header.TYP = cat(1, c_header.TYP, header_processed.EVENT.TYP);
            c_header.DUR = cat(1, c_header.DUR, header_processed.EVENT.DUR);
            c_header.POS = cat(1, c_header.POS, header_processed.EVENT.POS + size(signals{1, idx_band}, 1));
        else
            k = find(header_processed.EVENT.TYP == 1, 1);
            c_header.TYP = cat(1, c_header.TYP, header_processed.EVENT.TYP(k:end));
            c_header.DUR = cat(1, c_header.DUR, header_processed.EVENT.DUR(k:end));
            c_header.POS = cat(1, c_header.POS, header_processed.EVENT.POS(k:end) + size(signals{1, idx_band}, 1));
        end
        signals{1, idx_band} = cat(1, signals{1, idx_band}, signal_processed(:,:));
        headers{1, idx_band} = c_header;
    end
end


%% Labelling data 
events = headers{1,1};
sampleRate = events.sampleRate;
cuePOS = events.POS(ismember(events.TYP, classes));
cueDUR = events.DUR(ismember(events.TYP, classes));
cueTYP = events.TYP(ismember(events.TYP, classes));

fixPOS = events.POS(events.TYP == 786);
fixDUR = events.DUR(events.TYP == 786);

cfPOS = events.POS(events.TYP == 781);
cfDUR = events.DUR(events.TYP == 781);

minDurCue = min(cueDUR);
minDurFix = min(fixDUR);
ntrial = length(cuePOS);

%% Labeling data for the dataset
trial_start = nan(ntrial, 1);
trial_end = nan(ntrial, 1);
trial_typ = nan(ntrial, 1);
for idx_trial = 1:ntrial
    trial_start(idx_trial) = fixPOS(idx_trial);
    trial_typ(idx_trial) = cueTYP(idx_trial);
    trial_end(idx_trial) = cfPOS(idx_trial) + cfDUR(idx_trial) - 1;
end

min_trial_data = min(trial_end - trial_start+1);
trial_data = nan(min_trial_data, nbands, nchannels, ntrial); % data x bands x channels x trial
artifacts_data = nan(min_trial_data, ntrial); % data x trial
for idx_band = 1:nbands
    c_signal = signals{idx_band};
    c_artifact = artifacts;
    for trial = 1:ntrial
        c_start = trial_start(trial);
        c_end = trial_start(trial) + min_trial_data - 1;
        trial_data(:,idx_band,:,trial) = c_signal(c_start:c_end,:);
        artifacts_data(:,trial) = c_artifact(c_start:c_end,:);
    end
end

%% refactoring the data --> odd trial class 1 even class 2
idx_classes_trial = nan(ntrial/2, nclasses);
for idx_class = 1:nclasses
    idx_classes_trial(:,idx_class) = find(trial_typ == classes(idx_class));
end

tmp_data = nan(size(trial_data));
tmp_art = nan(size(artifacts_data));
trial_typ = nan(size(trial_typ));
i = 1;
for idx_trial_class = 1:2:ntrial
    for idx_class = 1:nclasses
        tmp_data(:,:,:,idx_trial_class + idx_class - 1) = trial_data(:,:,:,idx_classes_trial(i, idx_class));
        tmp_art(:,idx_trial_class + idx_class - 1) = artifacts_data(:,idx_classes_trial(i, idx_class));
        trial_typ(idx_trial_class + idx_class - 1) = classes(idx_class);
    end
    i = i + 1;
end
trial_data = tmp_data; % samples x bands x channels x trials
artifacts_data = tmp_art;

%% compute sparsity
% define regions
nsparsity = 1;
sparsity = nan(min_trial_data, ntrial, nsparsity*nbands); % sample x trial x sparsity*nbands
% o_l_ch = {'P3', 'O1', 'P5', 'P1', 'PO5', 'PO3', 'PO7'};
% o_r_ch = {'P4', 'O2', 'P2', 'P6', 'PO4', 'PO6', 'PO8'};
% c_l_ch = {'FC1', 'C3', 'CP1', 'FC3', 'C1', 'CP3'};
% c_r_ch = {'FC2', 'C4', 'CP2', 'FC4', 'C2', 'CP4'};

o_l_ch = {};
o_r_ch = {};
c_c_ch = {'Cz', 'CPz', 'FCz'};
c_l_ch = {'C3', 'CP1', 'C1'};
c_r_ch = {'C4', 'CP2', 'C2'};

[~, o_l] = ismember(o_l_ch, channels_label);
[~, o_r] = ismember(o_r_ch, channels_label);
[~, c_l] = ismember(c_l_ch, channels_label);
[~, c_r] = ismember(c_r_ch, channels_label);
[~, c_c] = ismember(c_c_ch, channels_label);

type = 'mi';

for c = 1:ntrial
    c_data = squeeze(trial_data(:,:,:,c)); % samples x band x channels

    for sample = 1:min_trial_data
        c_sample = squeeze(c_data(sample,:,:)); % bands x channels

        sparsity_vec = []; label_plot = [];

        for idx_band = 1:nbands
            tmp = squeeze(c_sample(idx_band,:)); % 1 x channels
            
            [tmp, label_plot_tmp] =  compute_features_icnic_mi(tmp, c_l, c_r, c_c);

            sparsity_vec = [sparsity_vec; tmp];
            label_plot = [label_plot, label_plot_tmp];
        end

        sparsity(sample, c, :) = sparsity_vec;
    end
end

for idx_band = 1:nbands
    for idx_s = 1:nsparsity
        idx = (idx_band-1)*nsparsity+idx_s;
        label_plot{idx} = [label_plot{idx}, ' ', bands_str{idx_band}];
    end
end

%% ----------------- gmm -----------------
% update to work with subbands -> tesista
choosen_band = 1;
K_range = 2:2; 
best_gmm = [];
min_bic = inf;

sparsity_cf = squeeze(sparsity(minDurFix+minDurCue+1:end, :,:));
artifacts_cf = squeeze(artifacts_data(minDurFix+minDurCue+1:end, :));

% z-score -> train and use the mu and var also for the test
data_3D = sparsity_cf(:, 1:ntrial, :);
data_2D = reshape(data_3D, size(data_3D, 1) * size(data_3D,2), size(data_3D,3));
artefact_2D = artifacts_cf(:, 1:ntrial);
artefact_1D = reshape(artefact_2D, size(artefact_2D, 1) * size(artefact_2D,2), 1);
data_2D_noArtif = data_2D(artefact_1D == 0,:);
mu_features = mean(data_2D_noArtif, 1);
sigma_features = std(data_2D_noArtif, 0, 1);
sigma_features(sigma_features == 0) = eps;
data_2D_noArtif = (data_2D_noArtif - mu_features) ./ sigma_features;

% extract data all trials
sparsity_cuecf = squeeze(sparsity(minDurFix+1:end, :,:));
data_2D = reshape(sparsity_cuecf, size(sparsity_cuecf, 1) * size(sparsity_cuecf,2), size(sparsity_cuecf,3));
data_standardized_2D = (data_2D - mu_features) ./ sigma_features;
sparsity_cuecf = reshape(data_standardized_2D, size(sparsity_cuecf, 1), ntrial, size(sparsity_cuecf,3));
sparsity_cf = sparsity_cuecf(minDurCue+1:end,:,:);

disp('Esecuzione di GMM sui dati di training globali...');
options = statset('MaxIter', 1000, 'Display', 'off');
for k = K_range
    try
        % RegularizationValue = 1e-5 evita che le gaussiane collassino su un punto
        gm_temp = fitgmdist(data_2D_noArtif, k, ...
                            'Options', options, ...
                            'CovarianceType', 'Full', ...
                            'SharedCovariance', false, ...
                            'RegularizationValue', 1e-5, ...
                            'Replicates', 150); 
        
        if gm_temp.BIC < min_bic
            min_bic = gm_temp.BIC;
            best_gmm = gm_temp;
        end
    catch
        continue;
    end
end

gmm_model = best_gmm;
K = gmm_model.NumComponents;
disp(['GMM ottimizzato: K = ' num2str(K) ' (BIC = ' num2str(min_bic) ')']);

log_features = gmm_model.mu;
mean_score = mean(log_features, 2); 
[~, idx_ic] = max(mean_score); 
[~, idx_nic] = min(mean_score);
fprintf('Mappatura Automatica:\n');
fprintf('Cluster %d (Valore %.2f) -> IC (Task/High Focus)\n', idx_ic, mean_score(idx_ic));
fprintf('Cluster %d (Valore %.2f) -> NIC (Rest/Low Focus)\n', idx_nic, mean_score(idx_nic));

classes_icnic = zeros(1,2);
classes_icnic(idx_ic) = 1;

% Crea le etichette finali
train_gmm = nan(ntrial * size(sparsity_cf, 1), nsparsity*nbands);
for c = 1:ntrial
    train_gmm((c-1)*size(sparsity_cf, 1) + 1: c * size(sparsity_cf, 1),:) = sparsity_cf(:,c,:);
end
P_soft = posterior(gmm_model, train_gmm);
cluster_labels = nan(size(sparsity_cf, 1), ntrial); % contains the prob to be ic
for c = 1:ntrial
    cluster_labels(:,c) = P_soft((c-1)*size(sparsity_cf, 1) + 1: c * size(sparsity_cf, 1), idx_ic);
end

fprintf('Mappatura: Cluster GMM %d -> "ic", Cluster GMM %d -> "nic"\n', idx_ic, idx_nic);

% plot the C
disp('centroids: ')
disp(gmm_model.mu)

%% --- VISUALIZZAZIONE GMM ---
% Ottieni le etichette "Hard" dal GMM (assegna ogni punto al cluster più probabile)
cluster_idx = cluster(gmm_model, data_2D_noArtif);

% Calcola la Silhouette
figure;
[s, ~] = silhouette(data_2D_noArtif, cluster_idx);
mean_sil = mean(s);

disp(['Silhouette Score Medio: ' num2str(mean_sil)]);
% Salva il grafico per il paper
title(['Silhouette Plot (Score: ' num2str(mean_sil, '%.2f') ')']);

eva_ch = evalclusters(data_2D_noArtif, cluster_idx, 'CalinskiHarabasz');
disp(['Calinski-Harabasz Index: ' num2str(eva_ch.CriterionValues)]);

eva_db = evalclusters(data_2D_noArtif, cluster_idx, 'DaviesBouldin');
disp(['Davies-Bouldin Index: ' num2str(eva_db.CriterionValues)]);

%% save the gmm
save_gmm(gmm_model, mu_features, sigma_features, filenames, save_path_gmm, o_l, o_r, c_c, c_l, c_r, excl_chs, channels_label, bands, classes_icnic, threshold_gmm_ic, type)


%% extract and save data for the QDA
data = trial_data(minDurCue+minDurFix+1:end,:,:,:); % data x bands x channels x trial
nsamples = size(data,1);
X = []; X_all = []; X_nic = [];
y = []; y_all = [];
count_artifact = 0; count_all = 0; count_rejected = 0;
for idx_band = 1:nbands
    tmp_X = []; tmp_X_all = [];  tmp_X_nic = [];
    y = []; y_all = []; y_ic = []; y_nic = [];
    trials = []; trials_all = [];
    for idx_trial =  1:ntrial
        for idx_sample = 1:nsamples
            if artifacts_cf(idx_sample,idx_trial) == 0 % no artifact
                tmp_X_all = [tmp_X_all; data(idx_sample,idx_band,:,idx_trial)];
                trials_all = [trials_all, idx_trial];
                y_all = [y_all; trial_typ(idx_trial)];
                if cluster_labels(idx_sample, idx_trial) >= threshold_gmm_ic % IC state
                    tmp_X = [tmp_X; data(idx_sample,idx_band,:,idx_trial)];
                    y = [y; trial_typ(idx_trial)];
                    trials = [trials; idx_trial];
                else
                    tmp_X_nic = [tmp_X_nic; data(idx_sample,idx_band,:,idx_trial)];
                    y_nic = [y_nic; trial_typ(idx_trial)];
                    count_rejected = count_rejected + 1/nbands;
                end
            else
                count_artifact = count_artifact + 1/nbands;
            end
            count_all = count_all + 1/nbands ;
        end
    end

    X = [X, tmp_X];
    X_all = [X_all, tmp_X_all];
    X_nic = [X_nic, tmp_X_nic];
end
disp(['all sample for the trials: ' num2str(count_all) ', rejected for artifact: ' num2str(count_artifact) ', rejected gmm: ' num2str(count_rejected) ', acepted: ' num2str(size(X, 1))])

%% Features selection QDA
% fisher score
fisher = nan(nbands*2, nchannels);
label_fisher = [];

for idx_ch=1:nchannels
    for idx_band = 1:nbands
        % IC
        mu1 = mean(log(X(y == classes(1),idx_band, idx_ch)));
        sigma1 = std(log(X(y == classes(1),idx_band, idx_ch)));
        mu2 = mean(log(X(y == classes(2),idx_band,idx_ch)));
        sigma2 = std(log(X(y == classes(2),idx_band, idx_ch)));
        fisher((idx_band-1)*3 + 1, idx_ch) = abs(mu1 - mu2)^2 / (sigma1^2 + sigma2^2);
        label_fisher = [label_fisher, {['IC', bands_str{idx_band}]}];

        % all
        mu1 = mean(log(X_all(y_all == classes(1), idx_band, idx_ch)));
        sigma1 = std(log(X_all(y_all == classes(1),idx_band, idx_ch)));
        mu2 = mean(log(X_all(y_all == classes(2),idx_band, idx_ch)));
        sigma2 = std(log(X_all(y_all == classes(2),idx_band, idx_ch)));
        fisher((idx_band-1)*3 +2, idx_ch) = abs(mu1 - mu2)^2 / (sigma1^2 + sigma2^2);
        label_fisher = [label_fisher, {['traditional', bands_str{idx_band}]}];

        % nic
        mu1 = mean(log(X_nic(y_nic == classes(1), idx_band, idx_ch)));
        sigma1 = std(log(X_nic(y_nic == classes(1),idx_band, idx_ch)));
        mu2 = mean(log(X_nic(y_nic == classes(2),idx_band, idx_ch)));
        sigma2 = std(log(X_nic(y_nic == classes(2),idx_band, idx_ch)));
        fisher((idx_band-1)*3 +3, idx_ch) = abs(mu1 - mu2)^2 / (sigma1^2 + sigma2^2);
        label_fisher = [label_fisher, {['nic', bands_str{idx_band}]}];
    end
end

figure();
imagesc(fisher')
colorbar;
yticks(1:nchannels); yticklabels(channels_label)
xticks(1:size(fisher, 1)); xticklabels(label_fisher)
sgtitle('gmm ic and classical fisher score')

figure('Color', 'w', 'Name', 'Fisher Score Comparison Bars', 'Position', [100, 100, 1200, 600]);
b = bar(fisher'); % Trasposto: X=Canali, Gruppi=Conditions
% Assegnazione colori ciclica (Rosso=IC, Grigio=All, Blu=NIC)
ic_color = [0.8 0.2 0.2];
all_color = [0.6 0.6 0.6];
nic_color = [0.2 0.2 0.8];
for k = 1:length(b)
    rem_k = mod(k-1, 3) + 1; % 1, 2, 3 ripetuto
    if rem_k == 1
        b(k).FaceColor = ic_color; % IC
    elseif rem_k == 2
        b(k).FaceColor = all_color; % All
    else
        b(k).FaceColor = nic_color; % NIC
    end
end
legend([b(1), b(2), b(3)], {'IC (Selected)', 'Traditional', 'NIC (Rejected)'}, 'Location', 'bestoutside');
xticks(1:nchannels); xticklabels(channels_label); xtickangle(45);
ylabel('Fisher Score');
title('Feature Separability: Does GMM select better data?');
grid on;


% R^2
for idx_band = 1:nbands
    r2_ic_allch = calc_r2_from_data(squeeze(log(X(:,idx_band,:))), y, 'Plot', false, 'ChanLabels', channels_label); 
    r2_all_allch = calc_r2_from_data(squeeze(log(X_all(:,idx_band,:))), y_all, 'Plot', false, 'ChanLabels', channels_label);
    r2_nic_allch = calc_r2_from_data(squeeze(log(X_nic(:,idx_band,:))), y_nic, 'Plot', false, 'ChanLabels', channels_label);


    figure('Color', 'w', 'Name', ['R2 Gain - Band ' bands_str{idx_band}], 'Position', [150, 150, 1000, 500]);
    data_to_plot = [r2_ic_allch(:), r2_all_allch(:), r2_nic_allch(:)];
    b_r2 = bar(data_to_plot);
    b_r2(1).FaceColor = ic_color;   % Rosso
    b_r2(2).FaceColor = all_color;  % Grigio
    b_r2(3).FaceColor = nic_color;  % Blu
    legend({'GMM IC (Signal)', 'Traditional (Mixed)', 'GMM NIC (Noise)'}, 'Location', 'best');
    ylabel('Signed R^2 (Class Separability)', 'FontSize', 12, 'FontWeight', 'bold');
    title(['Separability Analysis | Band: ' bands_str{idx_band}], 'FontSize', 14);
    grid on;
    xticks(1:nchannels); xticklabels(channels_label); xtickangle(45);
    yline(0, 'k-', 'LineWidth', 1);

    mean_r2_ic = mean(abs(r2_ic_allch));
    mean_r2_all = mean(abs(r2_all_allch));
    mean_r2_nic = mean(abs(r2_nic_allch));
    gain = mean_r2_ic - mean_r2_all;
    pct_gain = (gain / (mean_r2_all + eps)) * 100;
    subtitle({['IC vs All Improvement: +' num2str(pct_gain, '%.1f') '%'], ...
              ['Avg R^2: IC=' num2str(mean_r2_ic,'%.3f') ' | All=' num2str(mean_r2_all,'%.3f') ' | NIC=' num2str(mean_r2_nic,'%.3f')]}, ...
              'FontSize', 10, 'Color', 'k');
end

%% save data for qda
channels_labels =  [{{'C4'}}, {{'C4'}}]; % firs 8-13 then 18-24
idx_channels = [];
for i = 1:length(channels_labels)
    c_t = channels_labels{i};
    [~, tmp_idx] = ismember(c_t, channels_label);
    idx_channels = [idx_channels, {tmp_idx}];
end
save(save_path_qda_dataset, 'X', 'y', 'trials', 'gmm_file', 'classes', 'idx_channels', 'channels_labels', 'filenames', 'bands')
disp(['QDA model saved in ', save_path_qda_dataset]);




%% ----------- FUNCTIONS --------
% save gmm
function save_gmm(gmm_model, mu_features, sigma_features, files, save_path_gmm, o_l_idx, o_r_idx, c_c_idx, c_l_idx, c_r_idx, excluded_chs, channels_labels, band, classes_icnic, threshold_gmm_ic, type)
    % --- Dati GMM model ---
    % gmm_model:       gmm model 
    % mu_features:     mean of the data
    % sigma_features:  std of the data
    % files:           file from which the gmm is trained
    % DATAPAH:         where to save
    % subject:         subject of the experiment

    % model parameters
    K = gmm_model.NumComponents;
    nfeatures = gmm_model.NumVariables;
    means = gmm_model.mu;                    
    covariances = gmm_model.Sigma;          
    weights = gmm_model.ComponentProportion;
    weightsStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), weights, 'UniformOutput', false), ', ');
    rowStrings = cell(K, 1);
    for i = 1:K
        mean_row = means(i, :); 
        innerListStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), mean_row, 'UniformOutput', false), ', ');
        rowStrings{i} = sprintf('    - [%s]', innerListStr);
    end
    meansStr = strjoin(rowStrings, '\n');

    covStrings = cell(K, 1);
    for i = 1:K % Loop su ogni cluster K
        cov_matrix = covariances(:, :, i); % Estrai la matrice nfeatures x nfeatures
        matrixRowStrings = cell(nfeatures, 1);
        for j = 1:nfeatures % Loop su ogni riga della matrice
            cov_row = cov_matrix(j, :);
            innerListStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), cov_row, 'UniformOutput', false), ', ');
            matrixRowStrings{j} = sprintf('      - [%s]', innerListStr); % Indentazione YAML
        end
        % Combina le righe della matrice, con l'indicatore di lista YAML '-'
        covStrings{i} = sprintf('    - \n%s', strjoin(matrixRowStrings, '\n'));
    end
    covariancesStr = strjoin(covStrings, '\n');
    classes_icnic_str = strjoin(arrayfun(@(x) sprintf('%d', x), classes_icnic, 'UniformOutput', false), ', ');

    % meta data
    filenamesStr = strjoin(files, ';\n');

    muStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), mu_features, 'UniformOutput', false), ', ');
    sigmaStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), sigma_features, 'UniformOutput', false), ', ');

    rowStrings = cell(size(band,2), 1);
    for i = 1:size(band,2)
        c_row = band{i}; 
        innerListStr = strjoin(arrayfun(@(x) sprintf('%d', x), c_row, 'UniformOutput', false), ', ');
        rowStrings{i} = sprintf('    - [%s]', innerListStr);
    end
    bands_str = strjoin(rowStrings, '\n');

    o_l_channels = string(channels_labels(o_l_idx));
    o_l_channels = "'" + o_l_channels + "'";
    o_l_channels = join(o_l_channels, ", ");
    o_l_str = strjoin(arrayfun(@(x) sprintf('%d', x), o_l_idx, 'UniformOutput', false), ', ');

    o_r_channels = string(channels_labels(o_r_idx));
    o_r_channels = "'" + o_r_channels + "'";
    o_r_channels = join(o_r_channels, ", ");
    o_r_str = strjoin(arrayfun(@(x) sprintf('%d', x), o_r_idx, 'UniformOutput', false), ', ');

    c_c_channels = string(channels_labels(c_c_idx));
    c_c_channels = "'" + c_c_channels + "'";
    c_c_channels = join(c_c_channels, ", ");
    c_c_str = strjoin(arrayfun(@(x) sprintf('%d', x), c_c_idx, 'UniformOutput', false), ', '); 

    c_l_channels = string(channels_labels(c_l_idx));
    c_l_channels = "'" + c_l_channels + "'";
    c_l_channels = join(c_l_channels, ", ");
    c_l_str = strjoin(arrayfun(@(x) sprintf('%d', x), c_l_idx, 'UniformOutput', false), ', '); 

    c_r_channels = string(channels_labels(c_r_idx));
    c_r_channels = "'" + c_r_channels + "'";
    c_r_channels = join(c_r_channels, ", ");
    c_r_str = strjoin(arrayfun(@(x) sprintf('%d', x), c_r_idx, 'UniformOutput', false), ', ');

    excl_channels = string(channels_labels(excluded_chs));
    excl_channels = "'" + excl_channels + "'";
    excl_channels = join(excl_channels, ", ");
    excl_str = strjoin(arrayfun(@(x) sprintf('%d', x), excluded_chs, 'UniformOutput', false), ', ');

    % build the yaml
    yamlContent = sprintf(['GmmModelCfg:\n' ...
                     '  name: "gmm_model"\n' ...
                     '  filenames: "%s"\n'...
                     '  params:\n' ...
                     '    central_idx: [%s]\n' ...
                     '    centrial: [%s]\n' ...
                     '    occipital_left_idx: [%s]\n' ...
                     '    occipital_left: [%s]\n' ...
                     '    occipital_right_idx: [%s]\n' ...
                     '    occipital_right: [%s]\n' ...
                     '    central_left_idx: [%s]\n' ...
                     '    central_left: [%s]\n' ...
                     '    central_right_idx: [%s]\n' ...
                     '    central_right: [%s]\n' ...
                     '    excluded_idx: [%s]\n' ...
                     '    excluded: [%s]\n' ...
                     '    mu: [%s]\n' ...
                     '    sigma: [%s]\n' ...
                     '    band: \n%s\n' ...
                     '    threshold_gmm_ic: %d\n' ...
                     '    threshold_gmm_ic_discarded: %d\n' ...
                     '  model_params:\n' ...
                     '    nfeatures: %d\n' ...
                     '    type: "%s"\n' ...
                     '    classes: [%s] # 0=NIC, 1=IC\n', ...
                     '    K: %d\n' ...
                     '    weights: [%s]\n' ...
                     '    means: \n%s\n' ...
                     '    covariances: \n%s\n'], ...
                     filenamesStr, ...
                     c_c_str, ...
                     c_c_channels, ...
                     o_l_str, ...
                     o_l_channels, ...
                     o_r_str, ...
                     o_r_channels, ...
                     c_l_str, ...
                     c_l_channels, ...
                     c_r_str, ...
                     c_r_channels, ...
                     excl_str, ...
                     excl_channels, ...
                     muStr, ...
                     sigmaStr, ...
                     bands_str, ...
                     threshold_gmm_ic, ...
                     0.4, ...
                     nfeatures, ...
                     type, ...
                     classes_icnic_str, ...
                     K, ...
                     weightsStr, ...
                     meansStr, ...
                     covariancesStr);

    fileID = fopen(save_path_gmm, 'w');
    fprintf(fileID, '%s', yamlContent);
    fclose(fileID);

    disp(['GMM model saved in ', save_path_gmm]);
end