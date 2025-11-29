clear all; % close all;

addpath('/home/paolo/cvsa/ic_cvsa_ws/src/analysis_cvsa/equal_ros')

%% Initialization
DATAPAH = '/home/paolo/cvsa/ic_cvsa_ws/src/';
bands = [{[8 14]}];
bands_str = cellfun(@(x) sprintf('%d-%d', x(1), x(2)), bands, 'UniformOutput', false);
nbands = length(bands);
signals = cell(1, nbands);
artifacts = cell(1, nbands);
headers = cell(1, nbands);
for idx_band = 1:nbands
    headers{idx_band}.TYP = [];
    headers{idx_band}.DUR = [];
    headers{idx_band}.POS = [];
    signals{idx_band} = [];
    artifacts{idx_band} = [];
end
classes = [730 731];      
cf_event = 781;
fix_event = 786;
nchannels = 39;
nclasses = length(classes);
filterOrder = 4;
avg = 1;% 0.75;
threshold_gmm_ic = 0.7;

%% Load file
[filenames, pathname] = uigetfile('*.gdf', 'Select GDF Files', 'MultiSelect', 'on');
if ischar(filenames)
    filenames = {filenames};
end
subject = filenames{1}(1:2);
time_str = datestr(now, 'ddmmyyyy_HHMMSS');
gmm_file = ['gmm_' subject '_' time_str '.yaml'];
save_path_gmm = [DATAPAH, 'gmm_cvsa/cfg/' gmm_file];
save_path_qda_dataset = [DATAPAH 'qda_cvsa/create_qda/datasets/gmm/data_' subject '_' time_str '.mat'];

%% concatenate the files
nFiles = length(filenames);
for idx_file= 1: nFiles
    fullpath_file_shift = fullfile(pathname, filenames{idx_file});
    disp(['file (' num2str(idx_file) '/' num2str(nFiles)  '): ', filenames{idx_file}]);
    [c_signal,header] = sload(fullpath_file_shift);
    c_signal = c_signal(:,1:nchannels);
    channels_label = header.Label;
    sampleRate = header.SampleRate;


    disp('   [proc] power band');
    for idx_band = 1:nbands
        band = bands{idx_band};

        % for power band using hilbert transformation and artefact remotion -----------------------------------------------
        bufferSize = floor(avg*sampleRate);
        chunkSize = 32;
        eog.filterOrder = 4;
        eog.band = [1 7];
        eog.label = {'FP1', 'FP2', 'EOG'};
        eog.h_threshold = 85;
        eog.v_threshold = 85;
        muscle.filterOrder = 4;
        muscle.freq = 1; % remove antneuro problems
        muscle.threshold = 170;
        [signal_processed, header_processed] = processing_onlineROS_hilbert(c_signal, header, nchannels, bufferSize, filterOrder, band, chunkSize);
        artifact = artifact_rejection(c_signal, header, nchannels, bufferSize, chunkSize, eog, muscle);

        if all(subject == 'h8')
            signal_processed(:,23) = 0;
        end

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
        artifacts{1, idx_band} = cat(1, artifacts{1, idx_band}, artifact(:,:));
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
artifacts_data = nan(min_trial_data, nbands, ntrial); % data x bands x trial
for idx_band = 1:nbands
    c_signal = signals{idx_band};
    c_artifact = artifacts{idx_band};
    for trial = 1:ntrial
        c_start = trial_start(trial);
        c_end = trial_start(trial) + min_trial_data - 1;
        trial_data(:,idx_band,:,trial) = c_signal(c_start:c_end,:);
        artifacts_data(:,idx_band,trial) = c_artifact(c_start:c_end,:);
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
        tmp_art(:,:,idx_trial_class + idx_class - 1) = artifacts_data(:,:,idx_classes_trial(i, idx_class));
        trial_typ(idx_trial_class + idx_class - 1) = classes(idx_class);
    end
    i = i + 1;
end
trial_data = tmp_data; % samples x bands x channels x trials
artifacts_data = tmp_art;

%% compute sparsity
% define regions
nsparsity = 3;
sparsity = nan(min_trial_data, nbands, ntrial, nsparsity); % sample x band x trial x sparsity
o_l_ch = {'P3', 'O1', 'P5', 'P1', 'PO5', 'PO3', 'PO7'};
o_r_ch = {'P4', 'O2', 'P2', 'P6', 'PO4', 'PO6', 'PO8'};
c_l_ch = {'FC1', 'C3', 'CP1', 'FC3', 'C1', 'CP3'};
c_r_ch = {'FC2', 'C4', 'CP2', 'FC4', 'C2', 'CP4'};
excl_ch = {'FP1', 'FP2', 'EOG'};

[~, o_l] = ismember(o_l_ch, channels_label);
[~, o_r] = ismember(o_r_ch, channels_label);
[~, c_l] = ismember(c_l_ch, channels_label);
[~, c_r] = ismember(c_r_ch, channels_label);
[~, excl_chs] = ismember(excl_ch, channels_label);

type = 'cvsa';

for c = 1:ntrial
    c_data = squeeze(trial_data(:,:,:,c)); % samples x band x channels

    for sample = 1:min_trial_data
        c_sample = squeeze(c_data(sample,:,:)); % bands x channels

        for idx_band = 1:nbands
            tmp = squeeze(c_sample(idx_band,:)); % 1 x channels

            [sparsity(sample, idx_band, c,:), ~] = compute_features_icnic(tmp, type, o_l, o_r, c_l, c_r, excl_chs);
        end
    end
end

%% ----------------- gmm -----------------
% update to work with subbands -> tesista
K = 2;
choosen_band = 1;

sparsity_cf = squeeze(sparsity(minDurFix+minDurCue+1:end, choosen_band,:,:));
artifacts_cf = squeeze(artifacts_data(minDurFix+minDurCue+1:end, choosen_band,:));

% z-score -> train and use the mu and var also for the test
data_3D = sparsity_cf(:, :, :);
data_2D = reshape(data_3D, size(data_3D, 1) * size(data_3D,2), size(data_3D,3));
artefact_2D = artifacts_cf(:, :);
artefact_1D = reshape(artefact_2D, size(artefact_2D, 1) * size(artefact_2D,2), 1);
data_2D_noArtif = data_2D(artefact_1D == 0,:);
mu_features = mean(data_2D_noArtif, 1);
sigma_features = std(data_2D_noArtif, 0, 1);
sigma_features(sigma_features == 0) = eps;
data_2D_noArtif = (data_2D_noArtif - mu_features) ./ sigma_features;
data_standardized_2D = (data_2D - mu_features) ./ sigma_features;
sparsity_cf = reshape(data_standardized_2D, size(data_3D, 1), ntrial, size(data_3D,3));

disp('Esecuzione di GMM sui dati di training globali...');
options = statset('MaxIter', 1000, 'Display', 'off');

gmm_model = fitgmdist(data_2D_noArtif, K, 'Options', options, 'Replicates', 50);

[~, sort_order] = sort(gmm_model.mu(:, 1), 'descend');
idx_ic = sort_order(1);  % Index cluster IC
idx_nic = sort_order(2); 
classes_icnic = zeros(1,2);
classes_icnic(idx_ic) = 1;

% Crea le etichette finali
labels_gmm = {'IC', 'NIC'};

train_gmm = nan(ntrial * size(sparsity_cf, 1), nsparsity);
for c = 1:ntrial
    train_gmm((c-1)*size(sparsity_cf, 1) + 1: c * size(sparsity_cf, 1),:) = sparsity_cf(:,c,:);
end
P_soft = posterior(gmm_model, train_gmm);
cluster_labels = nan(size(sparsity_cf, 1), ntrial); % contains the prob to be ic
for c = 1:ntrial
    cluster_labels(:,c) = P_soft((c-1)*size(sparsity_cf, 1) + 1: c * size(sparsity_cf, 1),idx_ic);
end

fprintf('Mappatura: Cluster GMM %d -> "ic", Cluster GMM %d -> "nic"\n', idx_ic, idx_nic);

% plot the C
disp('centroids: ')
disp(gmm_model.mu)

%% save the gmm
save_gmm(gmm_model, mu_features, sigma_features, filenames, save_path_gmm, o_l, o_r, c_l, c_r, excl_chs, channels_label, bands(choosen_band), classes_icnic, threshold_gmm_ic, type)

%% extract and save data for the QDA
data = squeeze(trial_data(minDurCue+minDurFix+1:end,choosen_band,:,:)); % take just the 8-14 band
nsamples = size(data,1);
X = [];
y = [];
for idx_trial =  1:ntrial
    for idx_sample = 1:nsamples
        if artifacts_cf(idx_sample,idx_trial) == 0 % no artifact
            if cluster_labels(idx_sample, idx_trial) >= threshold_gmm_ic % IC state
                X = [X; data(idx_sample,:,idx_trial)];
                y = [y; trial_typ(idx_trial)];
            end
        end
    end
end
occipital = {'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'}; [~, ch_occipital] = ismember(occipital, channels_label);
bands = bands(choosen_band);
save(save_path_qda_dataset, 'X', 'y', 'gmm_file', 'classes', 'ch_occipital', 'occipital', 'filenames', 'bands')
disp(['QDA model saved in ', save_path_qda_dataset]);

%% ----------- FUNCTIONS --------
% save gmm
function save_gmm(gmm_model, mu_features, sigma_features, files, save_path_gmm, o_l_idx, o_r_idx, c_l_idx, c_r_idx, excluded_chs, channels_labels, band, classes_icnic, threshold_gmm_ic, type)
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
                     '    nfeatures: %d\n' ...
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
                     '  model_params:\n' ...
                     '    type: "%s"\n' ...
                     '    classes: [%s] # 0=NIC, 1=IC\n', ...
                     '    K: %d\n' ...
                     '    weights: [%s]\n' ...
                     '    means: \n%s\n' ...
                     '    covariances: \n%s\n'], ...
                     filenamesStr, ...
                     nfeatures, ...
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