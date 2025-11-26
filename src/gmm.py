#!/usr/bin/env python3

import yaml
import rospy
import numpy as np
from processing_cvsa.msg import eeg_power
from rosneuro_msgs.msg import NeuroOutput
from sklearn.mixture import GaussianMixture

class GMMClassifier:
    def __init__(self):
        rospy.init_node('gmm_classifier', anonymous=True)
        
        try:
            self.path_decoder = rospy.get_param('~path_gmm_model')
        except KeyError as e:
            rospy.logfatal(f"[GMM] Parametro mancante: {e}. Assicurati di lanciarlo con un launch file.")
            return
        conf = self.configure()
        
        if not conf:
            rospy.logfatal("[GMM] Erorr in the GMM configuration.")
            return
        else:
            rospy.loginfo("[GMM] GMM configurated correctly.")
        
        
        rospy.Subscriber('/cvsa/eeg_power', eeg_power, self.callback)
        self.pub = rospy.Publisher('/cvsa/neuroprediction/icnic', NeuroOutput, queue_size=10)

        rospy.spin()
        
    def configure(self):
        rospy.loginfo(f"[GMM] Loading GMM from: {self.path_decoder}")
        with open(self.path_decoder, 'r') as file:
            params = yaml.safe_load(file)['GmmModelCfg']['params']
            
        with open(self.path_decoder, 'r') as file:
            model_params = yaml.safe_load(file)['GmmModelCfg']['model_params']
            
        try:
            self.gmm_name = yaml.safe_load(open(self.path_decoder, 'r'))['GmmModelCfg']['name']

            # Load parameters for features extraction
            self.mu              = np.array(params['mu'])
            self.sigma           = np.array(params['sigma'])
            self.o_l             = np.sort(np.array(params['occipital_left_idx']) - 1)
            self.o_r             = np.sort(np.array(params['occipital_right_idx']) - 1)
            self.frontal         = np.sort(np.array(params['frontal_idx']) - 1)
            self.c_l             = np.sort(np.array(params['central_left_idx']) - 1)
            self.c_r             = np.sort(np.array(params['central_right_idx']) - 1)
            self.exclude_chs     = np.sort(np.array(params['excluded_idx']) - 1)
            self.nfeatures       = int(params['nfeatures'])
            self.bands_features  = np.array(params['band'])

            # Load parameters for the classifier
            self.K = int(model_params['K'])
            self.model = GaussianMixture(n_components=self.K, covariance_type='full')
            self.model.means_ = np.array(model_params['means'])
            self.model.weights_ = np.array(model_params['weights'])
            covariances = np.array(model_params['covariances'])
            self.model.covariances_ = covariances
            self.classes = np.array(model_params['classes'])
            self.type = model_params['type']

            self.model.precisions_cholesky_ = np.array(
                [np.linalg.cholesky(np.linalg.inv(cov)) for cov in covariances]
                )
        except KeyError as e:
            rospy.logwarn(f"[{self.gmm_name}] YAML file error parameter: {e}")
            return False
        except Exception as e:
            rospy.logwarn(f"[{self.gmm_name}]General error in the GMM model loading: {e}")
            return False
            
        return True


    def compute_sparsity_features(self, signal, nbands):
        sparsity = np.zeros(self.nfeatures)
        idx_sparsity = 0
        
        for idx_band in range(0, nbands):
            c_signal = signal[idx_band,:]
            # --- 1. Feature LI (Lateralization Index) ---
            if self.type == 'cvsa':
                P_left_window = np.mean(c_signal[self.o_l])
                P_right_window = np.mean(c_signal[self.o_r])
            elif self.type == 'mi':
                P_left_window = np.mean(c_signal[self.c_l])
                P_right_window = np.mean(c_signal[self.c_r])
            else:
                return

            denominator = P_right_window + P_left_window + np.finfo(float).eps
            LAP_history = (P_right_window - P_left_window) / denominator
            sparsity[idx_sparsity] = np.abs(LAP_history) # LI
            idx_sparsity += 1

            # --- 2. Feature GI (Gini * Occipital Power) ---
            all_chs = np.arange(len(c_signal))
            non_eog_chs = np.setdiff1d(all_chs, self.exclude_chs)
            global_mean = np.mean(c_signal[non_eog_chs])
            current_signal_normalized = c_signal - global_mean
            mean_roi = np.array([
                np.mean(current_signal_normalized[self.c_l]),
                np.mean(current_signal_normalized[self.c_r]),
                np.mean(current_signal_normalized[self.o_l]),
                np.mean(current_signal_normalized[self.o_r])
            ])
            mean_roi = np.abs(mean_roi)
            mean_roi_ordered = np.sort(mean_roi)
            n = len(mean_roi_ordered)
            total_sum = np.sum(mean_roi_ordered)
            if total_sum > 0:
                sum_roi_p = 0
                for i in range(n): 
                    sum_roi_p += (n - i) * mean_roi_ordered[i]

                gi = (1.0 / n) * (n + 1.0 - (2.0 * sum_roi_p) / total_sum)
            else:
                gi = 0

            sparsity[idx_sparsity] = gi # GI
            idx_sparsity += 1

            # --- 3. Feature GB (Global Brain Activity) ---
            sparsity[idx_sparsity] = global_mean # GB
            idx_sparsity += 1
        
        return sparsity

    def classify(self, dfet):
        if dfet is None:
            return 
        
        dfet_std = (dfet - self.mu) / self.sigma
        dfet_std_2d = dfet_std.reshape(1, -1)
        
        probabilities_soft = self.model.predict_proba(dfet_std_2d)[0]
        probabilities_hard = self.model.predict(dfet_std_2d)
        
        return [probabilities_soft, probabilities_hard]
        
    def callback(self, msg):
        data = msg.data
        nchannels = msg.nchannels
        nbands = msg.nbands
        all_bands = np.array(msg.bands).reshape(-1, 2)
        
        reshaped_data = np.array(data).reshape(nchannels, nbands)
        
        tmp = [] # [bands x channels]
        for i, c_band_features in enumerate(self.bands_features):
            for j, filter_band in enumerate(all_bands):
                if np.array_equal(c_band_features, filter_band):
                    tmp.append(reshaped_data[:, j])
                    break 
 
        dfet = self.compute_sparsity_features(np.array(tmp), nbands)
        
        [soft_proba, hard_prob] = self.classify(dfet)
        
        # publish the output
        output = NeuroOutput()
        output.header.stamp = rospy.Time.now()
        output.neuroheader.seq = msg.seq
        output.softpredict.data = soft_proba.tolist()
        output.hardpredict.data = hard_prob.tolist() 
        output.decoder.type = self.gmm_name
        output.decoder.path = self.path_decoder
        output.decoder.classes = self.classes.tolist()
        
        self.pub.publish(output)
        
   
if __name__ == '__main__':
    GMMClassifier()