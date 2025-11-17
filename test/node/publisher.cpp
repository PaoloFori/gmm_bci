#include <ros/ros.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <eigen3/Eigen/Dense> 
#include <vector>
#include <string>
#include <processing_cvsa/eeg_power.h> 
#include "gmm_cvsa/utils.hpp" 

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_publisher_csv");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~"); 

    std::string topic = "/cvsa/eeg_power";
    std::string csv_filename;
    double sample_rate;

    if (!private_nh.getParam("csv_file", csv_filename)) {
        ROS_ERROR("Parametro 'csv_file' non impostato! Specificare il percorso del CSV.");
        return 1;
    }
    if (!private_nh.getParam("sample_rate", sample_rate)) {
        ROS_ERROR("Parametro 'sample_rate' non impostato! (es. 16)");
        return 1;
    }

    ROS_INFO("Loading dati from: %s", csv_filename.c_str());
    Eigen::MatrixXd full_data;
    try {
        full_data = readCSV<double>(csv_filename); 
    } catch (const std::exception& e) {
        ROS_ERROR("Error during the CSV reading: %s", e.what());
        return 1;
    }
    
    int n_channels = full_data.cols();
    int total_samples = full_data.rows();
    ROS_INFO("Loaded data: %d samples x %d channels.", total_samples, n_channels);

    ros::Publisher pub = nh.advertise<processing_cvsa::eeg_power>(topic, 10);
    ros::Rate loop_rate(sample_rate);

    ROS_INFO("Waiting for a  subscriber on topic '%s'...", topic.c_str());
    while (ros::ok() && pub.getNumSubscribers() == 0) {
        ros::Duration(0.5).sleep(); 
        ROS_INFO_THROTTLE(5.0, "Still waiting...");
    }
    ROS_INFO("Subscriber connected. Start pubblication.");

    int current_sample = 0;
    while (ros::ok()) {
        
        if (current_sample > total_samples) {
            ROS_INFO("Fine del file CSV.");
            break;
        }

        Eigen::MatrixXd c_data = full_data.block(current_sample, 0, 1, n_channels);

        processing_cvsa::eeg_power msg;
        msg.header.stamp = ros::Time::now();
        msg.nchannels = n_channels;
        msg.nbands = 1;
        msg.seq = current_sample;
        msg.bands = {8,14};
        
        
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c_data_float;
        c_data_float = c_data.cast<float>();
        msg.data.assign(c_data_float.data(), c_data_float.data() + c_data_float.size());
        pub.publish(msg);
        
        current_sample += 1;

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}