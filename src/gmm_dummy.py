#!/usr/bin/env python3

import rospy
from processing_bci.msg import eeg_power
from rosneuro_msgs.msg import NeuroOutput

class DummyGMM:
    def __init__(self):
        rospy.init_node('dummy_gmm', anonymous=True)
        self.gmm_name = "gmm_model"
        self.classes = [1, 0]
        
        rospy.loginfo(f"[{self.gmm_name}] GMM configurated correctly.")

        rospy.Subscriber('/cvsa/eeg_power', eeg_power, self.callback)
        self.pub = rospy.Publisher('/cvsa/neuroprediction/icnic', NeuroOutput, queue_size=10)

        rospy.spin()
        
    def callback(self, msg):
        output = NeuroOutput()
        
        output.header.stamp = rospy.Time.now()
        output.neuroheader.seq = msg.seq
        
        # --- LOGICA DUMMY ---
        output.softpredict.data = [1.0, 0.0] 
        output.hardpredict.data = [1, 0] 
        output.decoder.type = self.gmm_name
        output.decoder.path = "N/A"
        output.decoder.classes = self.classes
        
        self.pub.publish(output)
        
   
if __name__ == '__main__':
    try:
        DummyGMM()
    except rospy.ROSInterruptException:
        pass