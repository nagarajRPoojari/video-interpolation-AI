import tensorflow as tf
import numpy as np
from model.layers import *

class SlowMo(tf.keras.Model):
    def __init__(self, num_frames=12, **kwargs):
        super(SlowMo, self).__init__(**kwargs)
        self.time_slices = tf.constant(np.linspace(0,1 , num_frames))
        self.flow_comp_layer = Unet(4)
        self.flow_intp_layer = FlowInterpreter()
        self.output_layer = Output()
        self.backwarp_layer_0 = Backwarp()
        self.backwarp_layer_1 = Backwarp()
    
    def call(self, inputs):
        frame_0, frame_1, frame_t= inputs
        
        t_indices = tf.gather(self.time_slices, frame_t)
        t_indices = tf.cast(t_indices, dtype=tf.float32)
        t_indices = t_indices[:, tf.newaxis, tf.newaxis, tf.newaxis]


        flow_input = tf.concat([frame_0, frame_1], axis=3)
        flow_out = self.flow_comp_layer(flow_input)


        flow_0_1, flow_1_0 = flow_out[:, :, :, :2], flow_out[:, :, :, 2:]
        optical_input = [frame_0, frame_1, flow_0_1, flow_1_0, t_indices]
        
        print('rr',flow_out.shape ,frame_0.shape , flow_0_1.shape ,flow_1_0.shape )
        
        
        flow_t_0, visibility_t_0, flow_t_1, visibility_t_1, frame_t_0, frame_t_1 = self.flow_intp_layer(optical_input)


        preds_input = [frame_0, flow_t_0, visibility_t_0, frame_1, flow_t_1, visibility_t_1, t_indices]
        predictions = self.output_layer(preds_input)

        warp0 = self.backwarp_layer_0([frame_1, flow_0_1])
        warp1 = self.backwarp_layer_1([frame_0, flow_1_0])
        losses_output = [flow_0_1, flow_1_0, warp0, warp1, frame_t_0, frame_t_1]
        return predictions, losses_output