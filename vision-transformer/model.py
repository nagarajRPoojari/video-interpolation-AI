import tensorflow as tf
import numpy as np

from base_layers import *
from decoder import *
from encoder import *



class Transformer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self,input_shape):
        self.encoder=Encoder()
        self.decoder=Decoder()
        super(Transformer, self).build(input_shape)
        
    def call(self,x):
        encoder_outputs=self.encoder(x)
        x=self.decoder(encoder_outputs)
        
        return x