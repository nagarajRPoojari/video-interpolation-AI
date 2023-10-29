import tensorflow as tf
import numpy as np

from base_layers import *


class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self,input_shape):
        self.upsample0=tf.keras.layers.UpSampling3D(size=2)
        self.upsample1=tf.keras.layers.UpSampling3D(size=2)
        self.upsample2=tf.keras.layers.UpSampling3D(size=2)
        self.upsample3=tf.keras.layers.UpSampling3D(size=2)
        self.add=tf.keras.layers.Add()
    
    def call(self,x):
        x0,x1,x2,x3=x
        
        y0=self.upsample0(x3)
        y0=self.add(y0,x2)
        
        y1=self.upsample0(y0)
        y1=self.add(y1,x1)
        
        y2=self.upsample0(y1)
        y2=self.add(y2,x0)

        y3=self.upsample0(y2)
        
        return y3