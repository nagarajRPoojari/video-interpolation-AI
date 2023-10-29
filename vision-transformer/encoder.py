import tensorflow as tf
import numpy as np

from base_layers import *


class ShallowEmbedding(tf.keras.layers.Layer):
    def __init__(self,d_model=8):
        super().__init__()
        self.d_model=d_model
        
    def build(self,input_shape):
        self.embeddings = tf.keras.layers.Dense(self.d_model , activation='relu')
        
    def call(self, x):
        return self.embeddings(x)
    

class DeepEmbedding(tf.keras.layers.Layer):
    def __init__(self,nf):
        super().__init__()
        self.nf=nf
        
    def build(self, input_shape):
        self.down0 = tf.keras.layers.Conv3D(filters=self.nf[-1], kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='same')
        self.down1 = tf.keras.layers.Conv3D(filters=self.nf[-2], kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='same')
        self.down2 = tf.keras.layers.Conv3D(filters=self.nf[-3], kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='same')
        self.down3 = tf.keras.layers.Conv3D(filters=self.nf[-4], kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='same')
        
        self.sts0=SepSTSBock()
        self.sts1=SepSTSBock()
        self.sts2=SepSTSBock()
        self.sts3=SepSTSBock()
        
        
        
    def call(self, x):
        
        x0=self.down0(x)

        
        x0=self.sts0(x0)
        
        x0=tf.squeeze(x0)
        x1=self.down1(x0)
        x1=self.sts1(x1)
        
        x1=tf.squeeze(x1)
        x2=self.down2(x1)
        x2=self.sts2(x2)
        
        x2=tf.squeeze(x2)
        x3=self.down3(x2)
        x3=self.sts3(x3)
        
        return x0,x1,x2,x3
    
    
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.nf=[1, 1, 1, 1]
        
    def build(self,input_shape):
        self.shallow=ShallowEmbedding()
        self.deep=DeepEmbedding(self.nf)
        
    def call(self,x):
        x=self.shallow(x)
        x=self.deep(x)
        return x