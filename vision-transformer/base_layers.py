import tensorflow as tf
import numpy as np


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,num_heads=2,key_dim=8,attention_axes=(2,3,4)):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,attention_axes=attention_axes)
    def call(self,x):
        return self.mha(query=x, key=x, value=x)
    
    
class MLP(tf.keras.layers.Layer):
    def __init__(self,input_dims, output_dims, mlp_ratio=1):
        super().__init__()
        self.input_dims=input_dims
        self.output_dims=output_dims
        self.mlp_ratio=mlp_ratio
        
    def build(self, input_shape):
        neuron_count=int(self.mlp_ratio * input_shape[-1])
        self.dense1=tf.keras.layers.Dense(neuron_count,activation='relu')
        self.dense2=tf.keras.layers.Dense(self.output_dims,activation='relu')
        
    def call(self,x):
        x=self.dense1(x)
        x=self.dense2(x)
        return x
    
    
class WindowMSA(tf.keras.layers.Layer):
    def __init__(self,num_heads=2,key_dim=8):
        super().__init__()
        self.num_heads=num_heads
        self.key_dim=key_dim

        
    def build(self,input_shape):
        self.layer_norm1=tf.keras.layers.LayerNormalization()
        self.layer_norm2=tf.keras.layers.LayerNormalization()
        self.spatial_msa=BaseAttention(self.num_heads, self.key_dim)
        self.temporal_msa=BaseAttention(self.num_heads, self.key_dim)
        self.mlp=MLP(input_shape[-1],input_shape[-1])
        
    def call(self,x):
        B,T,H,W,C=x.shape
        MH,MW=H//16,W//24

        self.temporal_window=(T,1,1)
        self.spatial_window=(1,MH,MW)
        
        
        x=self.layer_norm1(x)
        windows=normal_window_partition(x,self.spatial_window)
        
        windows=tf.transpose(windows, (1,0,2,3,4,5)) 
        x=self.spatial_msa(windows)
        x=tf.transpose(x, (1,0,2,3,4,5))
        

        x=reverse_window_partition(x,(B,T,H,W,C), self.spatial_window )
        windows=normal_window_partition(x, self.temporal_window)
        windows=tf.transpose(windows, (1,0,2,3,4,5))
        x=self.temporal_msa(windows)
        x=tf.transpose(x, (1,0,2,3,4,5))
        
        x=reverse_window_partition(x,(B,T,H,W,C),self.temporal_window)
        x=self.layer_norm2(x)
        x=self.mlp(x)
        return x   