import tensorflow as tf
from tensorflow.keras.layers import Conv2D , UpSampling2D  , LeakyReLU , AveragePooling2D
from tensorflow.keras.activations import sigmoid
import tensorflow_addons as tfa


class Encoder(tf.keras.layers.Layer):
    def __init__(self, filters , kernel_size , **kwargs):
        super(Encoder ,self).__init__(**kwargs)
        self.filters= filters
        self.kernel_size= kernel_size
        
    def build(self,input_shape):
        self.conv1= Conv2D(
            filters= self.filters,
            kernel_size= self.kernel_size,
            strides= 1,
            padding="same"
        )
        self.conv2= Conv2D(
            filters= self.filters,
            kernel_size= self.kernel_size,
            strides= 1,
            padding="same"
        )
        self.avg_pooling= AveragePooling2D()
        self.leaky_relu= LeakyReLU(alpha=0.1)
    
    def call(self, inputs, **kwargs):
        x= self.avg_pooling(inputs)
        x= self.conv1(x)
        x= self.leaky_relu(x)
        x= self.conv2(x)
        x= self.leaky_relu(x)
        return x

   
class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=3, strides=1, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=3, strides=1, padding="same"
        )
        self.interpolation = tf.keras.layers.UpSampling2D(interpolation="bilinear")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs, **kwargs):
        x, skip = inputs
        x = self.interpolation(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x_delta = skip.shape[1] - x.shape[1]
        y_delta = skip.shape[2] - x.shape[2]
        x = tf.pad(
            x, tf.convert_to_tensor([[0, 0], [0, x_delta], [0, y_delta], [0, 0]])
        )

        x = tf.keras.layers.Concatenate(axis=3)([x, skip])
        x = self.conv2(x)
        x = self.leaky_relu(x)
        return x



class Unet(tf.keras.layers.Layer):
    def __init__(self, output_depth, name="UNet", **kwargs):
        super(Unet, self).__init__(name=name, **kwargs)
        self.out_filters = output_depth

    def build(self, input_shape):
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=7, strides=1, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=7, strides=1, padding="same"
        )
        self.encoder1 = Encoder(64, 5)
        self.encoder2 = Encoder(128, 5)
        self.encoder3 = Encoder(256, 3)
        self.encoder4 = Encoder(512, 3)
        self.encoder5 = Encoder(512, 3)
        self.decoder1 = Decoder(512)
        self.decoder2 = Decoder(256)
        self.decoder3 = Decoder(128)
        self.decoder4 = Decoder(64)
        self.decoder5 = Decoder(32)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.out_filters, kernel_size=3, strides=1, padding="same"
        )

    def call(self, inputs, **kwargs):
        x_enc = self.conv1(inputs)
        x_enc = self.leaky_relu(x_enc)
        skip = self.conv2(x_enc)
        skip1 = self.leaky_relu(skip)
        skip2 = self.encoder1(skip1)
        skip3 = self.encoder2(skip2)
        skip4 = self.encoder3(skip3)
        skip5 = self.encoder4(skip4)
        x_enc = self.encoder5(skip5)
        x_dec = self.decoder1([x_enc, skip5])
        x_dec = self.decoder2([x_dec, skip4])
        x_dec = self.decoder3([x_dec, skip3])
        x_dec = self.decoder4([x_dec, skip2])
        x_dec = self.decoder5([x_dec, skip1])
        x_dec = self.conv3(x_dec)
        x_dec = self.leaky_relu(x_dec)
        return x_dec
    

  
class Backwarp(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Backwarp,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.backwarp_layer = tfa.image.dense_image_warp
        
    def call(self, inputs):
        image , flow = inputs
        image = self.backwarp_layer(image , flow)
        return image


class FlowInterpreter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FlowInterpreter, self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.flow_interpretation_layer = Unet(output_depth=5)
        self.backwarp_layer_0= Backwarp()
        self.backwarp_layer_1= Backwarp()
        
        
    def call(self,inputs):
        frame_0, frame_1 , flow_0_1 , flow_1_0 , t_indeces= inputs
        t0_coeff= (-(1-t_indeces) )* t_indeces
        t1_coeff= (1-t_indeces)*(1-t_indeces)
        print(t1_coeff.shape ,flow_1_0.shape  , t0_coeff.shape ,flow_0_1.shape,(t1_coeff * flow_1_0).shape , (t0_coeff * flow_0_1).shape)
        flow_t_1= (t1_coeff * flow_1_0) - (t0_coeff * flow_0_1)
        
        t1_coeff= t_indeces * t_indeces
        
        flow_t_0 = (t0_coeff * flow_0_1) + (t1_coeff * flow_1_0)
        print('r',frame_0.shape, flow_0_1.shape)
        approx_t_0 = self.backwarp_layer_0([frame_0 , flow_t_0])
        approx_t_1 = self.backwarp_layer_1([frame_1 , flow_t_1])
        
        flow_interpreter_input= tf.concat([
            frame_0, frame_1, flow_0_1, flow_1_0 , flow_t_1, flow_t_0, approx_t_1, approx_t_0
        ], axis=3)
        flow_interpreter_output= self.flow_interpretation_layer(flow_interpreter_input)
        
        delta_t_0 = flow_interpreter_output[:,:,:,:2]
        delta_t_1 = flow_interpreter_output[:,:,:,2:4]
        
        visibility_t_0 = sigmoid(flow_interpreter_output[:, :, :, 4:5])
        visibility_t_1 = 1 - visibility_t_0

        f_t0 = flow_t_0 + delta_t_0
        f_t1 = flow_t_1 + delta_t_1

        return f_t0, visibility_t_0, f_t1, visibility_t_1, approx_t_0, approx_t_1
        

class Output(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Output, self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.backwarp_0= Backwarp()
        self.backwarp_0= Backwarp()
        
    def call(self, inputs):
        frame_0, flow_t_0, visibility_t_0 , frame_1, flow_t_1, visibility_t_1 , t_indeces= inputs
        
        frame_t_0 = self.backwarp_0([frame_0, flow_t_0])
        frame_t_1 = self.backwarp_0([frame_1, flow_t_1])
        
        z= (1- t_indeces)* visibility_t_0 + (t_indeces * visibility_t_1) + 1e-12 # to avoid division by zero
        final_frame= ((1-t_indeces)* visibility_t_0)* frame_t_0 + (t_indeces * visibility_t_1) * frame_t_1
        return tf.divide(final_frame , z)