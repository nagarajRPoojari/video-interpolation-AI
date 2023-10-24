import tensorflow as tf
import numpy as np

class Losses:
    def __init__(self):
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
        self.vgg16 = tf.keras.Model(
            model.inputs, model.get_layer("block4_conv3").output, trainable=False
        )

    @tf.function
    def reconstruction_loss(self, y_true, y_pred):
        return self.mae(y_true, y_pred)

    @tf.function
    def perceptual_loss(self, y_true, y_pred):
        y_true = self.extract_feat(self.vgg16, y_true)
        y_pred = self.extract_feat(self.vgg16, y_pred)
        return self.mse(y_true, y_pred)

    @tf.function
    def extract_feat(self, feat_extractor, inputs):

        feats = inputs
        for layer in feat_extractor.layers:
            feats = layer(feats)
        return feats

    @tf.function
    def warping_loss(self, frame_0, frame_t, frame_1, backwarp_frames):
        return (
            self.mae(frame_0, backwarp_frames[0])
            + self.mae(frame_1, backwarp_frames[1])
            + self.mae(frame_t, backwarp_frames[2])
            + self.mae(frame_t, backwarp_frames[3])
        )

    @tf.function
    def smoothness_loss(self, f_01, f_10):
        delta_f_01 = self._compute_delta(f_01)
        delta_f_10 = self._compute_delta(f_10)
        return delta_f_01 + delta_f_10

    @tf.function
    def _compute_delta(self, frame):
        x = tf.reduce_mean(tf.abs(frame[:, 1:, :, :] - frame[:, :-1, :, :]))
        y = tf.reduce_mean(tf.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :]))
        return x + y

    @tf.function
    def compute_losses(self, predictions, loss_values, inputs, frames_t):
        frames_0, frames_1, _ = inputs
        f_01, f_10 = loss_values[:2]
        backwarp_frames = loss_values[2:]
        rec_loss = self.reconstruction_loss(frames_t, predictions)
        perc_loss = self.perceptual_loss(frames_t, predictions)
        smooth_loss = self.smoothness_loss(f_01, f_10)
        warp_loss = self.warping_loss(frames_0, frames_t, frames_1, backwarp_frames)
        REC_LOSS = 0.8
        PERCEP_LOSS = 0.005
        WRAP_LOSS =0.4
        SMOOTH_LOSS =1
        total_loss = (
            REC_LOSS * rec_loss
            + PERCEP_LOSS * perc_loss
            + WRAP_LOSS * warp_loss
            + SMOOTH_LOSS * smooth_loss
        )
        return total_loss, rec_loss, perc_loss, smooth_loss, warp_loss
