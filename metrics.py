import tensorflow as tf


@tf.function
def compute_psnr(frames_t, preds_t):
    return tf.image.psnr(frames_t, preds_t, max_val=1.0)


@tf.function
def compute_ssim(frames_t, preds_t):
    return tf.image.ssim(frames_t, preds_t, max_val=1.0)


@tf.function
def compute_metrics(frames_t, predictions):
    psnr = compute_psnr(frames_t, predictions)
    ssim = compute_ssim(frames_t, predictions)
    return psnr, ssim