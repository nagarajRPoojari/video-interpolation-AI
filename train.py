import tensorflow as tf
import argparse
from slomo.model import SlowMo
from losses import Losses
from dataset_loader  import *
from pathlib import Path
import datetime

def train(data_dir:str , model_dir:str , log_dir:Path, batch_size:int == 32 , epochs:int== 1):
    tf.keras.backend.clear_session()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    
    
    data_dir = Path(data_dir)
    train_ds = load_dataset(data_dir / "train", batch_size)
    len_train = tf.data.experimental.cardinality(train_ds).numpy()
    progbar = tf.keras.utils.Progbar(len_train)
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    chckpnt_dir = model_dir / "chckpnt"
    chckpnt_dir.mkdir(parents=True, exist_ok=True)
    
    
    model = SlowMo(num_frames=12)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, str(chckpnt_dir), max_to_keep=3)
    num_epochs = 1
    
    
    
    loss_obj= Losses()
    epochs=2
    for epoch in range(epochs):
        print('epoch : ', epoch)


        
        for step, frames in enumerate(train_ds):
            print(step)
            inputs, targets = frames
            loss_values = train_step(
                model, inputs, targets, optimizer, loss_obj
            )
            progbar.update(
                step + 1,
                [
                    ("total_loss", loss_values[0]),
                    ("rec_loss", loss_values[1]),
                    ("perc_loss", loss_values[2]),
                    ("smooth_loss", loss_values[3]),
                    ("warping_loss", loss_values[4])
                ],
            )
    
    
    
def train_step(model , inputs, targets , optimizer , loss_obj):
    
    with tf.GradientTape() as tape :
        pred, loss= model(inputs )
        loss= loss_obj.compute_losses(
            pred, loss, inputs, targets
        )
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
    
    




def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="path to dataset folder", dest="data_dir")
    parser.add_argument("--model", help="path where to save model", required=True)
    parser.add_argument("--epochs", help="number of epochs", default=40, type=int)
    parser.add_argument(
        "--batch_size", help="size of the batch", default=7, type=int,
    )
    return parser.parse_args()

def main():

    log_dir = Path('./')
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir / current_time / "train"
   # train_log_dir.mkdir(parents=True, exist_ok=True)

    args = arg_parser()
    train(args.data_dir, args.model, train_log_dir, args.epochs, args.batch_size)
    
if __name__ == "__main__":
    main()
