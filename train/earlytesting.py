import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from testprocess import *
from sklearn.preprocessing import RobustScaler
import argparse

gpu_indices = [7]  # Replace with the actual GPU indices you want to use

# Get the list of available GPUs
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    # Set the desired GPU devices
    visible_devices = [physical_devices[i] for i in gpu_indices]
    tf.config.experimental.set_visible_devices(visible_devices, 'GPU')

    # Allow memory growth for each GPU to avoid allocation errors
    for device in visible_devices:
        tf.config.experimental.set_memory_growth(device, True)
    

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--n_runs', type=int, default=200)

args = parser.parse_args()

model_type = args.model_type
batch_size = args.batch_size
n_runs = args.n_runs

df = pd.read_csv('../image_ds.csv')
redshifts = np.array(df['z'])

datasize = df.shape[0]

grz_imgs = np.memmap('../grz.dat', dtype='float32',
                     mode='r', shape=(datasize, 64, 64, 3))
w1w2_imgs = np.memmap('../w1w2.dat', dtype='float32',
                      mode='r', shape=(datasize, 32, 32, 2))

test_idx = np.load('test_idx_123.npy')

test_z = redshifts[test_idx]
test_grz_imgs = grz_imgs[test_idx]
test_w1w2_imgs = w1w2_imgs[test_idx]

# scaler_grz = RobustScaler()
# scaler_w1w2 = RobustScaler()

# test_grz_imgs = scaler_grz.fit_transform(test_grz_imgs.reshape(-1, test_grz_imgs.shape[-1])).reshape(test_grz_imgs.shape)
# test_w1w2_imgs = scaler_w1w2.fit_transform(test_w1w2_imgs.reshape(-1, test_w1w2_imgs.shape[-1])).reshape(test_w1w2_imgs.shape)

test_data = [test_grz_imgs, test_w1w2_imgs]

strategy = tf.distribute.MirroredStrategy()
print('Number of devices:{}'.format(strategy.num_replicas_in_sync))

if model_type == 'point':

    from model_point import *
    
    base_dir = 'CNN_point0/epochs'
    model_dir = os.path.join(base_dir, 'epoch_59')
    saved_model_name = model_dir
    # saved_model_name = 'model_cnn_point'
    # saved_model_name = os.path.join(model_dir, saved_model_name)
    
    with strategy.scope():
        model = ResNet18()

    prediction_point(test_data, test_z,
                    base_dir, model, 
                    saved_model_name, model_name='point')
    
elif model_type == 'mnf':
    print(f'Model: {model_type}')
    from model import *
    
    base_dir = 'CNN_mnf'
    model_dir = os.path.join(base_dir, 'CNN_model')
    saved_model_name = 'model_cnn_mnf.tf'
    saved_model_name = os.path.join(model_dir, saved_model_name)
    base_model_dir = 'CNN_base/CNN_model/model_cnn_point'
    base = base_model(base_model_dir)
    model = MNF_model(base)
    
    prediction_bnn(test_data, test_z, 
                   base_dir, model, 
                   saved_model_name, model_name=model_type,
                   n_runs=n_runs)
    
elif model_type == 'dropout':
    print(f'Model: {model_type}')
    from model import *
    
    base_dir = 'CNN_dropout'
    model_dir = os.path.join(base_dir, 'CNN_model')
    saved_model_name = 'model_cnn_dropout.tf'
    saved_model_name = os.path.join(model_dir, saved_model_name)
    base_model_dir = 'CNN_base/CNN_model/model_cnn_point'
    base = base_model(base_model_dir)
    model = Dropout_model(base, dropout_rate=0.01)
    
    prediction_bnn(test_data, test_z, 
                   base_dir, model, 
                   saved_model_name, model_name=model_type,
                   n_runs=n_runs)

else:
    print('wrong model!')