import tensorflow as tf
import numpy as np
from data_loading import *
from testprocess import *
import argparse

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass
        
gpu_indices = [0, 1, 5, 6]  # Replace with the actual GPU indices you want to use
#gpu_indices = [5, 7]

# Get the list of available GPUs
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    # Set the desired GPU devices
    visible_devices = [physical_devices[i] for i in gpu_indices]
    tf.config.experimental.set_visible_devices(visible_devices, 'GPU')

    # Allow memory growth for each GPU to avoid allocation errors
    for device in visible_devices:
        tf.config.experimental.set_memory_growth(device, True)

    

# tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices:{}'.format(strategy.num_replicas_in_sync))

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--n_runs', type=int, default=200)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--exist_model', type=str, default=None)

args = parser.parse_args()

target = args.target
model_type = args.model_type
batch_size = args.batch_size
epochs = args.epochs
n_runs = args.n_runs
dropout_rate = args.dropout_rate
exist_model = args.exist_model

# assert os.path.exists(exist_model), 'model do not exist'

print('---logging---')
print('target: ', target)
print('model_type: ', model_type)
print('epochs: ', epochs)
print('batch_size: ', batch_size)
if model_type != 'point':
    print('n_runs: ', n_runs)
    if model_type == 'dropout':
        print('dropout_rate: ', dropout_rate)
print('exist_model', exist_model)
print('---end logging---')

base_dir = '..'

with_photometry = False
if target == 'ELG':
    with_photometry = False


train_dataset, val_dataset, train_size, test_data, test_labels = dataset_loading(
    base_dir, batch_size=batch_size, target=target, with_photometry=with_photometry)

if model_type == 'point':

    from model_point import *

    base_dir = f'CNN_point_{target}'
    os.makedirs(base_dir, exist_ok=True)
    model_dir = os.path.join(base_dir, 'CNN_model')
    saved_model_name = 'model_cnn_point'
    saved_model_name = os.path.join(model_dir, saved_model_name)

    mcp = tf.keras.callbacks.ModelCheckpoint(saved_model_name,
                                             monitor='val_loss', mode='min', save_best_only=True)
    oee = OutputEveryEpoch(test_data, test_labels, base_dir)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = ResNet18(with_photometry=with_photometry)
        model.summary()
        


    #model = ResNet18()
    #model.summary()

    # model.load_weights('CNN_point/CNN_model/model_cnn_point')
    
    os.makedirs(f'CNN_point_{target}/epochs', exist_ok=True)

    his = model.fit(train_dataset, epochs=epochs, verbose=2,
                    validation_data=val_dataset, callbacks=[mcp, oee])

    plot_his(his, base_dir, model_name=model_type)
    prediction_point(test_data, test_labels, base_dir,
                     model, saved_model_name, model_name=model_type)

elif model_type == 'flipout':

    from model import *

    base_dir = 'CNN_flipout'
    os.makedirs(base_dir, exist_ok=True)
    model_dir = os.path.join(base_dir, 'CNN_model')
    saved_model_name = 'model_cnn_flipout.tf'
    saved_model_name = os.path.join(model_dir, saved_model_name)

    mcp = tf.keras.callbacks.ModelCheckpoint(saved_model_name,
                                             monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    base_model_dir = 'CNN_point0/CNN_model/model_cnn_point'
    base = base_model(base_model_dir)
    model = Flipout_model(base, train_size=train_size)
    model.summary()

    his = model.fit(train_dataset, epochs=epochs, verbose=2,
                    validation_data=val_dataset, callbacks=[mcp])
    plot_his(his, base_dir, model_name=model_type)
    prediction_bnn(test_data, test_labels, base_dir,
                   model, saved_model_name, model_name=model_type, n_runs=n_runs)
    
elif model_type == 'mnf':
    
    from model import *

    with strategy.scope():
      
        base_dir = f'CNN_mnf_{target}'
        os.makedirs(base_dir, exist_ok=True)
        model_dir = os.path.join(base_dir, 'CNN_model')
        saved_model_name = 'model_cnn_mnf.tf'
        saved_model_name = os.path.join(model_dir, saved_model_name)

        mcp = tf.keras.callbacks.ModelCheckpoint(saved_model_name,
                                                monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
        base_model_dir = f'CNN_point_{target}_0/CNN_model/model_cnn_point'
        base = base_model(base_model_dir)
        model = MNF_model(base)
        
        if exist_model is not None:
            model.load_weights(exist_model)

    his = model.fit(train_dataset, epochs=epochs, verbose=2,
                    validation_data=val_dataset, callbacks=[mcp])
    plot_his(his, base_dir, model_name=model_type)
    prediction_bnn(test_data, test_labels, base_dir,
                   model, saved_model_name, model_name=model_type, n_runs=n_runs)
    

elif model_type == 'dropout':

    from model import *
    
    with strategy.scope():
        base_dir = f'CNN_dropout_{target}'
        os.makedirs(base_dir, exist_ok=True)
        model_dir = os.path.join(base_dir, 'CNN_model')
        saved_model_name = 'model_cnn_dropout.tf'
        saved_model_name = os.path.join(model_dir, saved_model_name)
        
        mcp = tf.keras.callbacks.ModelCheckpoint(saved_model_name,
                                             monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
        
        base_model_dir = f'CNN_point_{target}_0/CNN_model/model_cnn_point'
        if target == 'NON':
            base_model_dir = f'CNN_point_{target}_1/CNN_model/model_cnn_point'
        base = base_model(base_model_dir)
        model = Dropout_model(base, dropout_rate=dropout_rate)
        
        if exist_model is not None:
            model.load_weights(exist_model)

    his = model.fit(train_dataset, epochs=epochs, verbose=2,
                    validation_data=val_dataset, callbacks=[mcp])
    plot_his(his, base_dir, model_name=model_type)
    prediction_bnn(test_data, test_labels, base_dir,
                   model, saved_model_name, model_name=model_type, n_runs=n_runs)
