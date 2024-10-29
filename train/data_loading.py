import numpy as np
import tensorflow as tf
import h5py
from itertools import cycle
import os
import re
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d

# choice_grz = np.arange(8).tolist()
# cycle_grz = cycle(choice_grz)

# choice_w = np.arange(8).tolist()
# cycle_w = cycle(choice_w)


def transform_grz(arr, rand):
    # rand = next(cycle_grz)
    if rand == 0:
        transform = np.rot90(arr, k=0, axes=(0, 1))
    elif rand == 1:
        transform = np.rot90(arr, k=1, axes=(0, 1))
    elif rand == 2:
        transform = np.rot90(arr, k=2, axes=(0, 1))
    elif rand == 3:
        transform = np.flip(arr, axis=0)
    elif rand == 4:
        transform = np.flip(arr, axis=1)
    elif rand == 5:
        transform = np.rot90(np.flip(arr, axis=0), axes=(0, 1))
    elif rand == 6:
        transform = np.rot90(np.flip(arr, axis=1), axes=(0, 1))
    elif rand == 7:
        transform = arr + np.random.normal(scale=1e-6, size=arr.shape)
    else:
        transform = arr

    return transform


def transform_w1w2(arr, rand):
    # rand = next(cycle_w)
    if rand == 0:
        transform = np.rot90(arr, k=0, axes=(0, 1))
    elif rand == 1:
        transform = np.rot90(arr, k=1, axes=(0, 1))
    elif rand == 2:
        transform = np.rot90(arr, k=2, axes=(0, 1))
    elif rand == 3:
        transform = np.flip(arr, axis=0)
    elif rand == 4:
        transform = np.flip(arr, axis=1)
    elif rand == 5:
        transform = np.rot90(np.flip(arr, axis=0), axes=(0, 1))
    elif rand == 6:
        transform = np.rot90(np.flip(arr, axis=1), axes=(0, 1))
    elif rand == 7:
        transform = arr + np.random.normal(scale=1e-6, size=arr.shape)
    else:
        transform = arr

    return transform

def augment(arr):
    rot1 = np.rot90(arr, k=0, axes=(1, 2))
    rot2 = np.rot90(arr, k=1, axes=(1, 2))
    rot3 = np.rot90(arr, k=2, axes=(1, 2))
    flip1 = np.flip(arr, axis=1)
    flip2 = np.flip(arr, axis=2)
    fr1 = np.rot90(np.flip(arr, axis=1), axes=(1, 2))
    fr2 = np.rot90(np.flip(arr, axis=2), axes=(1, 2))
    return np.concatenate((arr, rot1, rot2, rot3, flip1, flip2, fr1, fr2), axis=0)


def dataset_loading(base_dir, batch_size, target, with_photometry=False):
    
    file = {'BGS': '../BGS_img_ds.csv',
            'LRG': '../LRG_img_ds.csv',
            'ELG': '../ELG_img_ds.csv',
            'NON': '../NON_img_ds.csv',
            'comb': '../ALL_img_ds.csv',
            'BGS_20': '../BGS_img_ds.csv',
            'ELG_zou': '../ELG_zou_img_ds.csv',
            'LRG_zou_supple': '../LRG_zou_supple.csv'}
    
    df = pd.read_csv(file[target])
    
    #g, r, z, w1, w2 = df['flux_g'], df['flux_r'], df['flux_z'], df['flux_w1'], df['flux_w2']
    datasize = df.shape[0]
    
    # idx = (g - r > -0.2) & (g - r < 2.50) & (r - z > 0.) & (r - z < 2.1)\
    #     & (r - w1 > -1.16) & (r - w1 < 4.0) & (r - w2 > -1.94) & (r - w2 < 3.4) #& (df['z'] < 1.0)
    
    # idx_data = np.where(idx == True)[0]
    redshifts = df['z'].values
    
    low_z = np.percentile(redshifts, 0.3)
    high_z = np.percentile(redshifts, 99.7)
    
    # if target == 'ELG':
    #     low_z = 0.6
    #     high_z = 1.6
    
    idx_data = np.where((redshifts > low_z) & (redshifts < high_z))[0]
    # idx_data = idx_data[0:300000]
    #if target == 'NON':
    #    idx_data = np.where((redshifts > low_z) & (redshifts < high_z) & (df['flux_z'].values < 21.3))[0]
    
    grz_imgs = np.memmap(f'../grz_{target}.dat', dtype='float32',
                         mode='r', shape=(datasize, 64, 64, 3))
    w1w2_imgs = np.memmap(f'../w1w2_{target}.dat', dtype='float32',
                          mode='r', shape=(datasize, 32, 32, 2))
    
    if with_photometry:
        photo = df.loc[:, 'flux_g': 'flux_w2']
        photo = np.array(photo)
        bn = photo.shape[-1]
        colors = []
        for i in range(bn - 1):
            for j in range(i + 1, bn):
                colors.append(photo[:, i] - photo[:, j])
        colors = np.array(colors).T
        photo = colors
    
    
    size = idx_data.shape[0]
    test_size = int(0.2 * size)
    val_size = int(0.1 * size)
    
    seed = 123
    np.random.seed(seed)
    test_idx = np.random.choice(idx_data, test_size, replace=False)
    
    train_idx = np.setdiff1d(idx_data, test_idx)
    
    train_size = train_idx.shape[0]
    
    np.save(f'test_idx_{seed}_{target}.npy', test_idx)
    
    print('reading in data')
    
    test_z = redshifts[test_idx]
    test_grz_imgs = grz_imgs[test_idx]
    test_w1w2_imgs = w1w2_imgs[test_idx]
    if with_photometry:
        test_photo = photo[test_idx]
    
#     scaler_grz = RobustScaler()
#     scaler_w1w2 = RobustScaler()
    
#     test_grz_imgs = scaler_grz.fit_transform(test_grz_imgs.reshape(-1, test_grz_imgs.shape[-1])).reshape(test_grz_imgs.shape)
#     test_w1w2_imgs = scaler_w1w2.fit_transform(test_w1w2_imgs.reshape(-1, test_w1w2_imgs.shape[-1])).reshape(test_w1w2_imgs.shape)
    
#     def trans_grz(imgs):
#         return scaler_grz.transform(imgs.reshape(-1, imgs.shape[-1])).reshape(imgs.shape)
        
#     def trans_w1w2(imgs):
#         return scaler_w1w2.transform(imgs.reshape(-1, imgs.shape[-1])).reshape(imgs.shape)
    
    train_z = redshifts[train_idx]
    
    # hist, bins_edges = np.histogram(train_z, bins=50, density=True)
    # bins = (bins_edges[1:] + bins_edges[:-1])/2.
    # interp = interp1d(bins, hist, fill_value='extrapolate')
    # weights = 1 / interp(train_z)
    
    transform_n = np.tile(np.arange(9), train_z.shape[0])
    
    train_idx = np.repeat(train_idx, 9)
    train_z = np.repeat(train_z, 9)
    
    
    shuffle_idx = np.random.choice(train_idx.shape[0], train_idx.shape[0], replace=False)
    train_idx = train_idx[shuffle_idx]
    train_z = train_z[shuffle_idx]
    transform_n = transform_n[shuffle_idx]
    
    
    def train_generator():
        for iid, rand in zip(train_idx, transform_n):
            yield transform_grz(grz_imgs[iid], rand), transform_w1w2(w1w2_imgs[iid], rand)
            
    def test_generator():
        for iid in test_idx:
            yield grz_imgs[iid], w1w2_imgs[iid]
                
    
#     def train_generator():
#         for iid in train_idx:
#             yield transform_grz(trans_grz(grz_imgs[iid])), transform_w1w2(trans_w1w2(w1w2_imgs[iid]))
            
#     def test_generator():
#         for iid in test_idx:
#             yield trans_grz(grz_imgs[iid]), trans_w1w2(w1w2_imgs[iid])
            
    print('Create dataset')
    
                
    train_ds = tf.data.Dataset.from_generator(train_generator,
                                              output_signature=(
                                                  tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(32, 32, 2), dtype=tf.float32),
                                              ))
    train_z_ds = tf.data.Dataset.from_tensor_slices(train_z)
    train_ds = tf.data.Dataset.zip((train_ds, train_z_ds))
    train_ds = train_ds.batch(batch_size).prefetch(batch_size)
    
    test_ds = tf.data.Dataset.from_generator(test_generator,
                                             output_signature=(
                                                  tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(32, 32, 2), dtype=tf.float32),
                                              ))
    test_z_ds = tf.data.Dataset.from_tensor_slices(test_z)
    test_ds = tf.data.Dataset.zip((test_ds, test_z_ds))
    test_ds = test_ds.batch(batch_size).prefetch(batch_size)
    
    
    test_data = [test_grz_imgs, test_w1w2_imgs]
    
    if with_photometry:
        def train_generator():
            for iid, rand in zip(train_idx, transform_n):
                yield transform_grz(grz_imgs[iid], rand), transform_w1w2(w1w2_imgs[iid], rand), \
                        photo[iid]
                
        def test_generator():
            for iid in test_idx:
                yield grz_imgs[iid], w1w2_imgs[iid], photo[iid]
                
        train_ds = tf.data.Dataset.from_generator(train_generator, 
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(32, 32, 2), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(10,), dtype=tf.float32),
                                                  ))
        train_z_ds = tf.data.Dataset.from_tensor_slices(train_z)
        train_ds = tf.data.Dataset.zip((train_ds, train_z_ds))
        train_ds = train_ds.batch(batch_size).prefetch(batch_size)
        
        test_ds = tf.data.Dataset.from_generator(test_generator,
                                                 output_signature=(
                                                      tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(32, 32, 2), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(10,), dtype=tf.float32),
                                                  ))
        test_z_ds = tf.data.Dataset.from_tensor_slices(test_z)
        test_ds = tf.data.Dataset.zip((test_ds, test_z_ds))
        test_ds = test_ds.batch(batch_size).prefetch(batch_size)
        
        test_data = [test_grz_imgs, test_w1w2_imgs, test_photo]
        

    return train_ds, test_ds, train_size, test_data, test_z#, weights