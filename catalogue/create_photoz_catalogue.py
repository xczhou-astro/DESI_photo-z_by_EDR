import numpy as np
import os
from astropy.io import fits
from PIL import Image
from joblib import Parallel, delayed
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
import pandas as pd
import re
import tensorflow as tf
import gc
import subprocess
from model import *
from scipy.spatial import cKDTree
import time


batch_size = 8192 * 2
# gpu_indices = [0, 1, 2, 4]  # Replace with the actual GPU indices you want to use
gpu_indices = [2, 4] # south
# gpu_indices = [1, 3] # north

# Get the list of available GPUs
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    # Set the desired GPU devices
    visible_devices = [physical_devices[i] for i in gpu_indices]
    tf.config.experimental.set_visible_devices(visible_devices, 'GPU')

    # Allow memory growth for each GPU to avoid allocation errors
    for device in visible_devices:
        tf.config.experimental.set_memory_growth(device, True)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices:{}'.format(strategy.num_replicas_in_sync))
        
def softplus(x):
    return np.log(np.exp(x) + 1)

def find_dir(ra, dec, dir_info, dirs):
    
    ra_min = dir_info[:, 2]
    dec_min = dir_info[:, 3]
    ra_max = dir_info[:, 4]
    dec_max = dir_info[:, 5]
    
    idx = np.where((ra_min < ra) & (ra_max > ra) & (dec_min < dec) & (dec_max > dec))[0]
    
    directory = None
    if len(idx) != 0:
        idx = idx[0]
        directory = dirs[idx]
    return directory

def photoz_for_images(grz, w1w2, target):
    
    model_targets = {'BGS': model_BGS,
                    'LRG': model_LRG,
                    'NON': model_NON}
    
    model = model_targets[target]
    
    # tf.keras.backend.clear_session()
    file = open(f'CNN_mnf/CNN_mnf_{target}/calibration_params.txt')
    string = file.readline()
    alpha = float(string.split('=')[-1])
    file.close()
    
    datasize = grz.shape[0]
    
#     test_data = [grz, w1w2]
    
#     datasize = grz.shape[0]
    
#     n_runs = 200
#     z_pred_n_runs = np.zeros((n_runs, datasize, 2))
#     for i in range(n_runs):
#         z_pred_n_runs[i, ...] = np.reshape(model.predict(test_data, batch_size=32768, verbose=0),
#                                            (datasize, 2))
#         tf.keras.backend.clear_session()
#         gc.collect()
    
    ### modification ###
    
    grz = np.tile(grz, (10, 1, 1, 1))
    w1w2 = np.tile(w1w2, (10, 1, 1, 1))
    
    test_data = [grz, w1w2]
    
    prediction = []
    for i in range(20):
        pred = model.predict(test_data, batch_size=batch_size, verbose=0)
        pred = np.reshape(pred, (10, datasize, 2))
        prediction.append(pred)
        tf.keras.backend.clear_session()
        gc.collect()
        
    z_pred_n_runs = np.concatenate(prediction, axis=0)
    
    del test_data
    gc.collect()
    
    ### modification ###
    
    z_pred_n_runs[:, :, 1] = 1e-4 + softplus(z_pred_n_runs[:, :, 1] * 0.01)

    mubar = np.average(z_pred_n_runs[:, :, 0], axis=0)
    aleatoric = np.average(z_pred_n_runs[:, :, 1] ** 2, axis=0)
    epistemic = np.average(z_pred_n_runs[:, :, 0] ** 2 - mubar ** 2, axis=0)

    stdsquare = aleatoric + epistemic

    std = np.sqrt(stdsquare) * alpha
    aleatoric = np.sqrt(aleatoric) * alpha
    epistemic = np.sqrt(epistemic) * alpha
    
    mubar[mubar < 0] = -10.
    
    return np.column_stack((mubar, std, aleatoric, epistemic))

def extract_fits(directory, band='g'):
    
    filename = os.path.join(directory, 'legacysurvey-' + directory.split('/')[-1] + f'-image-{band}.fits.fz')
    
    def read_file(filename):
        try:
            file = fits.open(filename)
        except Exception as e:
            return None

        try:
            header = file[1].header
            sky = file[1].data
        except:
            return None
        return (header, sky)
    
    info = read_file(filename)
    if info is None:
        
        print(filename.split('/')[-1])
        
        sfilename = os.path.join('raw_images', filename.split('/')[-1])
        
        if os.path.exists(sfilename) != True:
        
            command = 'wget https://' + filename.strip('/mnt/') + ' -O raw_images/' + sfilename.split('/')[-1]
            process = subprocess.Popen(command, shell=True)
            process.wait()
        
        info = read_file(sfilename)
    
    header, sky = info
    
    if band == 'W1':
        sky = sky * 10**(-2.699/2.5)
    elif band == 'W2':
        sky = sky * 10**(-3.339/2.5)
    else:
        sky = sky
    
    return header, sky

def get_cutout(ra, dec, size, pixel, wcs, sky):
    # wcs = WCS(header)
    
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    cutout = Cutout2D(sky, target, size * u.arcsec, wcs=wcs)
    
    img = Image.fromarray(cutout.data)
    img = img.resize((pixel, pixel), resample=Image.Resampling.LANCZOS)
    
    img = np.array(img).astype(np.float32)
    
    return img

def galaxy_imgs(ras, decs, size, pixel, wcs, sky):
    
    imgs = Parallel(n_jobs=-1)(delayed(get_cutout)(ra, dec, size, pixel, wcs, sky)
                                      for ra, dec in zip(ras, decs))
    imgs = np.expand_dims(np.array(imgs), axis=-1)
    
    return imgs

def grz_w1w2_imgs(srcs, directory):
    
    ras, decs = srcs['ra'].values, srcs['dec'].values
    grz_imgs = []
    for band in ['g', 'r', 'z']:
        header, sky = extract_fits(directory, band)
        wcs = WCS(header)
        imgs = galaxy_imgs(ras, decs, size=10., pixel=64, wcs=wcs, sky=sky)
        grz_imgs.append(imgs)
        
    w1w2_imgs = []
    for band in ['W1', 'W2']:
        header, sky = extract_fits(directory, band)
        wcs = WCS(header)
        imgs = galaxy_imgs(ras, decs, size=10., pixel=32, wcs=wcs, sky=sky)
        w1w2_imgs.append(imgs)
        
    grz_imgs = np.concatenate(grz_imgs, axis=-1)
    w1w2_imgs = np.concatenate(w1w2_imgs, axis=-1)
    return grz_imgs, w1w2_imgs

def match_with_DESI_specz(coord_spec, redshift, ds):
    coord_c = SkyCoord(ds['ra'].values * u.deg, ds['dec'].values * u.deg)
    idx, d2d, _ = coord_c.match_to_catalog_sky(coord_spec)
    idx_d2d = d2d < 1.5 * u.arcsec
    idx_ds_matched = idx_d2d
    idx_spec_matched = idx[idx_d2d]
    specz = np.ones((ds.shape[0])) * -10.
    specz[idx_ds_matched] = redshift[idx_spec_matched]
    df_specz = pd.DataFrame(specz.T, columns=['zspec'])
    # print(df_specz)
    ds_all = pd.concat([ds, df_specz], axis=1)
    return ds_all

def main_processing(sources, dir_info, dirs, target):
    # directories = []
    # for ra, dec, in zip(sources['ra'], sources['dec']):
    #     directories.append(find_dir(ra, dec, dir_info, dirs))
    # directories = np.array(directories)
    directories = Parallel(n_jobs=-1, backend='threading')(delayed(find_dir)(ra, dec, dir_info, dirs)
                                                          for ra, dec in zip(sources['ra'], sources['dec']))
    
    directories = np.array(directories)
    
    idx_dirs = directories != None
    sources = sources.loc[idx_dirs]
    sources.reset_index(drop=True, inplace=True)
    
    dirs_for_sources = directories[idx_dirs]
    
    unique_dirs, counts = np.unique(dirs_for_sources, return_counts=True)
    
    print(f'{unique_dirs.shape[0]} unique directories')
    
    cumsum = np.cumsum(counts)
    assert cumsum[-1] == sources.shape[0], 'unmatch'
    threshold = 50000
    nparts = sources.shape[0] // threshold + 1
    
    
    def get_images(uni_dir):
        idx_srcs = np.where(dirs_for_sources == uni_dir)[0]
        srcs = sources.iloc[idx_srcs]
        srcs.reset_index(drop=True, inplace=True)
        grz, w1w2 = grz_w1w2_imgs(srcs, uni_dir)
        return grz, w1w2, srcs
    
    
    df_results = []
    
    for i in range(nparts):
        print(f'Processing {i + 1} part...')
        start = time.time()
        idx_part = (cumsum > threshold * i) & (cumsum <= threshold * (i + 1)) 
        uni_dirs_part = unique_dirs[idx_part]
        
        all_grz = []
        all_w1w2 = []
        dfd = []
#         for uni_dir in uni_dirs_part:
#             # idx_srcs = dirs_for_sources == uni_dir
#             idx_srcs = np.where(dirs_for_sources == uni_dir)[0]
#             srcs = sources.iloc[idx_srcs]
#             srcs.reset_index(drop=True, inplace=True)
#             grz, w1w2 = grz_w1w2_imgs(srcs, uni_dir)
            
#             all_grz.append(grz)
#             all_w1w2.append(w1w2)
#             dfd.append(srcs)
        
        content = Parallel(n_jobs=-1)(delayed(get_images)(uni_dir) for uni_dir in uni_dirs_part)
        for i in range(len(content)):
            all_grz.append(content[i][0])
            all_w1w2.append(content[i][1])
            dfd.append(content[i][2])
        
        df_part = pd.concat(dfd, axis=0)
        df_part.reset_index(drop=True, inplace=True)
        
        all_grz = np.concatenate(all_grz, axis=0)
        all_w1w2 = np.concatenate(all_w1w2, axis=0)
        end = time.time()
        print('Getting image: ', end - start)
        
        print('grz:', all_grz.shape)
        print('w1w2:', all_w1w2.shape)
        
        print('Calculating photoz')
        
        start = time.time()
        results = photoz_for_images(all_grz, all_w1w2, target)
        end = time.time()
        print('Calc photoz:', end - start)
        
        # del all_grz, all_w1w2
        # gc.collect()
        df_result_part = pd.DataFrame(results,
                                      columns=['zphot', 'zerr', 'aleatoric', 'epistemic'])
        df_result_part = pd.concat([df_part, df_result_part], axis=1)
        
        df_results.append(df_result_part)
    
    df_results = pd.concat(df_results, axis=0)
    df_results.reset_index(drop=True, inplace=True)
    return df_results
        
    
        
def get_photoz_brick_target(sources_all, dir_info, dirs):
    df_all = []
    for target in ['BGS', 'LRG', 'NON']:
        sources = sources_all[sources_all[target] == True]
        if target == 'NON':
            idx_thre = sources['flux_z'] < 21.3
            sources = sources[idx_thre]
        
        sources.reset_index(drop=True, inplace=True)
        
        num_targets = sources.shape[0]
        print(f'num of {target} targets: ', num_targets)
        
        if num_targets != 0:
            df_results = main_processing(sources, dir_info, dirs, target)
            df_all.append(df_results)
            
    df_all = pd.concat(df_all, axis=0)
    df_all.reset_index(drop=True, inplace=True)
    return df_all


def initialize_model(target):
    with strategy.scope():
        model = tf.keras.models.load_model(f'Saved_models/{target}_model', compile=False)
    return model

os.makedirs('raw_images', exist_ok=True)

dir_info = np.load('../useful_info/dir_info.npy')
dirs = np.load('../useful_info/dirs_with_grzw1w2.npy', allow_pickle=True)

north_ds_ls = os.listdir('../north_ds')
south_ds_ls = os.listdir('../south_ds')

north_ds_ls.sort()
south_ds_ls.sort()

north_ds_ls = np.array(north_ds_ls)
south_ds_ls = np.array(south_ds_ls)

os.makedirs('photoz_catalogue_north', exist_ok=True)
os.makedirs('photoz_catalogue_south', exist_ok=True)

exist_catalog_north = os.listdir('photoz_catalogue_north')
exist_catalog_north = np.array(exist_catalog_north)
north_ds_ls = np.setdiff1d(north_ds_ls, exist_catalog_north)

exist_catalog_south = os.listdir('photoz_catalogue_south')
exist_catalog_south = np.array(exist_catalog_south)
south_ds_ls = np.setdiff1d(south_ds_ls, exist_catalog_south)

##
south_ds_ls = south_ds_ls[::-1]
##

north_ds = ['../north_ds/' + name for name in north_ds_ls]
south_ds = ['../south_ds/' + name for name in south_ds_ls]

BGS = pd.read_csv('../BGS_specz_ds.csv')
LRG = pd.read_csv('../LRG_specz_ds.csv')
ELG = pd.read_csv('../ELG_specz_ds.csv')

comb = pd.concat([BGS, LRG, ELG], axis=0)
coord_spec = SkyCoord(comb['target_ra'].values * u.deg, comb['target_dec'].values * u.deg)
spec_z = comb['z'].values

model_BGS = initialize_model('BGS')
model_LRG = initialize_model('LRG')
model_NON = initialize_model('NON')

# for filename in north_ds:
#     print(filename)
    
#     dataset = pd.read_csv(filename)
#     dataset.drop(labels=['release', 'brickid', 'brickname', 'objid', 'maskbits',
#                          'sersic', 'fibertotflux_g', 'fibertotflux_r', 'fibertotflux_z'], axis=1, inplace=True)
    
#     if len(dataset) != 0:
        
#         num_sources = dataset.shape[0]
#         print('Total sources: ', num_sources)
        
#         ds = get_photoz_brick_target(dataset, dir_info, dirs)
        
#         ds_matched_specz = match_with_DESI_specz(coord_spec, spec_z, ds)
        
#         saved_filename = 'photoz_catalogue_north_new/' + filename.split('/')[-1]
#         ds_matched_specz.to_csv(saved_filename, index=False)


for filename in south_ds:
    print(filename)
    
    dataset = pd.read_csv(filename)
    dataset.drop(labels=['release', 'brickid', 'brickname', 'objid', 'maskbits',
                         'sersic', 'fibertotflux_g', 'fibertotflux_r', 'fibertotflux_z'], axis=1, inplace=True)
    
    if len(dataset) != 0:
        
        num_sources = dataset.shape[0]
        print('Total sources: ', num_sources)
        
        ds = get_photoz_brick_target(dataset, dir_info, dirs)
        
        ds_matched_specz = match_with_DESI_specz(coord_spec, spec_z, ds)
        
        saved_filename = 'photoz_catalogue_south/' + filename.split('/')[-1]
        ds_matched_specz.to_csv(saved_filename, index=False)