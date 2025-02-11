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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str)

args = parser.parse_args()

target = args.target

grz_dir = f'grz_{target}'
w1w2_dir = f'w1w2_{target}'

os.makedirs(grz_dir, exist_ok=True)
os.makedirs(w1w2_dir, exist_ok=True)

def extract_image(src_ra, src_dec, size, pixel, base_dir, band='g'):
    
    '''
    size: 
    in arcsec, size of the cutout, not radius
    '''
    try:
        filename = os.path.join(base_dir, 'legacysurvey-' + base_dir.split('/')[-1] + f'-image-{band}.fits.fz')
        file = fits.open(filename)
    except Exception as e:
        return None
        
    try:
        header = file[1].header
        sky = file[1].data
    except:
        return None
    
    wcs = WCS(header)
    
    #print(wcs)
    target = SkyCoord(ra=src_ra * u.deg, dec=src_dec * u.deg)
    
    cutout = Cutout2D(sky, target, size * u.arcsec, wcs=wcs)
    
    img = Image.fromarray(cutout.data)
    img = img.resize((pixel, pixel), resample=Image.Resampling.LANCZOS)
    
    img = np.array(img).astype(np.float32)
    
    if band == 'W1':
        img = img * 10**(-2.699/2.5)
    elif band == 'W2':
        img = img * 10**(-3.339/2.5)
    else:
        img = img
    
    return img

def grz_img(ra, dec, z, size, pixel, base_dir, num):
    grz = np.zeros((pixel, pixel, 3))
    

    bands = ['g', 'r', 'z']
    err = False
    for i, bd in enumerate(bands):
        img = extract_image(ra, dec, size, pixel, base_dir, bd)
        if img is None:
            err = True
            break
        grz[:, :, i] = img
        
    filename = grz_dir + f'/grz_ra_{np.around(ra, 6)}_dec_{np.around(dec, 6)}_z_{np.around(z, 6)}_id_{num}.npy'
    if err is True:
        filename = grz_dir + f'/grz_ra_{np.around(ra, 6)}_dec_{np.around(dec, 6)}_z_{np.around(z, 6)}_id_{num}_err.npy'
        
    np.save(filename, grz)
    
def w1w2_img(ra, dec, z, size, pixel, base_dir, num):
    w1w2 = np.zeros((pixel, pixel, 2))
    
    bands = ['W1', 'W2']
    err = False
    for i, bd in enumerate(bands):
        img = extract_image(ra, dec, size, pixel, base_dir, bd)
        if img is None:
            err = True
            break
        w1w2[:, :, i] = img
    
    filename = w1w2_dir + f'/w1w2_ra_{np.around(ra, 6)}_dec_{np.around(dec, 6)}_z_{np.around(z, 6)}_id_{num}.npy'
    if err is True:
        filename = w1w2_dir + f'/w1w2_ra_{np.around(ra, 6)}_dec_{np.around(dec, 6)}_z_{np.around(z, 6)}_id_{num}_err.npy'
        
    np.save(filename, w1w2)
    
df = pd.read_csv(f'{target}_img_ds.csv')

# only use limited data
# df = df.loc[:500000]

nums = np.arange(df.shape[0])

# info = np.column_stack((df['target_ra'], df['target_dec'], df['z']))

info = np.column_stack((df['target_ra'], df['target_dec'], df['z']))

prefix = '/mnt/portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9'
dirs = np.array([os.path.join(prefix, dd) for dd in df['dir']])

grz_ls = os.listdir(grz_dir)
indices = np.array([int(re.findall('\d+', name)[-1]) for name in grz_ls])

res_idx = np.setdiff1d(nums, indices)

info_res = info[res_idx]
dirs_res = dirs[res_idx]
nums_res = nums[res_idx]

Parallel(n_jobs=24)(delayed(grz_img)(*catalogue, 10, 64, dd, nn) for catalogue, dd, nn in zip(info_res, dirs_res, nums_res))

Parallel(n_jobs=24)(delayed(w1w2_img)(*catalogue, 10, 32, dd, nn) for catalogue, dd, nn in zip(info, dirs, nums))