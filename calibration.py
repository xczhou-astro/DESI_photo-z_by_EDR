import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import bisect
import os
import argparse
import shutil
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--bin_size', type=float, default=0.05)
parser.add_argument('--result_file', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--quality_threshold', type=float, default=None)

args = parser.parse_args()

bin_size = args.bin_size
result_file = args.result_file
quality_threshold = args.quality_threshold
model_type = args.model_type

result = np.load(result_file)

# dataset = pd.read_csv('../DESI_galaxy_dataset.csv')
# test_idx = np.load('test_idx_123.npy')

# dataset = dataset.loc[test_idx]
# dataset.reset_index(drop=True, inplace=True)

z_true = result['z_true']
z_pred = result['z_pred'][:, 0]
z_errs = result['z_pred'][:, 1]

CI = np.arange(0.0, 1.0 + 0.05, 0.05)

def sigma(z_pred, z_spec):
    del_z = z_pred - z_spec
    sigma_nmad = 1.48 * \
        np.median(np.abs((del_z - np.median(del_z))/(1 + z_spec)))
    return np.around(sigma_nmad, 3)


def eta(z_pred, z_spec):
    delt_z = np.abs(z_pred - z_spec) / (1 + z_spec)
    et = np.sum((delt_z > 0.15)) / np.shape(z_pred)[0] * 100
    # et = (np.shape(np.where(delt_z > 0.15 * (1 + z_spec))[0])[0] / np.shape(z_pred)[0]) * 100
    return np.around(et, 2)

def curve(alpha=1):
    counts = []
    for low, high in zip(CI[0:-1], CI[1:]):
        ll, lh = norm.interval(low, z_pred, z_errs * alpha)
        hl, hh = norm.interval(high, z_pred, z_errs * alpha)
        
        idx_1 = (z_true > lh) & (z_true < hh)
        idx_2 = (z_true > hl) & (z_true < ll)
        
        idx = idx_1 | idx_2
        counts.append(np.sum(idx))
        
    cp = []
    ini = 0
    cp.append(ini)
    for i in range(len(counts)):
        ini = (ini + counts[i])
        cp.append(ini)
        
    cp = [p/z_true.shape[0] for p in cp]
    return cp

def root_func(x):
    return np.sum(curve(x) - np.arange(0.0, 1.0 + bin_size, bin_size))

def calibration():
    ini = root_func(x=1)
    if np.abs(ini) < 1e-3:
        alpha = 1.0
        return alpha
    
    aa = np.arange(0.1, 3.0, 0.2)
    roots = []
    for a in aa:
        roots.append(root_func(a))
    roots = np.array(roots)
    
    index_inverse = np.where(roots > 0)[0][0]
    low = aa[index_inverse - 1]
    high = aa[index_inverse]
    
    print(low, high)
    alpha = bisect(root_func, low, high, xtol=1e-4)
    return alpha

alpha = calibration()
print(alpha)

save_dir = f'CNN_{model_type}/outputs'
os.makedirs(save_dir, exist_ok=True)

alpha_filename = os.path.join(save_dir, f'calibration_params.txt')
with open(alpha_filename, 'w') as f:
    f.write(f's = {alpha:.3f}')
f.close()

shutil.copy(result_file, os.path.join(save_dir, 'result.npz'))

z_errs_cal = z_errs * alpha
idx_quality = z_errs_cal < 0.1

z_true_quality = z_true[idx_quality]
z_pred_quality = z_pred[idx_quality]
z_errs_quality = z_errs_cal[idx_quality]

ds_quality = dataset[idx_quality]
ds_quality.reset_index(drop=True, inplace=True)

df_photoz = pd.DataFrame(np.column_stack((z_true_quality, z_pred_quality, z_errs_quality)),
                         columns=['z_true', 'z_pred', 'z_errs'])

catalogue = pd.concat([ds_quality, df_photoz], axis=1)

cal_result_file = os.path.join(save_dir, 'result_calibrated_quality.csv')

catalogue.to_csv(cal_result_file, index=False)

plt.figure(figsize=(6, 6))
plt.plot(CI, curve(alpha=1), label='Before')
plt.plot(CI, curve(alpha=alpha), label='After')
plt.plot(CI, CI, linestyle='dashed', c='k')
plt.legend()
plt.xlabel('Confidence interval')
plt.ylabel('Coverage probability')
figname = os.path.join(save_dir, f'qq_plot.png')
plt.savefig(figname)

plt.figure(figsize=(8, 8))
plt.errorbar(z_true_quality, z_pred_quality, 
             yerr=z_errs_quality, fmt='.', c='red', markersize=3,
             ecolor='lightblue', elinewidth=0.5)
a = np.arange(7)
plt.plot(a, a, c='k', alpha=0.5)
plt.plot(a, 1.15 * a + 0.15, 'k--', alpha=0.5)
plt.plot(a, 0.85 * a - 0.15, 'k--', alpha=0.5)
sigma_all = sigma(z_pred_quality, z_true_quality)
eta_all = eta(z_pred_quality, z_true_quality)
aver_uncertainty = np.around(np.average(z_errs_quality), 3)
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)
plt.title('$\eta = $' + str(eta_all) +
          '  $\sigma_{NMAD} = $' + str(sigma_all) +
          '  $\overline{E}$ =' + str(aver_uncertainty), fontsize=12)
plt.xlabel('$z_{true}$')
plt.ylabel('$z_{pred}$')
figname = os.path.join(save_dir, f'result_calibrated.png')
plt.savefig(figname)

