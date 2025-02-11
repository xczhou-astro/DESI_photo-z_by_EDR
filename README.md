# DESI_photo-z_by_EDR

Relevant code for "Estimating Photometric Redshifts for Galaxies from the DESI Legacy Imaging Surveys with Bayesian Neural Networks Trained by DESI EDR".

Paper link: https://academic.oup.com/mnras/article/536/3/2260/7919747

## dataset

Dataset includes the scripts to extract galaxy images from DESI DR9 data. 

## Train

Train includes the scripts for training the Bayesian neural networks to estimate photo-z from galaxy images in $g$, $r$, $z$, $W1$ and $W2$ bands. 

Two kinds of Bayesian networks, Monte-Carlo Dropout (MC-Dropout) and Multiplicative Normalizing Flows ([MNF](https://github.com/janosh/tf-mnf)) are implemented.

## catalogue

Catalogue includes the scripts for creating photo-z catalogue for sources in 14,000 $\deg^2$ of DESI Legacy Imaging Surveys. 