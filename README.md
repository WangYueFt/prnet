# Partial Registration Network

## Prerequisites 
PyTorch=1.0.1: https://pytorch.org  (PyTorch 1.1 has a svd bug, which will crash the training)

scipy>=1.2 

numpy

h5py

tqdm

sklearn

## Conda environment 

conda env create -f environment.yml --name prnet

conda activate prnet

## Training

### exp1 modelnet40

python main.py --exp_name=exp1

### exp2 modelnet40 unseen

python main.py --exp_name=exp2 --unseen=True

## exp3 modelnet40 gaussian noise

python main.py --exp_name=exp3 --gaussian_noise=True

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_NeurIPS,
	  title={PRNet: Self-Supervised Learning for Partial-to-Partial Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {33rd Conference on Neural Information Processing Systems (To appear)},
	  year={2019}
	}

## Code Reference

Code reference: Deep Closest Point

## License
MIT License
