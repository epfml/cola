r"""
Pickling LIBSVM dataset to a directory such that:

outdir/
    - samples/
        - K/
            - X/        
                - 0     # (n_samples_at_0, n_features)
                - 1
                - ...
            - y/        # (n_samples_at_0, )
                - 0
                - 1
                - ...
            - indices   # (n_samples, ) which is sequential order for all ranks.
    - features/
        - K/
            - X/        
                - 0     # (n_samples, n_features_at_0)
                - 1
                - ...
            - y/        # (n_samples, )
                - 0
                - 1
                - ...
            - indices   # (n_features, ) which is sequential order for all ranks.
"""
import argparse
import os
import numpy as np
import joblib
from sklearn.datasets import load_svmlight_file

parser = argparse.ArgumentParser(description="""""")
parser.add_argument('--input_file', type=str, help='input libsvm')
parser.add_argument('--K', type=int, help='Number of splits')
parser.add_argument('--outdir', type=str, help="""save to outdir/split_by/K/0 outdir/split_by/K/1.""")
args = parser.parse_args()

K = args.K

X, y = load_svmlight_file(args.input_file)

print("Data is loaded: {}".format(X.shape))

n_samples, n_features = X.shape

""" split by samples """
print("-------------------------------------------------------------")
output_folder = os.path.join(args.outdir, 'samples', str(K))
os.makedirs(os.path.join(output_folder, 'X'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'y'), exist_ok=True)
print("The output folder is : {}".format(output_folder))

indices = np.arange(n_samples)
np.random.seed(seed=42)
np.random.shuffle(indices)
block_size = int(np.ceil(n_samples / K))

beg = 0
for k in range(K):
    print("Dumping for rank {}".format(k))
    file_x = os.path.join(output_folder, 'X', str(k))
    file_y = os.path.join(output_folder, 'y', str(k))
    joblib.dump(X[beg:beg+block_size, :], file_x)
    joblib.dump(y[beg:beg+block_size], file_y)
    beg += block_size

print("Dumping indices")
joblib.dump(indices, os.path.join(output_folder, 'indices'))

""" split by features """
print("-------------------------------------------------------------")
output_folder = os.path.join(args.outdir, 'features', str(K))
os.makedirs(os.path.join(output_folder, 'X'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'y'), exist_ok=True)
print("The output folder is : {}".format(output_folder))

indices = np.arange(n_features)
np.random.seed(42)
np.random.shuffle(indices)
block_size = int(np.ceil(n_features / K))

beg = 0
for k in range(K):
    print("Dumping for rank {}".format(k))
    file_x = os.path.join(output_folder, 'X', str(k))
    file_y = os.path.join(output_folder, 'y', str(k))
    joblib.dump(X[:, beg:beg+block_size], file_x)
    joblib.dump(y, file_y)
    beg += block_size

print("Dumping indices")
joblib.dump(indices, os.path.join(output_folder, 'indices'))


print("Finished.")
print("=" * 80)
