# CoLa
A Communication Efficient Decentralized Linear Learning.

### Docker
Build the docker image and set up the environment.
```bash
# Build docker images
docker build -t cola .

# Create a container with dataset directory mounted.
path_to_datasets = ''
path_to_cache = ''
docker run --rm -it \
    --mount type=bind,source=path_to_datasets,target=/datasets\
    --mount type=bind,source=path_to_cache,target=/cache\
    cola:latest bash
```

### Dataset
Download the dataset a local directory which will be mounted.
```bash
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/url_original.tar.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/url_combined.bz2
```

### Launch
Execute MPI jobs in docker.
```bash
# Joblib can save the time spend on 
export JOBLIB_CACHE_DIR='/cache'
export OUTPUT_DIR='./'
URL_DATASET_PATH='/datasets/url_combined.bz2'
EPSILON_DATASET_PATH='/datasets/epsilon_normalized.bz2'

world_size=20
mpirun -n $world_size python scripts/run_cola.py \
    --split_by 'features' \
    --max_global_steps 10 \
    --graph_topology 'complete' \
    --exit_time 1000.0 \
    --theta 1e-7 \
    --l1_ratio 1 \
    --lambda_ 1e-4 \
    --local_iters 10.0 \
    --output_dir ${OUTPUT_DIR} \
    --dataset_size 'all' \
    --ckpt_freq 2 \
    --dataset_path ${EPSILON_DATASET_PATH} \
    --dataset epsilon \
    --solvername ElasticNet \
    --algoritmname cola
```

# Reference
If you want to use this code for research, please cite the following paper

* Smith, Virginia, et al. "CoCoA: A general framework for communication-efficient distributed optimization." Journal of Machine Learning Research 18 (2018): 230.
* He, Lie, An Bian, and Martin Jaggi. "COLA: Communication-Efficient Decentralized Linear Learning." arXiv preprint arXiv:1808.04883 (2018).


We thank [PyTorch](https://pytorch.org/) for the distributed communication module and [scikit-learn](http://scikit-learn.org/stable/) for the coordinate solver modules.

* Buitinck, Lars, et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013).
* Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of machine learning research 12.Oct (2011): 2825-2830.
* Paszke, Adam, et al. "Automatic differentiation in pytorch." (2017).
