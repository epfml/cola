# CoLa
>A **Co**mmunication Efficient Framework For Decentralized **L**inear Le**a**rning.

Similar to [CoCoA](https://arxiv.org/abs/1611.02189), CoLa is communication efficient. Processes communicate with their neighborhood updates only after solving a local subproblem. The local solver can be approximate solvers: for coordinate solvers, we can control the number of coordinates chosen. In this project, we use scikit-learn or Cython to implement the local coordinate solver.

CoLa extends [CoCoA](https://arxiv.org/abs/1611.02189) to the decentralized setting where the communication between processes are limited. This extension is non-trivial because

* dataset cannot be shuffled across nodes;
* there is no central server and not all of the p2p communication are avaiable;

This leads to the heterogeneity in data distribution distribution and biased local updates. CoLa achieves linear rate for strongly convex objective and sublinear rate for general convex objective.

## Installation
Docker
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

## Usage example
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

## Datasets
Download the LIBSVM dataset for linear learning.
```bash
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/url_original.tar.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/url_combined.bz2
```

## Release History

* 0.1.0
    - The first proper release


## License

TBD.

## Reference
If you want to use this code for research, please cite the following paper

* Smith, Virginia, et al. "CoCoA: A general framework for communication-efficient distributed optimization." Journal of Machine Learning Research 18 (2018): 230.
* He, Lie, An Bian, and Martin Jaggi. "COLA: Communication-Efficient Decentralized Linear Learning." arXiv preprint arXiv:1808.04883 (2018).


We thank [PyTorch](https://pytorch.org/) for the distributed communication module and [scikit-learn](http://scikit-learn.org/stable/) for the coordinate solver modules.

* Buitinck, Lars, et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013).
* Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of machine learning research 12.Oct (2011): 2825-2830.
* Paszke, Adam, et al. "Automatic differentiation in pytorch." (2017).


