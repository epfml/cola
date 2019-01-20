# CoLa: Decentralized Linear Learning

Decentralized machine learning is a promising emerging paradigm in view of global challenges of data ownership and privacy. We consider learning of linear classification and regression models, in the setting where the training data is decentralized over many user devices, and the learning algorithm must run on-device, on an arbitrary communication network, without a central coordinator. We propose COLA, a new decentralized training algorithm with strong theoretical guarantees and superior practical performance. Our framework overcomes many limitations of existing methods, and achieves communication efficiency, scalability, elasticity as well as resilience to changes in data and participating devices.

[Paper appearing at NeurIPS 2018](https://arxiv.org/abs/1808.04883)

The CoLa algorithm scheme is communication efficient, as is its centralized counterpart [CoCoA](https://arxiv.org/abs/1611.02189). Processes communicate with their neighborhood nodes only after working on a data-local subproblem for a flexible amount of time, before exchanging a single parameter vector with neighbor nodes. The local solver can be any existing approximate solver: we e.g. provide coordinate solver in our implementation, with flexible number of local coordinate updates, here calling scikit-learn or Cython. Communication between nodes is done via pytorch / MPI.

## Getting Started
Build a docker image or pull an existing one from dockerhub
```bash
# Build docker images
docker build -t cola .
```
Run the docker with
```bash
# Create a container with dataset directory mounted.
docker run --rm -it --mount type=bind,source=path_to_datasets,target=/datasets \
    cola:latest bash
```

### Dataset
Download the dataset. A local directory which will be mounted.
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
If you use this code, please cite the following [paper](https://arxiv.org/abs/1808.04883)

    @inproceedings{cola2018nips,
       author = {He, L. and Bian, A. and Jaggi, M.},
        title = "{COLA: Decentralized Linear Learning}",
      booktitle = {NeurIPS 2018 - Advances in Neural Information Processing Systems},
         year = 2018,
    }

We thank [PyTorch](https://pytorch.org/) for the distributed communication module and [scikit-learn](http://scikit-learn.org/stable/) for the coordinate solver modules.

* Buitinck, Lars, et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013).
* Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of machine learning research 12.Oct (2011): 2825-2830.
* Paszke, Adam, et al. "Automatic differentiation in pytorch." (2017).


