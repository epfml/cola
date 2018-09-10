# CoLa
Communication-Efficient Decentralized Linear Learning

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
    --logfile ${OUTPUT_DIR}/cola.csv \
    --weightfile ${OUTPUT_DIR}/weight_feature.npy \
    --dataset_size 'all' \
    --dataset_path ${EPSILON_DATASET_PATH} \
    --dataset epsilon \
    --solvername ElasticNet \
    --algoritmname cola
```

# Reference
If you use this code, please cite the following paper

    @inproceedings{cola2018nips,
       author = {He, L. and Bian, A. and Jaggi, M.},
        title = "{COLA: Communication-Efficient Decentralized Linear Learning}",
      booktitle = {NIPS 2018 - Advances in Neural Information Processing Systems},
         year = 2018,
    }

We thank [PyTorch](https://pytorch.org/) for the distributed communication module and [scikit-learn](http://scikit-learn.org/stable/) for the coordinate solver modules.
