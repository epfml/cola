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
    --logfile ${OUTPUT_DIR}/cola.csv \
    --weightfile ${OUTPUT_DIR}/weight_feature.npy \
    --dataset_size 'all' \
    --dataset_path ${EPSILON_DATASET_PATH} \
    --dataset epsilon \
    --solvername ElasticNet \
    --algoritmname cola
```

# Reference
If you want to use this code for research, please cite the following paper

    @article{smith2018cocoa,
      title={CoCoA: A general framework for communication-efficient distributed optimization},
      author={Smith, Virginia and Forte, Simone and Ma, Chenxin and Tak{\'a}{\v{c}}, Martin and Jordan, Michael I and Jaggi, Martin},
      journal={Journal of Machine Learning Research},
      volume={18},
      number={230},
      pages={1--49},
      year={2018}
    }
    @article{2018arXiv180804883H,
       author = {{He}, L. and {Bian}, A. and {Jaggi}, M.},
        title = "{COLA: Communication-Efficient Decentralized Linear Learning}",
      journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1808.04883},
     primaryClass = "cs.DC",
     keywords = {Computer Science - Distributed, Parallel, and Cluster Computing, Computer Science - Machine Learning, Statistics - Machine Learning},
         year = 2018,
        month = aug,
       adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180804883H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


We thank [PyTorch](https://pytorch.org/) for the distributed communication module and [scikit-learn](http://scikit-learn.org/stable/) for the coordinate solver modules.

    @inproceedings{sklearn_api,
      author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
                   Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
                   Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
                   and Jaques Grobler and Robert Layton and Jake VanderPlas and
                   Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
      title     = {{API} design for machine learning software: experiences from the scikit-learn
                   project},
      booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
      year      = {2013},
      pages = {108--122},
    }
    @article{scikit-learn,
     title={Scikit-learn: Machine Learning in {P}ython},
     author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
             and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
             and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
             Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
     journal={Journal of Machine Learning Research},
     volume={12},
     pages={2825--2830},
     year={2011}
    }
    @article{paszke2017automatic,
      title={Automatic differentiation in PyTorch},
      author={Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam},
      year={2017}
    }
