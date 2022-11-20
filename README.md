# Collaborative Filtering
The detailed description for the collaborative filtering code [is in the sub-folder SparkCFR](SparkCFR/README.md)

# VMIS-kNN
The repository contains implementation and experiment setup for the VMIS-kNN methods

The original repository for the Serenade Recommender System can be found here [Serenade: Low-Latency Session-Based Recommendations](https://github.com/bolcom/serenade)

The original experiments repository can be found at [Serenad experiments](https://github.com/bolcom/serenade-experiments-sigmod)

## Structure
```
├── .gitignore
├── LICENSE.md
├── README.md
├── SparkCFR                     -- everything about collaborative filtering
├── preprocessing                -- preprocessing python scripts
├── evaluator                    -- evaluator used by collaborative filtering
├── kdd_experiment 
│   ├── Default.toml             -- config file to run the Serenade server with
│   ├── evaluator                -- softlink to the modified evaluator compiled executable
│   ├── getResult.py             -- to get recommendation result from Serenade server
│   ├── hyperparameter_search    -- softlink to the hyperparameter search executable
│   └── result_3_model
│       ├── run.sh               -- a script to run the experiments
│       ├── 30music_hyper.txt    -- experiment results
│       └── ....
└── serenade
    ├── Cargo.toml               -- rust project file
    ├── ...                      -- some bash scripts and dockerfile from original repo
    ├── src
    |   ├── bin
    |   |   ├── evaluator.rs  -- evaluate an indexing method with a training and test dataset and compute it's accuracy metrics
    |   |   └── paper_micro_benchmark_runtimes.rs  -- benchmark latency performance for indexes
    │   ├── config.rs
    │   ├── config_processors.rs
    │   ├── dataframeutils.rs
    │   ├── endpoints
    │   ├── hyperparameter
    │   ├── io.rs
    │   ├── lib.rs
    │   ├── metrics              -- implementation of different metrics
    │   ├── sessions
    │   ├── stopwatch.rs
    │   └── vmisknn              -- implementation of the vs-knn, vmis-knn and our modified vmis-knn
    │       ├── mod.rs
    │       ├── offline_index.rs                -- vmis_knn index implementation for accuracy evaluation
    │       ├── similarity_hashed.rs
    │       ├── similarity_indexed.rs
    │       ├── vmisknn_index.rs                -- vmis_knn index implementation for performance benchmarks
    │       ├── vmisknn_index_noopt.rs
    │       ├── vmisknn_index_smallopt.rs
    │       ├── vmisknn_modified_index.rs        -- our modified vmis_knn index implementation for accuracy evaluation
    │       ├── vmisknn_modified_micro_index.rs  -- our modified vmis_knn index implementation for performance benchmarks
    │       └── vsknn_index.rs
    └── start_webservice.sh
```

## Usage
### Requirements
- python
    - numpy
    - pandas
- rust tool chain

### Preprocessing
1. modify the path and run the preprocessing [python script](./preprocessing/) modified from [session-rec](https://github.com/rn5l/session-rec/tree/master/preprocessing/session_based)

### Compilation
```shell
cd ./serenade
cargo build --release
```

### Experiment
1. modify the `DATASET_PREFIX` in the [experiment running script](kdd_experiment/result_3_model/run.sh)
2. execute ```./run.sh``` in `kdd_experiment/result_3_model`, and results will be generated in the same directory

### Environment
Tested on macOS Monterey 12.5.1
