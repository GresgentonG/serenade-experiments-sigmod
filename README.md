# Collaborative Filtering
The detailed description for the collaborative filtering code [is in the sub-folder SparkCFR](SparkCFR/README.md)

# VMIS-kNN
The repository contains implementation and experiment setup for the VMIS-kNN methods

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
    │   ├── bin                  -- source file to the executables 
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
    │       ├── offline_index.rs
    │       ├── similarity_hashed.rs
    │       ├── similarity_indexed.rs
    │       ├── tree_index.rs
    │       ├── vmisknn_index.rs
    │       ├── vmisknn_index_noopt.rs
    │       ├── vmisknn_index_smallopt.rs
    │       ├── vmisknn_simplified_index.rs   -- our modified vmis-knn
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