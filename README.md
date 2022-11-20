# Session-Based Recommendation experiments SIGMOD'22
<img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="250" align="right">

The original repository for the Serenade Recommender System can be found here [Serenade: Low-Latency Session-Based Recommendations](https://github.com/bolcom/serenade)

The original experiments repository can be found at [Serenad experiments](https://github.com/bolcom/serenade-experiments-sigmod)

## KDDB experiments and changes

```
├── SparkCFR  -- spark implementation for collaborative filtering base line
├── pom.xml    -- for maven
└── serenade
    └── bin
    |   ├── evaluator.rs  -- evaluate an indexing method with a training and test dataset and compute it's accuracy metrics
    |   └── paper_micro_benchmark_runtimes.rs  -- benchmark latency performance for indexes
    └── vmisknn  -- various indexing algorithms
        ├── vmisknn_index.rs  -- vmis_knn index implementation for performance benchmarks
        ├── offline_index.rs  -- vmis_knn index implementation for accuracy evaluation
        ├── vmisknn_modified_index.rs  -- our modified vmis_knn index implementation for accuracy evaluation
        └── vmisknn_modified_micro_index.rs  -- our modified vmis_knn index implementation for performance benchmarks
```

Instructions for running the evaluator and benchmarks are in `serenade/README.md`. Specific file paths for datasets can be found in code files.
