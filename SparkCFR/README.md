# SparkCFR
This is an implementation for collaborative filtering in session-based recommendation experiment on Spark.

## Structure
```
├── README.md  -- this file
├── pom.xml    -- for maven
└── src
    └── main
        ├── resources
        └── scala
            ├── CFR.scala  -- scala Object for Spark job
            └── Preprocessing.scala  -- scala Object for data preprocessing

```

## Usage
### Requirements
- maven
- scala 2.12
- jre 1.8
- Spark(Optional)

### Preprocessing
1. modify the path and run the preprocessing [python script](./preprocessing/) modified from [session-rec](https://github.com/rn5l/session-rec/tree/master/preprocessing/session_based)
2. Remove the header line at the head of the data file
3. Set the correct `dst` path in `Preprocessing.scala`
4. Run the Preprocessing by 
```shell
    scala Preprocessing.scala ${full_path} ${test_path}
```
where the `full_path` and `test_path` are the paths to the `events_train_full.9.txt` and `events_test.9.txt`
5. You will get `test$i.txt`, `train$i.txt`, and `testSecondHalf$i.txt` where i is 0-9

### Compilation
No need.

### Experiment
1. Modify the paths in `CFR.scala`
2. Modify the `index` variable from 0 to 9 and run the `CFR.scala`
3. Put all the `prediction*.txt` into one `prediction.txt`
4. Run the evaluator with the `events_train_full.9.txt` and `prediction.txt`

### Example
Run `CFR.scala` directly. It will generate a `prediction9.txt` under `/tmp`.

### Environment
Tested on macOS Catalina 10.15.7