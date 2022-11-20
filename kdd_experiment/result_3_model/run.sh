set DATASET_PREFIX './comp5331/datasets_preprocessed/'
for dataset in '30music' 'aotm' 'digi' 'nowplaying' 'retailrocket' 'rsc15-clicks' 'xing';
    set train_hyper $DATASET_PREFIX$dataset'_train_hyperparam.0.txt'
    set test_hyper $DATASET_PREFIX$dataset'_test_hyperparam.0.txt'
    set train $DATASET_PREFIX$dataset'_train.0.txt'
    set test $DATASET_PREFIX$dataset'_test.0.txt'
    set hyper_result $dataset'_hyper.txt'
    ../hyperparameter_search $train_hyper $test_hyper > $hyper_result
    set m $(grep -Eo "m_most_recent_sessions: [0-9]+" $hyper_result | grep -Eo "\d+")
    set k $(grep -Eo "neighborhood_size_k: [0-9]+" $hyper_result | grep -Eo "\d+")
    set s $(grep -Eo "max_items_in_session: [0-9]+" $hyper_result | grep -Eo "\d+")
    echo $m
    echo $k
    echo $s
    echo ==========
    ../evaluator $train $test $m $k $s > $dataset'_result.txt'
end


    
    