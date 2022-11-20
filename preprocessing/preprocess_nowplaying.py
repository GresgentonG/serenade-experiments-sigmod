import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta
import random
import time
# from theano.tensor import inplace

#data config (all methods)
DATA_PATH = '/Users/crescentonc/code/comp5331/datasets_dropbox/nowplaying/raw/'
DATA_PATH_PROCESSED = '/Users/crescentonc/code/comp5331/datasets_preprocessed/'
#DATA_FILE = 'yoochoose-clicks-10M'
DATA_FILE = 'nowplaying'

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#days test default config 
#DAYS_TEST = 90

#slicing default config
NUM_SLICES = 1
DAYS_OFFSET = 0
DAYS_SHIFT = 95
DAYS_TRAIN = 90
DAYS_TEST = 5

#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
    
    
def load_data( file ) : 
    
    #load csv
    data = pd.read_csv( file+'.csv', sep='\t' )
    #user_id,session_id,track_id,time_stamp,artist_id
    
    #data.sort_values( by=['Time'], inplace=True )
    #data['SessionId'] = data.groupby( [data.SessionId] ).grouper.group_info[0]
    
    data.sort_values( by=['SessionId','Time'], inplace=True )
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    


    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data


def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) : 
    # data = data
    #y?
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>1 ].index)]
    
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;
    
    
def slice_data( data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN, days_test=DAYS_TEST ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test ) :
    
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
    
    
    start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    middle =  end - ((end - start) // 10)
    
    #prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection( lower_end ))]
    
    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format( slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat() ) )
    
    #split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index
    
    train = data[np.in1d(data.SessionId, sessions_train)]
    
    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat() ) )
    
    train = train[['SessionId', 'ItemId', 'Time']]
    train_hyperparam = train[:len(train) // 10]
    train.to_csv(output_file + '_train.'+str(slice_id)+'.txt', sep='\t', index=False)
    train_hyperparam.to_csv(output_file + '_train_hyperparam.'+str(slice_id)+'.txt', sep='\t', index=False)

    
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat() ) )
    
    test = test[['SessionId', 'ItemId', 'Time']]
    test_hyperparam = train[:len(test) // 10]
    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)
    test_hyperparam.to_csv(output_file + '_test_hyperparam.'+str(slice_id)+'.txt', sep='\t', index=False)

# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    
#    preprocess_info();
#    preprocess_org();
    preprocess_slices()
