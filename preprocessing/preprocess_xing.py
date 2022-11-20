import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

#data config (all methods)
DATA_PATH = '/Users/crescentonc/code/comp5331/datasets_dropbox/xing/xing2016/'
DATA_PATH_PROCESSED = '/Users/crescentonc/code/comp5331/datasets_preprocessed/'
DATA_FILE = 'xing'
SESSION_LENGTH = 30 * 60 #30 minutes

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#min date config
MIN_DATE = '2014-04-01'

#days test default config 
DAYS_TEST = 1

#slicing default config
NUM_SLICES = 1
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 9
DAYS_TEST = 1


#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )

    
def load_data( file ) : 
    
    #load csv
    data = pd.read_csv( file+'.csv', sep='\t', header=0, usecols=[0,1,2,3], dtype={0:np.int64, 1:np.int64, 2:np.int32, 3:np.int64})
    #specify header names
    del data['interaction_type']
    data.columns = ['UserId', 'ItemId', 'Time']
#     data['Time'] = (data.Time / 1000).astype( int )
    
    #sessionize
    data.sort_values(by=['UserId', 'Time'], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data['Time'].values)
    # check which of them are bigger then session_th
    split_session = tdiff > SESSION_LENGTH
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data['UserId'].values[1:] != data['UserId'].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['SessionId'] = session_ids
    
    data.sort_values( ['SessionId','Time'], ascending=True, inplace=True )
    
    
    print(data)
    
    #output
    
    print( data.Time.min() )
    print( data.Time.max() )
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    # del data['TimeTmp']
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )

    print("----------------")
#     print(data)
#     quit()
    return data[len(data) // 10:]


def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) : 
    
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
    
    
def slice_data( data, output_file, num_slices, days_offset, days_shift, days_train, days_test ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test ) :
    
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
      
    
    start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    middle =  end - ((end - start) // 8)
    # end =  middle + timedelta( days_test )
    
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
    # reorder the columns and delete the  to match the input requirement for the serenade server
    del train['UserId']
    train = train[['SessionId', 'ItemId', 'Time']]
    train_hyperparam = train[:len(train) // 50]
    train.to_csv(output_file + '_train.'+str(slice_id)+'.txt', sep='\t', index=False)
    train_hyperparam.to_csv(output_file + '_train_hyperparam.'+str(slice_id)+'.txt', sep='\t', index=False)
    
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat() ) )
    del test['UserId']
    test = test[['SessionId', 'ItemId', 'Time']]
    test_hyperparam = test[:len(test) // 50]
    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)
    test_hyperparam.to_csv(output_file + '_test_hyperparam.'+str(slice_id)+'.txt', sep='\t', index=False)


# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    # data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    # data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    # print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
    #       format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
    preprocess_slices();
