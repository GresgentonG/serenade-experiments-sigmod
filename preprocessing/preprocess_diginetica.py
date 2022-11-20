import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# data config (all methods)
DATA_PATH = '/Users/crescentonc/code/comp5331/datasets_dropbox/diginetica/raw/'
DATA_PATH_PROCESSED = '/Users/crescentonc/code/comp5331/datasets_preprocessed/'
# DATA_FILE = 'yoochoose-clicks-10M'
# DATA_FILE = 'train-clicks'
# MAP_FILE = 'train-queries'
# MAP_FILE2 = 'train-item-views'
DATA_FILE = 'digi'

# COLS=[0,1,2]
COLS = [0, 2, 3, 4]


# filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# min date config
MIN_DATE = '2016-02-01'

# days test default config
DAYS_TEST = 7

# slicing default config
NUM_SLICES = 5
DAYS_OFFSET = 45
DAYS_SHIFT = 18
DAYS_TRAIN = 25
DAYS_TEST = 7

# retraining default config
DAYS_RETRAIN = 1



# preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                            min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                            min_date=MIN_DATE, days_test=DAYS_TEST):
    data = load_data(path + file)
    # data = filter_min_date(data, min_date)
    data = filter_data(data, min_item_support, min_session_length)
    slice_data( data, path_proc+file, 1, 0, 0, 0, 0 )


def load_data(file):
    data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, dtype={0: np.int32, 1: np.int64, 2: np.int64, 3: str})
    # specify header names
    # data.columns = ['SessionId', 'Time', 'ItemId','Date']
    data.columns = ['SessionId', 'ItemId', 'Time', 'Date']
    data = data[['SessionId', 'Time', 'ItemId', 'Date']]
    print(data)
    data['Time'] = data.Time.fillna(0).astype(np.int64)
    # convert time string to timestamp and remove the original column
    # start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
    data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
    data['Time'] = (data['Time'] / 1000)
    data['Time'] = data['Time'] + data['Datestamp']
    data['TimeO'] = data.Time.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data = data.groupby('SessionId').apply(lambda x: x.sort_values('Time'))     # data = data.sort_values(['SessionId'],['Time'])
    data.index = data.index.get_level_values(1)
    data = data[['SessionId', 'ItemId', 'Time']]
    return data


def filter_data(data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH):
    # filter session length
    session_lengths = data.groupby('SessionId').size()
    session_lengths = session_lengths[ session_lengths >= min_session_length ]
    data = data[np.in1d(data.SessionId, session_lengths.index)]

    # filter item support
    data['ItemSupport'] = data.groupby('ItemId')['ItemId'].transform('count')
    data = data[data.ItemSupport >= min_item_support]

    # filter session length again, after filtering items
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    
    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set default \n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))
    del data['ItemSupport']
    return data;

def slice_data(data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN, days_test=DAYS_TEST ):
    for slice_id in range(0, num_slices):
        split_data_slice(data, output_file, slice_id, days_offset + (slice_id * days_shift), days_train, days_test)


def split_data_slice(data, output_file, slice_id, days_offset, days_train, days_test):
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(),
                 data_end.isoformat()))

    start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    middle =  end - ((end - start) // 14)

    # prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection(lower_end))]

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format(slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(),
                 start.date().isoformat(), middle.date().isoformat(), end.date().isoformat()))

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index

    train = data[np.in1d(data.SessionId, sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(),
                 middle.date().isoformat()))
    train_hyperparam = train[:train.size // 10]
    train.to_csv(output_file + '_train.' + str(slice_id) + '.txt', sep='\t', index=False)
    train_hyperparam.to_csv(output_file + '_train_hyperparam.' + str(slice_id) + '.txt', sep='\t', index=False)

    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(),
                 end.date().isoformat()))
    test_hyperparam = test[:test.size // 10]
    test.to_csv(output_file + '_test.' + str(slice_id) + '.txt', sep='\t', index=False)
    test_hyperparam.to_csv(output_file + '_test_hyperparam.' + str(slice_id) + '.txt', sep='\t', index=False)


# -------------------------------------
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    #preprocess_info()
    preprocess_org_min_date(min_date=MIN_DATE, days_test=DAYS_TEST)
    #preprocess_days_test(days_test=DAYS_TEST)
    #preprocess_slices()
