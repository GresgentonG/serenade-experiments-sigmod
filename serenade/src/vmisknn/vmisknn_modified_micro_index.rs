use std::collections::BinaryHeap;
use std::error::Error;

use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::HashMap;
use hashbrown::HashSet;
use itertools::Itertools;

use crate::vmisknn::similarity_hashed::SimilarityComputationHash;
use crate::vmisknn::SessionScore;

pub struct VMISSkNNModifiedMicroIndex {
    sessions_for_item: HashMap<u64, Vec<u32>>,
    historical_sessions_max_time_stamp: Vec<u32>,
}

impl VMISSkNNModifiedMicroIndex {
    pub fn new(path_train: &str, n_most_recent_sessions: usize) -> Self {
        //println!("Reading inputs for tree index from {}...", path_train);
        let data_train = read_from_file(path_train);
        let (
            historical_sessions_train,
            _historical_sessions_id_train,
            historical_sessions_max_time_stamp,
        ) = data_train.unwrap();

        //println!("Creating index...");
        let historical_sessions = prepare_hashmap(
            &historical_sessions_train,
            &historical_sessions_max_time_stamp,
            n_most_recent_sessions,
        );

        VMISSkNNModifiedMicroIndex {
            sessions_for_item: historical_sessions,
            historical_sessions_max_time_stamp,
        }
    }
}

pub fn read_from_file(
    path: &str,
) -> Result<(Vec<Vec<u64>>, Vec<Vec<usize>>, Vec<u32>), Box<dyn Error>> {
    // Creates a new csv `Reader` from a file
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(path)?;

    // Vector initialization
    let mut session_id: Vec<usize> = Vec::new();
    let mut item_id: Vec<usize> = Vec::new();
    let mut time: Vec<usize> = Vec::new();

    reader.deserialize().for_each(|result| {
        if result.is_ok() {
            let raw: (usize, usize, f64) = result.unwrap();
            let (a_session_id, a_item_id, a_time): (usize, usize, usize) =
                (raw.0, raw.1, raw.2.round() as usize);

            session_id.push(a_session_id);
            item_id.push(a_item_id);
            time.push(a_time);
        } else {
            eprintln!("Unable to parse input!");
        }
    });

    // Sort by session id - the data is unsorted
    let mut session_id_indices: Vec<usize> = (0..session_id.len()).into_iter().collect();
    session_id_indices.sort_by_key(|&i| &session_id[i]);
    let session_id_sorted: Vec<usize> = session_id_indices.iter().map(|&i| session_id[i]).collect();
    let item_id_sorted: Vec<usize> = session_id_indices.iter().map(|&i| item_id[i]).collect();
    let time_sorted: Vec<usize> = session_id_indices.iter().map(|&i| time[i]).collect();

    // Get unique session ids
    session_id.sort_unstable();
    session_id.dedup();

    // Create historical sessions array (deduplicated), historical sessions id array and array with max timestamps.
    //let mut i: usize = 0;
    let mut historical_sessions: Vec<Vec<u64>> = Vec::with_capacity(session_id.len());
    let mut historical_sessions_id: Vec<Vec<usize>> = Vec::with_capacity(session_id.len());
    let mut historical_sessions_max_time_stamp: Vec<u32> = Vec::with_capacity(session_id.len());
    let mut history_session: Vec<u64> = Vec::with_capacity(1000);
    let mut history_session_id: Vec<usize> = Vec::with_capacity(1000);
    let mut max_time_stamp: usize = time_sorted[0];
    // Push initial session and item id
    history_session.push(item_id_sorted[0] as u64);
    history_session_id.push(item_id_sorted[0]);
    // Loop over length of data
    for i in 1..session_id_sorted.len() {
        if session_id_sorted[i] == session_id_sorted[i - 1] {
            if !history_session.contains(&(item_id_sorted[i] as u64)) {
                history_session.push(item_id_sorted[i] as u64);
                history_session_id.push(session_id_sorted[i]);
                if time_sorted[i] > max_time_stamp {
                    max_time_stamp = time_sorted[i];
                }
            }
        } else {
            let mut history_session_sorted = history_session.clone();
            history_session_sorted.sort_unstable();
            historical_sessions.push(history_session_sorted);
            historical_sessions_id.push(history_session_id.clone());
            historical_sessions_max_time_stamp.push(max_time_stamp as u32);
            history_session.clear();
            history_session_id.clear();
            history_session.push(item_id_sorted[i] as u64);
            history_session_id.push(session_id_sorted[i]);
            max_time_stamp = time_sorted[i];
        }
    }

    Ok((
        historical_sessions,
        historical_sessions_id,
        historical_sessions_max_time_stamp,
    ))
}

// Custom binary search because this is stable unlike the rust default (i.e. this always returns right-most index in case of duplicate entries instead of a random match)
fn binary_search_right(array: &[u64], key: u64) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if unsafe { array.get_unchecked(mid) } > &key {
            top = mid;
        } else {
            bottom = mid + 1;
        }
    }

    if top > 0 {
        if array[top - 1] == key {
            Ok(top - 1)
        } else {
            Err(top - 1)
        }
    } else {
        Err(top)
    }
}

fn binary_search_left(array: &[u64], key: u64) -> Result<usize, usize> {
    let mut top: usize = array.len();
    let mut mid: usize;
    let mut bottom: usize = 0;

    if top == 0 {
        return Err(0);
    }

    while bottom < top {
        mid = bottom + (top - bottom) / 2;
        if unsafe { array.get_unchecked(mid) } < &key {
            bottom = mid + 1;
        } else {
            top = mid;
        }
    }

    if top < array.len() {
        if array[top] == key {
            Ok(top)
        } else {
            Err(top)
        }
    } else {
        Err(top)
    }
}

pub(crate) fn prepare_hashmap(
    historical_sessions: &[Vec<u64>],
    timestamps: &[u32],
    n_most_recent_sessions: usize,
) -> HashMap<u64, Vec<u32>> {
    // Initialize arrays
    let historical_sessions_length: usize = historical_sessions.iter().map(|x| x.len()).sum();
    let mut historical_sessions_values = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_session_indices = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_indices = Vec::with_capacity(historical_sessions_length);
    let mut historical_sessions_timestamps = Vec::with_capacity(historical_sessions_length);
    let mut iterable = 0_usize;

    // Create (i) vector of historical sessions, (ii) vector of historical session indices, (iii) vector of session indices
    for (session_id, session) in historical_sessions.iter().enumerate() {
        for (item_id, _) in session.iter().enumerate() {
            historical_sessions_values.push(historical_sessions[session_id][item_id]);
            historical_sessions_indices.push(iterable);
            historical_sessions_session_indices.push(session_id);
            historical_sessions_timestamps.push(timestamps[session_id]);
            iterable += 1;
        }
    }

    // Sort historical session values and session indices array
    historical_sessions_indices.sort_by_key(|&i| historical_sessions_values[i]);
    let historical_sessions_values_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_values[i] as u64)
        .collect();
    let historical_sessions_session_indices_sorted: Vec<u32> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_session_indices[i] as u32)
        .collect();
    let historical_sessions_timestamps_sorted: Vec<u64> = historical_sessions_indices
        .iter()
        .map(|&i| historical_sessions_timestamps[i] as u64)
        .collect();

    // Get unique item_ids and create hashmap
    let mut unique_items = historical_sessions_values_sorted.clone();
    unique_items.dedup();
    let mut historical_sessions_hashed = HashMap::with_capacity(unique_items.len());

    // Loop over unique items to remove all sessions per item older than n_most_recent_sessions and fill hashmap with n_most_recent_similar_sessions per item.
    for current_item in unique_items.iter() {
        let left_index =
            binary_search_left(&historical_sessions_values_sorted, *current_item).unwrap();
        let right_index =
            binary_search_right(&historical_sessions_values_sorted, *current_item).unwrap();
        let current_item_timestamps: Vec<u64> =
            historical_sessions_timestamps_sorted[left_index..right_index + 1].to_vec();
        let current_item_similar_sessions_ids: Vec<u32> =
            historical_sessions_session_indices_sorted[left_index..right_index + 1].to_vec();
        // Sort session ids by reverse timestamp and truncate to n_most_recent_sessions
        let mut timestamp_indices: Vec<usize> = (0..current_item_timestamps.len()).collect();
        timestamp_indices.sort_by_key(|&i| current_item_timestamps[i]);
        let mut current_item_similar_sessions_id_sorted: Vec<u32> = timestamp_indices
            .iter()
            .map(|&i| current_item_similar_sessions_ids[i] as u32)
            .collect();
        current_item_similar_sessions_id_sorted.reverse();
        current_item_similar_sessions_id_sorted.truncate(n_most_recent_sessions);
        // Store (item, similar_sessions) in hashmap
        historical_sessions_hashed.insert(*current_item, current_item_similar_sessions_id_sorted);
    }

    // Return hashmap
    historical_sessions_hashed
}

impl SimilarityComputationHash for VMISSkNNModifiedMicroIndex {
    fn items_for_session(&self, _session: &u32) -> &HashSet<u64, DefaultHashBuilder> {
        unimplemented!()
    }

    fn idf(&self, _item_id: &u64) -> f64 {
        unimplemented!()
    }

    fn find_neighbors(
        &self,
        evolving_session: &[u64],
        k: usize,
        _m: usize,
    ) -> BinaryHeap<SessionScore> {
        // set of elements a historical session has in common with
        // evolving session. Use an index that already contains a mapping
        // from evolving session item to the historic session that contains it.
        let mut similar_session_count: HashMap<u32, HashSet<usize>> = HashMap::new();
        for (pos, item_id) in evolving_session.iter().enumerate() {
            if let Some(similar_sessions) = self.sessions_for_item.get(item_id) {
                for session_id in similar_sessions {
                    if let Some(items) = similar_session_count.get_mut(session_id) {
                        items.insert(pos);
                    } else {
                        similar_session_count.insert(*session_id, HashSet::new());
                    }
                }
            }
        }

        // find top k session by number of items in common with evolving session
        let top_k_sessions = similar_session_count
            .iter()
            .into_iter()
            .sorted_by_key(|(session_id, count)| {
                (
                    count.len(),
                    self.historical_sessions_max_time_stamp[**session_id as usize],
                )
            })
            .rev()
            .take(k);

        // compute score of sessions based on decaying importance evolving
        // session items in order of recency
        let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);
        for (session_id, common_item_set) in top_k_sessions {
            let session_score = common_item_set
                .iter()
                .map(|pos| (pos + 1) as f64 / evolving_session.len() as f64)
                .sum();
            closest_neighbors.push(SessionScore {
                id: *session_id,
                score: session_score,
            })
        }

        closest_neighbors
    }
}
