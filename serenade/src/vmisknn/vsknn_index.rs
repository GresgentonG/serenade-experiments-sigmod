extern crate hashbrown;

use std::collections::BinaryHeap;
use std::error::Error;

use chrono::NaiveDateTime;
use hashbrown::{HashMap, HashSet};
//use std::time::Instant;
use hashbrown::hash_map::DefaultHashBuilder as RandomState;
use itertools::Itertools;
use tdigest::TDigest;

use crate::dataframeutils::TrainingDataStats;
use crate::io::{ItemId, Time, TrainingSessionId};
use crate::vmisknn::similarity_hashed::idf;
use crate::vmisknn::{SessionScore, SessionTime};

use super::similarity_indexed::SimilarityComputationNew;

pub struct VSkNNIndex {
    session_index: HashMap<TrainingSessionId, Vec<ItemId>>,
    session_max_order: HashMap<TrainingSessionId, Time>,
    item_index: HashMap<ItemId, HashSet<TrainingSessionId>>,
    item_idfs: HashMap<ItemId, f64>,
}

pub fn read_from_file(
    path: &str,
) -> Result<(Vec<(TrainingSessionId, ItemId, Time)>, TrainingDataStats), Box<dyn Error>> {
    // Creates a new csv `Reader` from a file
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(path)?;

    // Vector initialization
    let mut session_id: Vec<usize> = Vec::with_capacity(100_000_000);
    let mut item_id: Vec<usize> = Vec::with_capacity(100_000_000);
    let mut time: Vec<usize> = Vec::with_capacity(100_000_000);

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

    let res1 = (session_id_sorted
        .clone()
        .into_iter()
        .zip(item_id_sorted.clone().into_iter())
        .zip(time_sorted.clone().into_iter()))
    .map(|entry| (entry.0 .0 as u32, entry.0 .1 as u64, entry.1 as usize))
    .collect_vec();

    let stats = {
        // Get unique session ids
        session_id.sort_unstable();
        session_id.dedup();

        let qty_records = time_sorted.len();
        let qty_unique_session_ids = session_id.len();

        // Get unique item ids
        // let mut unique_item_ids = item_id.clone();
        item_id.sort_unstable();
        item_id.dedup();
        let qty_unique_item_ids = item_id.len();

        let min_time = time.iter().min().unwrap();
        let min_time_date_time = NaiveDateTime::from_timestamp(*min_time as i64, 0);
        let max_time = time.iter().max().unwrap();
        let max_time_date_time = NaiveDateTime::from_timestamp(*max_time as i64, 0);

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

        let qty_events = historical_sessions
            .iter()
            .map(|items| items.len() as f64)
            .collect_vec();
        let qty_events_digest = TDigest::new_with_size(100);
        let qty_events_digest = qty_events_digest.merge_unsorted(qty_events);

        println!("Using hardcoded session duration percentiles.");
        let session_duration_p05 = 14_u64;
        let session_duration_p25 = 77_u64;
        let session_duration_p50 = 248_u64;
        let session_duration_p75 = 681_u64;
        let session_duration_p90 = 1316_u64;
        let session_duration_p95 = 1862_u64;
        let session_duration_p99 = 3359_u64;
        let session_duration_p99_5 = 4087_u64;
        let session_duration_p100 = 539931_u64;

        // Session qty event percentiles:  p5=2 p25=2 p50=3 p75=6 p90=10 p95=14 p99=27 p99.5=34 p100=9408
        let qty_events_p05 = qty_events_digest.estimate_quantile(0.05).round() as u64;
        let qty_events_p25 = qty_events_digest.estimate_quantile(0.25).round() as u64;
        let qty_events_p50 = qty_events_digest.estimate_quantile(0.50).round() as u64;
        let qty_events_p75 = qty_events_digest.estimate_quantile(0.75).round() as u64;
        let qty_events_p90 = qty_events_digest.estimate_quantile(0.90).round() as u64;
        let qty_events_p95 = qty_events_digest.estimate_quantile(0.95).round() as u64;
        let qty_events_p99 = qty_events_digest.estimate_quantile(0.99).round() as u64;
        let qty_events_p99_5 = qty_events_digest.estimate_quantile(0.995).round() as u64;
        let qty_events_p100 = qty_events_digest.estimate_quantile(1.0).round() as u64;

        TrainingDataStats {
            descriptive_name: path.to_string(),
            qty_records,
            qty_unique_session_ids,
            qty_unique_item_ids,
            min_time_date_time,
            max_time_date_time,
            session_duration_p05,
            session_duration_p25,
            session_duration_p50,
            session_duration_p75,
            session_duration_p90,
            session_duration_p95,
            session_duration_p99,
            session_duration_p99_5,
            session_duration_p100,
            qty_events_p05,
            qty_events_p25,
            qty_events_p50,
            qty_events_p75,
            qty_events_p90,
            qty_events_p95,
            qty_events_p99,
            qty_events_p99_5,
            qty_events_p100,
        }
    };

    Ok((res1, stats))
}

impl SimilarityComputationNew for VSkNNIndex {
    fn items_for_session(&self, session: &TrainingSessionId) -> &[u64] {
        &self.session_index[session]
    }

    fn idf(&self, item: &ItemId) -> f64 {
        self.item_idfs[item]
    }

    fn find_neighbors(
        &self,
        evolving_session: &[ItemId],
        k: usize,
        m: usize,
    ) -> BinaryHeap<SessionScore> {
        let num_items_in_evolving_session = evolving_session.len();

        let mut most_recent_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(m);

        for session_item in evolving_session.iter() {
            if let Some(sessions) = self.sessions_for_item(session_item) {
                for session in sessions {
                    let max_order = self.max_order_for_session(session);

                    let session_with_age = SessionScore::new(*session, max_order as f64);

                    if most_recent_neighbors.len() < m {
                        most_recent_neighbors.push(session_with_age);
                    } else {
                        let mut top = most_recent_neighbors.peek_mut().unwrap();
                        if session_with_age.score > top.score {
                            *top = session_with_age;
                        }
                    }
                }
            }
        }

        let mut closest_neighbors: BinaryHeap<SessionScore> = BinaryHeap::with_capacity(k);

        for neighbor_session in most_recent_neighbors.into_iter() {
            let mut similarity = 0_f64;

            let other_session_items = self.items_for_session(&neighbor_session.id);

            //            let start_time = Instant::now();
            // Decayed dot product
            for (pos, item_id) in evolving_session.iter().enumerate() {
                if other_session_items.contains(item_id) {
                    let decay_factor = (pos + 1) as f64 / num_items_in_evolving_session as f64;
                    similarity += decay_factor;
                }
            }
            // let duration = start_time.elapsed();
            // let duration_as_micros:f64 = duration.as_micros() as f64;
            // if duration_as_micros > 500_f64 {
            //     println!("slow session matching:{} micros, evolving_session.len():{}, other_session_items.len:{}", duration_as_micros, evolving_session.len(), other_session_items.len());
            // }

            if similarity > 0.0 {
                // Update heap holding top-n scored items for this item
                let scored_session = SessionScore::new(neighbor_session.id, similarity);

                if closest_neighbors.len() < k {
                    closest_neighbors.push(scored_session);
                } else {
                    let mut bottom = closest_neighbors.peek_mut().unwrap();
                    if scored_session.score > bottom.score {
                        *bottom = scored_session;
                    }
                }
            }
        }

        closest_neighbors
    }

    fn find_attributes(&self, _item_id: &u64) -> Option<&super::offline_index::ProductAttributes> {
        todo!()
    }
}

impl VSkNNIndex {
    pub fn new(
        interactions: Vec<(TrainingSessionId, ItemId, Time)>,
        sample_size_m: usize,
        max_qty_session_items: usize,
    ) -> Self {
        // start only need to retain sample_size_m sessions per item
        let valid_session_ids: HashSet<u32> = interactions
            .iter()
            .cloned()
            .map(|(session_id, item_id, time)| (item_id, SessionTime::new(session_id, time as u32)))
            .into_group_map()
            .into_iter()
            .flat_map(|(_item_id, mut session_id_with_time)| {
                session_id_with_time.sort();
                session_id_with_time.dedup();
                session_id_with_time.sort_unstable_by(|left, right| {
                    // We keep the sessions with the largest time values
                    left.cmp(right).reverse()
                });
                if session_id_with_time.len() > sample_size_m {
                    // we remove the sessions per item with the lowest time values
                    session_id_with_time.truncate(sample_size_m);
                }
                if session_id_with_time.len() > max_qty_session_items {
                    // this training session has too many items and does not contribute to improving predictions
                    session_id_with_time.clear();
                }

                let session_ids: HashSet<u32> = session_id_with_time
                    .iter()
                    .map(|session_id_time| session_id_time.session_id)
                    .collect();
                session_ids
            })
            .collect();
        // end only need to retain sample_size_m sessions per item

        let mut historical_session_index: HashMap<TrainingSessionId, Vec<ItemId>> = HashMap::new();
        let mut historical_session_max_order: HashMap<TrainingSessionId, Time> = HashMap::new();
        let mut historical_item_index: HashMap<ItemId, HashSet<TrainingSessionId>> = HashMap::new();

        //let mut ignored_training_rows = 0;
        for (session_id, item_id, order) in interactions.into_iter() {
            if !valid_session_ids.contains(&session_id) {
                //ignored_training_rows += 1;
                continue;
            }
            let session_items = historical_session_index
                .entry(session_id)
                .or_insert(Vec::new());
            session_items.push(item_id);

            let current_max_order = historical_session_max_order
                .entry(session_id)
                .or_insert(order);
            if order > *current_max_order {
                *current_max_order = order;
            }

            let item_sessions = historical_item_index
                .entry(item_id)
                .or_insert(HashSet::new());
            item_sessions.insert(session_id);
        }

        //println!("Ignored training events: {} ", ignored_training_rows);
        //println!("Sessions used: {}", valid_session_ids.len());
        //println!("Items used: {}", historical_item_index.len());

        let num_historical_sessions = historical_session_index.len();

        let item_idfs: HashMap<u64, f64> = historical_item_index
            .iter()
            .map(|(item, session_ids)| {
                let item_idf = idf(num_historical_sessions, session_ids.len());

                (*item, item_idf)
            })
            .collect();

        VSkNNIndex {
            session_index: historical_session_index,
            session_max_order: historical_session_max_order,
            item_index: historical_item_index,
            item_idfs,
        }
    }

    pub fn new_from_csv(path_train: &str, n_most_recent_sessions: usize) -> Self {
        //println!("Reading inputs for tree index from {}...", path_train);
        let data_train = read_from_file(path_train).unwrap();
        VSkNNIndex::new(
            data_train.0,
            n_most_recent_sessions,
            data_train.1.qty_events_p99_5 as usize,
        )
    }

    fn sessions_for_item(&self, item: &u64) -> Option<&HashSet<u32, RandomState>> {
        self.item_index.get(item) // move object and ownership to function that call us
    }

    fn max_order_for_session(&self, session: &TrainingSessionId) -> Time {
        self.session_max_order[session]
    }
}
