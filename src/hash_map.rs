use std::borrow::Borrow;
use std::cmp::Eq;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::iter::IntoIterator;
use std::marker::PhantomData;
use std::vec::IntoIter;

type Directory<K, V> = Vec<Option<Item<K, V>>>;

pub struct HashMap<K, V> {
    directory: Directory<K, V>,
    // real_size is the number of live key value pairs in the map.
    real_size: usize,
    // effective_size is the number of used slots in the directory
    //  (i.e live + tomb-stoned key value pairs)
    effective_size: usize,
}

struct Item<K, V> {
    key: K,
    value: Option<V>,
    probe_length: usize,
}

impl<K, V> Item<K, V> {
    fn new(key: K, value: V) -> Self {
        Item {
            key,
            value: Some(value),
            probe_length: 0,
        }
    }
}

// Generic Key Value Iterator.
pub struct KeyIter<'a, K, V> {
    inner: KeyValueIter<'a, K, V>,
}

impl<'a, K, V> Iterator for KeyIter<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(key, _)| key)
    }
}
pub struct KeyValueIter<'a, K, V> {
    next_index: usize,
    directory: &'a Directory<K, V>,
}

impl<'a, K, V> Iterator for KeyValueIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_index < self.directory.len() {
            let maybe_item = &self.directory[self.next_index];
            self.next_index = self.next_index + 1;

            if let it @ Some(_) = maybe_item {
                let maybe_kv = it
                    .as_ref()
                    .and_then(|item| item.value.as_ref().map(|value| (&item.key, value)));

                if maybe_kv.is_some() {
                    return maybe_kv;
                }
            }
        }
        None
    }
}

// Entry API
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K, V> {
    key: K,
    value: &'a mut V,
}

pub struct VacantEntry<'a, K, V> {
    key: K,
    value: PhantomData<V>,
    map: &'a mut HashMap<K, V>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {}

impl<'a, K, V> VacantEntry<'a, K, V> {
    pub fn insert(self, default: V) -> &'a mut V
    where
        K: Hash + Eq,
    {
        let i = self.map.insert_item(self.key, default);
        self.map.directory[i]
            .as_mut()
            .unwrap()
            .value
            .as_mut()
            .unwrap()
    }
}

impl<'a, K, V> Entry<'a, K, V> {
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => &entry.key,
            Entry::Vacant(entry) => &entry.key,
        }
    }

    pub fn or_insert(self, default: V) -> &'a mut V
    where
        K: Hash + Eq,
    {
        match self {
            Entry::Occupied(entry) => entry.value,
            Entry::Vacant(entry) => entry.insert(default),
        }
    }
}

// HashMap Impl
impl<'a, K, V> IntoIterator for &'a HashMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = KeyValueIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        KeyValueIter {
            next_index: 0,
            directory: &self.directory,
        }
    }
}

impl<K, V> HashMap<K, V> {
    pub fn with_capacity(initial_capacity: usize) -> Self {
        let initial_capacity = if initial_capacity == 0 {
            1
        } else {
            initial_capacity
        };
        let mut directory: Directory<K, V> = Vec::with_capacity(initial_capacity);
        directory.resize_with(initial_capacity, Option::default);
        HashMap {
            directory,
            real_size: 0,
            effective_size: 0,
        }
    }

    pub fn new() -> Self {
        HashMap::with_capacity(3)
    }

    pub fn iter(&self) -> KeyValueIter<K, V> {
        KeyValueIter {
            next_index: 0,
            directory: &self.directory,
        }
    }

    pub fn keys(&self) -> KeyIter<K, V> {
        KeyIter {
            inner: KeyValueIter {
                next_index: 0,
                directory: &self.directory,
            },
        }
    }

    pub fn insert(&mut self, key: K, value: V)
    where
        K: Hash + Eq,
    {
        self.insert_item(key, value);
    }

    fn insert_item(&mut self, key: K, value: V) -> usize
    where
        K: Hash + Eq,
    {
        // Update the item's value if it already exists.
        if let Some(i) = self.index_of(&key) {
            let item = self.directory[i].as_mut().unwrap();
            item.value.replace(value);
            return i;
        }

        if self.effective_size == self.directory.len() {
            // Resize
            let new_capacity = self.real_size * 2;
            let mut new_directory: Directory<K, V> = Vec::with_capacity(new_capacity);
            new_directory.resize_with(new_capacity, Option::default);

            self.real_size = 0;
            self.effective_size = 0;
            std::mem::replace::<Directory<K, V>>(&mut self.directory, new_directory)
                .into_iter()
                .filter_map(|item| item)
                .filter(|item| item.value.is_some())
                .for_each(|item| self.insert(item.key, item.value.unwrap()));
        }

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        let num_slots = self.directory.len();
        let slot = (hash % num_slots as u64) as usize;

        let mut i = slot;
        let mut curr_probe_length = 0;
        let mut curr_item = Item::new(key, value);
        while let Some(item) = &mut self.directory[i] {
            if curr_probe_length > item.probe_length {
                curr_item.probe_length = curr_probe_length;
                curr_item = self.directory[i].replace(curr_item).unwrap();
                curr_probe_length = curr_item.probe_length;
            }

            i = (i + 1) % num_slots;
            curr_probe_length = curr_probe_length + 1;
        }
        curr_item.probe_length = curr_probe_length;
        self.directory[i] = Some(curr_item);
        self.real_size = self.real_size + 1;
        self.effective_size = self.effective_size + 1;
        return i;
    }

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            hasher.finish()
        };

        let num_slots = self.directory.len();
        let slot = (hash % num_slots as u64) as usize;

        let mut i = slot;
        while let it @ Some(_) = &self.directory[i] {
            if key == it.as_ref().unwrap().key.borrow() {
                return it.as_ref().and_then(|item| item.value.as_ref());
            }

            i = (i + 1) % num_slots;

            if i == slot {
                // We have wrapped around.
                break;
            }
        }

        None
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get_item(key).and_then(|item| item.value.as_mut())
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let removed = self.get_item(key).and_then(|item| item.value.take());
        if removed.is_some() {
            self.real_size = self.real_size - 1;
        }

        removed
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    pub fn entry(&mut self, key: K) -> Entry<K, V>
    where
        K: Hash + Eq,
    {
        match self.index_of(&key) {
            Some(index) => {
                return Entry::Occupied(OccupiedEntry {
                    key,
                    value: (&mut self.directory[index])
                        .as_mut()
                        .unwrap()
                        .value
                        .as_mut()
                        .unwrap(),
                })
            }
            None => Entry::Vacant(VacantEntry {
                key,
                value: PhantomData,
                map: self,
            }),
        }
    }

    pub fn drain(&mut self) -> IntoIter<(K, V)> {
        self.real_size = 0;
        self.effective_size = 0;
        let ve = self
            .directory
            .iter_mut()
            .filter_map(|maybe_item| match maybe_item {
                Some(_) => {
                    let item = maybe_item.take().unwrap();
                    match item.value {
                        Some(_) => Some((item.key, item.value.unwrap())),
                        None => None,
                    }
                }
                None => None,
            })
            .collect::<Vec<(_, _)>>();
        ve.into_iter()
    }

    pub fn len(&self) -> usize {
        self.real_size
    }

    fn get_item<Q>(&mut self, key: &Q) -> Option<&mut Item<K, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.index_of(key)
            .and_then(move |i| self.directory[i].as_mut())
    }

    fn index_of<Q>(&mut self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            hasher.finish()
        };

        let num_slots = self.directory.len();
        let slot = (hash % num_slots as u64) as usize;
        let mut i = slot;
        loop {
            if let it @ Some(..) = &self.directory[i] {
                if key == it.as_ref().unwrap().key.borrow() {
                    break it
                        .as_ref()
                        .and_then(|item| item.value.as_ref().and(Some(i)));
                }

                i = (i + 1) % num_slots;

                if i == slot {
                    // We have wrapped around.
                    break None;
                }
            } else {
                break None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Entry;
    use super::HashMap;

    #[test]
    fn insert_and_get_value() {
        let mut map = HashMap::new();
        assert!(map.get("a").is_none());
        map.insert("a", "cat");
        map.insert("b", "bat");
        assert_eq!(map.get("a").unwrap(), &"cat");
        assert_eq!(map.get("a").unwrap(), &"cat");
        assert_eq!(map.get("b").unwrap(), &"bat");
        assert!(map.get("c").is_none());
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn update_value() {
        let mut map = HashMap::with_capacity(2);

        map.insert(&1, "a");
        map.insert(&2, "b");
        map.insert(&3, "c");

        assert_eq!(3, map.len());
        map.insert(&2, "e");
        assert_eq!(3, map.len());
        map.insert(&4, "d");
        assert_eq!(4, map.len());

        assert_eq!(map.get(&1).unwrap(), &"a");
        assert_eq!(map.get(&2).unwrap(), &"e");
        assert_eq!(map.get(&3).unwrap(), &"c");
        assert_eq!(map.get(&4).unwrap(), &"d");
    }

    #[test]
    fn get_value_when_directory_is_full() {
        let mut map = HashMap::with_capacity(2);
        let keys = vec![1, 2, 3, 4];
        let values = vec!["a", "b", "c", "d"];

        keys.iter().zip(values.iter()).for_each(|(&key, &value)| {
            map.insert(key, value);
        });
        assert_eq!(keys.len(), map.len());

        assert!(map.get(&2).is_some());
        assert!(map.get(&5).is_none());
        assert!(map.get(&10).is_none());

        keys.iter().zip(values.iter()).for_each(|(key, &value)| {
            assert!(*map.get(key).unwrap() == value);
        });
        assert_eq!(keys.len(), map.len());
    }

    #[test]
    fn remove() {
        let mut map = HashMap::with_capacity(2);

        map.insert(1, "a".to_string());
        map.insert(2, "b".to_string());
        map.insert(3, "c".to_string());

        assert_eq!(3, map.len());
        assert_eq!(map.get(&1).unwrap(), "a");
        assert_eq!(map.get(&2).unwrap(), "b");
        assert_eq!(map.get(&3).unwrap(), "c");

        assert_eq!(map.remove(&3).unwrap(), "c");
        assert!(map.get(&3).is_none());
        assert!(map.remove(&3).is_none());
        assert!(map.remove(&4).is_none());
        assert_eq!(2, map.len());

        map.insert(4, "d".to_string());
        map.insert(5, "e".to_string());
        assert_eq!(4, map.len());

        assert_eq!(map.get(&1).unwrap(), "a");
        assert_eq!(map.get(&2).unwrap(), "b");
        assert!(map.get(&3).is_none());
        assert_eq!(map.get(&4).unwrap(), "d");
        assert_eq!(map.get(&5).unwrap(), "e");
    }

    #[test]
    fn iter() {
        let mut map = HashMap::<i32, i32>::with_capacity(2);
        assert_eq!(map.iter().count(), 0);

        map.insert(1, 2);
        map.insert(2, 4);
        map.insert(3, 6);
        let mut v = map.iter().collect::<Vec<(&i32, &i32)>>();
        v.sort();
        assert_eq!(v, vec![(&1, &2), (&2, &4), (&3, &6)]);

        map.remove(&2);
        map.remove(&1);
        let mut v = map.iter().collect::<Vec<(&i32, &i32)>>();
        v.sort();
        assert_eq!(v, vec![(&3, &6)]);
    }

    #[test]
    fn keys() {
        let mut map = HashMap::<i32, i32>::with_capacity(2);
        assert_eq!(map.keys().count(), 0);

        map.insert(1, 2);
        map.insert(2, 4);
        map.insert(3, 6);
        let mut v = map.keys().collect::<Vec<&i32>>();
        v.sort();
        assert_eq!(v, vec![&1, &2, &3]);

        map.remove(&2);
        map.remove(&1);
        let mut v = map.keys().collect::<Vec<&i32>>();
        v.sort();
        assert_eq!(v, vec![&3]);
    }

    #[test]
    fn entry_key() {
        let mut map = HashMap::with_capacity(2);
        map.insert("a", "b");

        fn entry_to_option<K, V>(entry: Entry<K, V>) -> Option<Entry<K, V>> {
            match entry {
                entry @ Entry::Occupied(_) => Some(entry),
                entry @ Entry::Vacant(_) => None,
            }
        }

        let entry = entry_to_option(map.entry("a"));
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().key(), &"a");

        let entry = entry_to_option(map.entry("b"));
        assert!(entry.is_none());
    }

    #[test]
    fn entry_or_insert() {
        let mut map = HashMap::with_capacity(2);
        map.insert("a", "b");
        map.entry("a").or_insert("c");
        map.entry("b").or_insert("c");

        assert_eq!(map.get("a").unwrap(), &"b");
        assert_eq!(map.get("b").unwrap(), &"c");
    }

    #[test]
    fn into_iterator() {
        let mut map = HashMap::with_capacity(2);
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        map.insert(4, "d");
        map.remove(&3);

        let mut result = vec![];
        for kv in &map {
            result.push(kv);
        }
        result.sort_by(|a, b| (*a.0).partial_cmp(b.0).unwrap());

        assert_eq!(result, vec![(&1, &"a"), (&2, &"b"), (&4, &"d")])
    }

    #[test]
    fn drain() {
        let mut map = HashMap::with_capacity(2);
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        map.insert(4, "d");
        assert_eq!(map.len(), 4);

        let mut result = vec![];
        for kv in map.drain() {
            result.push(kv);
        }
        result.sort_by(|a, b| (a.0).partial_cmp(&(b.0)).unwrap());
        assert_eq!(result, vec![(1, "a"), (2, "b"), (3, "c"), (4, "d")]);

        assert_eq!(map.len(), 0);
        let mut result = vec![];
        for kv in &map {
            result.push(kv);
        }
        assert_eq!(result, vec![]);

        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        map.insert(4, "d");
        map.insert(5, "e");
        assert_eq!(map.len(), 5);

        let mut result = vec![];
        for kv in &map {
            result.push(kv);
        }
        result.sort_by(|a, b| (*a.0).partial_cmp(b.0).unwrap());
        assert_eq!(
            result,
            vec![(&1, &"a"), (&2, &"b"), (&3, &"c"), (&4, &"d"), (&5, &"e")]
        )
    }
}
