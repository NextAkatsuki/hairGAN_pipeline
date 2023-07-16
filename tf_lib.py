import tensorflow as tf
import multiprocessing


def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset


def disk_image_batch_dataset(img_paths,
                             batch_size,
                             labels=None,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """Batch dataset of disk image for PNG and JPEG.

    Parameters
    ----------
    img_paths : 1d-tensor/ndarray/list of str
    labels : nested structure of tensors/ndarrays/lists

    """
    if labels is None:
        memory_data = img_paths
    else:
        memory_data = (img_paths, labels)

    def parse_fn(path, *label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)  # fix channels to 3
        return (img,) + label

    if map_fn:  # fuse `map_fn` and `parse_fn`
        def map_fn_(*args):
            return map_fn(*parse_fn(*args))
    else:
        map_fn_ = parse_fn

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat)

    return dataset


class Checkpoint:
    """Enhanced "tf.train.Checkpoint"."""

    def __init__(self,
                 checkpoint_kwargs,  # for "tf.train.Checkpoint"
                 directory,  # for "tf.train.CheckpointManager"
                 max_to_keep=5,
                 keep_checkpoint_every_n_hours=None):
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep, keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            self.__getattribute__(attr)  # this will raise an exception


def summary(name_data_dict,
            step=None,
            types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
            historgram_buckets=None,
            name='summary'):
    """Summary.

    Examples
    --------
    >>> summary({'a': data_a, 'b': data_b})

    """
    def _summary(name, data):
        if data.shape == ():
            tf.summary.scalar(name, data, step=step)
        else:
            if 'mean' in types:
                tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step)
            if 'std' in types:
                tf.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step)
            if 'max' in types:
                tf.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step)
            if 'min' in types:
                tf.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step)
            if 'sparsity' in types:
                tf.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data), step=step)
            if 'histogram' in types:
                tf.summary.histogram(name, data, step=step, buckets=historgram_buckets)

    with tf.name_scope(name):
        for name, data in name_data_dict.items():
            _summary(name, data)
