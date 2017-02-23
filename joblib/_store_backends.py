"""Storage providers backends for Memory caching."""

import re
import os
import os.path
import time
import datetime
import json
import shutil
import warnings
import collections
import operator
from abc import ABCMeta, abstractmethod
from ._compat import with_metaclass, _basestring
from .logger import format_time
from .disk import mkdirp, memstr_to_bytes
from . import numpy_pickle

CacheItemInfo = collections.namedtuple('CacheItemInfo',
                                       'path size last_access')


class StoreBackendBase(with_metaclass(ABCMeta)):
    """Helper abc which defines all methods a StorageBackend must implement."""

    open_object = None
    object_exists = None

    @abstractmethod
    def create_location(self, location):
        """Create location on store."""

    @abstractmethod
    def clear_location(self, obj):
        """Clear object on store"""

    @abstractmethod
    def get_cache_items(self):
        """Returns the whole list of items available in cache."""

    @abstractmethod
    def configure(self, location, *args, **kwargs):
        """Configure the store"""


class StoreManagerMixin(object):
    """Class providing all logic for managing the cache in a generic way.

    The StoreBackend subclass has to implement 3 methods: create_location,
    clear_location and configure. The StoreBackend also has to provide
    open_object and object_exists methods by monkey matching them
    in the configure method. The open_object method has to return a file-like
    object.

    All values are cached on a filesystem-like backend, in a deep directory
    structure.

    """

    def load_result(self, func_id, args_id, **kwargs):
        """Load computation output from store."""
        full_path = os.path.join(self.cachedir, func_id, args_id)
        filename = os.path.join(full_path, 'output.pkl')
        if not self.object_exists(filename):
            raise KeyError("Non-existing cache value (may have been "
                           "cleared).\nFile %s does not exist" % filename)

        if 'verbose' in kwargs and kwargs['verbose'] > 1:
            verbose = kwargs['verbose']
            signature = ""
            try:
                if 'metadata' in kwargs and kwargs['metadata'] is not None:
                    metadata = kwargs['metadata']
                    args = ", ".join(['%s=%s' % (name, value)
                                      for name, value
                                      in metadata['input_args'].items()])
                    signature = "%s(%s)" % (os.path.basename(func_id), args)
                else:
                    signature = os.path.basename(func_id)
            except KeyError:
                pass

            if 'timestamp' in kwargs and kwargs['timestamp'] is not None:
                ts_string = ("{0: <16}"
                             .format(format_time(time.time() -
                                                 kwargs['timestamp'])))
            else:
                ts_string = ""

            if verbose < 10:
                print('[Memory]{0}: Loading {1}...'.format(ts_string,
                                                           str(signature)))
            else:
                print('[Memory]{0}: '
                      'Loading {1} from {2}'.format(ts_string,
                                                    str(signature),
                                                    full_path))

        mmap_mode = None if 'mmap_mode' not in kwargs else kwargs['mmap_mode']

        # file-like object cannot be used when mmap_mode is set
        if mmap_mode is None:
            with self.open_object(filename, "rb") as f:
                result = numpy_pickle.load(f)
        else:
            result = numpy_pickle.load(filename, mmap_mode=mmap_mode)
        return result

    def dump_result(self, func_id, args_id, result, compress=False, **kwargs):
        """Dump computation output in store."""
        try:
            result_dir = os.path.join(self.cachedir, func_id, args_id)
            if not self.object_exists(result_dir):
                self.create_location(result_dir)
            filename = os.path.join(result_dir, 'output.pkl')
            if 'verbose' in kwargs and kwargs['verbose'] > 10:
                print('Persisting in %s' % result_dir)

            with self.open_object(filename, "wb") as f:
                numpy_pickle.dump(result, f, compress=compress)
        except:
            " Race condition in the creation of the directory "

    def clear_result(self, func_id, args_id):
        """Clear computation output in store."""
        result_dir = os.path.join(self.cachedir, func_id, args_id)
        if self.object_exists(result_dir):
            self.clear_location(result_dir)

    def contains_result(self, func_id, args_id, **kwargs):
        """Check computation output is available in store."""
        result_dir = os.path.join(self.cachedir, func_id, args_id)
        filename = os.path.join(result_dir, 'output.pkl')

        return self.object_exists(filename)

    def get_result_info(self, func_id, args_id):
        """Return information about cached result."""
        return {'location': os.path.join(self.cachedir, func_id, args_id)}

    def get_metadata(self, func_id, args_id):
        """Return actual metadata of a computation."""
        try:
            directory = os.path.join(self.cachedir, func_id, args_id)
            filename = os.path.join(directory, 'metadata.json')
            with self.open_object(filename, 'rb') as f:
                return json.loads(f.read().decode('utf-8'))
        except:
            return {}

    def store_metadata(self, func_id, args_id, metadata):
        """Store metadata of a computation."""
        try:
            directory = os.path.join(self.cachedir, func_id, args_id)
            self.create_location(directory)
            with self.open_object(os.path.join(directory, 'metadata.json'),
                                  'wb') as f:
                f.write(json.dumps(metadata).encode('utf-8'))
        except:
            pass

    def contains_cached_func(self, func_id):
        """Check cached function is available in store."""
        func_dir = os.path.join(self.cachedir, func_id)
        return self.object_exists(func_dir)

    def clear_cached_func(self, func_id):
        """Clear all references to a function in the store."""
        func_dir = os.path.join(self.cachedir, func_id)
        if self.object_exists(func_dir):
            self.clear_location(func_dir)

    def store_cached_func_code(self, func_id, func_code=None):
        """Store the code of the cached function."""
        func_dir = os.path.join(self.cachedir, func_id)
        if not self.object_exists(func_dir):
            self.create_location(func_dir)

        if func_code is not None:
            filename = os.path.join(func_dir, "func_code.py")
            with self.open_object(filename, 'wb') as f:
                f.write(func_code.encode('utf-8'))

    def get_cached_func_code(self, func_id):
        """Store the code of the cached function."""
        filename = os.path.join(self.cachedir, func_id, "func_code.py")
        try:
            with self.open_object(filename, 'rb') as f:
                return f.read().decode('utf-8')
        except:
            raise

    def get_cached_func_info(self, func_id):
        """Return information related to the cached function if it exists."""
        return {'location': os.path.join(self.cachedir, func_id)}

    def clear(self):
        """Clear the whole store content."""
        self.clear_location(self.cachedir)

    def reduce_cache_size(self, bytes_limit):
        """Reduce cache size to keep it under the given bytes limit."""
        cache_items_to_delete = self._get_cache_items_to_delete(
            bytes_limit)

        for cache_item in cache_items_to_delete:
            if self.verbose > 10:
                print('Deleting cache item {0}'.format(cache_item))
            try:
                self.clear_location(cache_item.path)
            except OSError:
                # Even with ignore_errors=True can shutil.rmtree
                # can raise OSErrror with [Errno 116] Stale file
                # handle if another process has deleted the folder
                # already.
                pass

    def _get_cache_items_to_delete(self, bytes_limit):
        """Get cache items to delete to keep the cache under a size limit."""
        if isinstance(bytes_limit, _basestring):
            bytes_limit = memstr_to_bytes(bytes_limit)

        cache_items = self.get_cache_items()
        cache_size = sum(item.size for item in cache_items)

        to_delete_size = cache_size - bytes_limit
        if to_delete_size < 0:
            return []

        # We want to delete first the cache items that were accessed a
        # long time ago
        cache_items.sort(key=operator.attrgetter('last_access'))

        cache_items_to_delete = []
        size_so_far = 0

        for item in cache_items:
            if size_so_far > to_delete_size:
                break

            cache_items_to_delete.append(item)
            size_so_far += item.size

        return cache_items_to_delete

    def __repr__(self):
        """Printable representation of the store location."""
        return self.cachedir


class FileSystemStoreBackend(StoreBackendBase, StoreManagerMixin):
    """A StoreBackend used with local or network file systems."""

    def clear_location(self, location):
        """Delete location on store."""
        shutil.rmtree(location, ignore_errors=True)

    def create_location(self, location):
        """Create object location on store"""
        mkdirp(location)

    def get_cache_items(self):
        """Returns the whole list of items available in cache."""
        cache_items = []

        for dirpath, dirnames, filenames in os.walk(self.cachedir):
            is_cache_hash_dir = re.match('[a-f0-9]{32}',
                                         os.path.basename(dirpath))

            if is_cache_hash_dir:
                output_filename = os.path.join(dirpath, 'output.pkl')
                try:
                    last_access = os.path.getatime(output_filename)
                except OSError:
                    try:
                        last_access = os.path.getatime(dirpath)
                    except OSError:
                        # The directory has already been deleted
                        continue

                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn)
                                      for fn in filenames]
                    dirsize = sum(os.path.getsize(fn)
                                  for fn in full_filenames)
                except OSError:
                    # Either output_filename or one of the files in
                    # dirpath does not exist any more. We assume this
                    # directory is being cleaned by another process already
                    continue

                cache_items.append(CacheItemInfo(dirpath, dirsize,
                                                 last_access))

        return cache_items

    def configure(self, location, **kwargs):
        """Configure the store backend."""

        # attach required methods using monkey patching trick.
        self.open_object = open
        self.object_exists = os.path.exists
        self.cachedir = os.path.join(location, 'joblib')

        if not os.path.exists(self.cachedir):
            mkdirp(self.cachedir)

        # computation results can be stored compressed for faster I/O
        self.compress = (False if 'compress' not in kwargs
                         else kwargs['compress'])

        # FileSystemStoreBackend can be used with mmap_mode options under
        # certain conditions.
        mmap_mode = None
        if 'mmap_mode' in kwargs:
            mmap_mode = kwargs['mmap_mode']
            if self.compress and mmap_mode is not None:
                warnings.warn('Compressed results cannot be memmapped in a '
                              'filesystem store. Option will be ignored.',
                              stacklevel=2)

        self.mmap_mode = mmap_mode

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
