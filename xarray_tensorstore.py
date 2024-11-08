# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for loading TensorStore data into Xarray."""
from __future__ import annotations

import dataclasses
import math
import os.path
import re
from typing import Optional, TypeVar

import numpy as np
import tensorstore
import xarray
from xarray.core import indexing


__version__ = '0.1.5'  # keep in sync with setup.py


Index = TypeVar('Index', int, slice, np.ndarray, None)
XarrayData = TypeVar('XarrayData', xarray.Dataset, xarray.DataArray)


def _numpy_to_tensorstore_index(index: Index, size: int) -> Index:
  """Switch from NumPy to TensorStore indexing conventions."""
  # https://google.github.io/tensorstore/python/indexing.html#differences-compared-to-numpy-indexing
  if index is None:
    return None
  elif isinstance(index, int):
    # Negative integers do not count from the end in TensorStore
    return index + size if index < 0 else index
  elif isinstance(index, slice):
    start = _numpy_to_tensorstore_index(index.start, size)
    stop = _numpy_to_tensorstore_index(index.stop, size)
    if stop is not None:
      # TensorStore does not allow out of bounds slicing
      stop = min(stop, size)
    return slice(start, stop, index.step)
  else:
    assert isinstance(index, np.ndarray)
    return np.where(index < 0, index + size, index)


@dataclasses.dataclass(frozen=True)
class _TensorStoreAdapter(indexing.ExplicitlyIndexed):
  """TensorStore array that can be wrapped by xarray.Variable.

  We use Xarray's semi-internal ExplicitlyIndexed API so that Xarray will not
  attempt to load our array into memory as a NumPy array. In the future, this
  should be supported by public Xarray APIs, as part of the refactor discussed
  in: https://github.com/pydata/xarray/issues/3981
  """

  array: tensorstore.TensorStore
  future: Optional[tensorstore.Future] = None

  @property
  def shape(self) -> tuple[int, ...]:
    return self.array.shape

  @property
  def dtype(self) -> np.dtype:
    return self.array.dtype.numpy_dtype

  @property
  def ndim(self) -> int:
    return len(self.shape)

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  def __getitem__(self, key: indexing.ExplicitIndexer) -> _TensorStoreAdapter:
    index_tuple = tuple(map(_numpy_to_tensorstore_index, key.tuple, self.shape))
    if isinstance(key, indexing.OuterIndexer):
      # TODO(shoyer): fix this for newer versions of Xarray.
      # We get the error message:
      # AttributeError: '_TensorStoreAdapter' object has no attribute 'oindex'
      indexed = self.array.oindex[index_tuple]
    elif isinstance(key, indexing.VectorizedIndexer):
      indexed = self.array.vindex[index_tuple]
    else:
      assert isinstance(key, indexing.BasicIndexer)
      indexed = self.array[index_tuple]
    # Translate to the origin so repeated indexing is relative to the new bounds
    # like NumPy, not absolute like TensorStore
    translated = indexed[tensorstore.d[:].translate_to[0]]
    return type(self)(translated)

  def __setitem__(self, key: indexing.ExplicitIndexer, value) -> None:
    index_tuple = tuple(map(_numpy_to_tensorstore_index, key.tuple, self.shape))
    if isinstance(key, indexing.OuterIndexer):
      self.array.oindex[index_tuple] = value
    elif isinstance(key, indexing.VectorizedIndexer):
      self.array.vindex[index_tuple] = value
    else:
      assert isinstance(key, indexing.BasicIndexer)
      self.array[index_tuple] = value
    # Invalidate the future so that the next read will pick up the new value
    object.__setattr__(self, 'future', None)

  # xarray>2024.02.0 uses oindex and vindex properties, which are expected to
  # return objects whose __getitem__ method supports the appropriate form of
  # indexing.
  @property
  def oindex(self) -> _TensorStoreAdapter:
    return self

  @property
  def vindex(self) -> _TensorStoreAdapter:
    return self

  def transpose(self, order: tuple[int, ...]) -> _TensorStoreAdapter:
    transposed = self.array[tensorstore.d[order].transpose[:]]
    return type(self)(transposed)

  def read(self) -> _TensorStoreAdapter:
    future = self.array.read()
    return type(self)(self.array, future)

  def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:  # type: ignore
    future = self.array.read() if self.future is None else self.future
    return np.asarray(future.result(), dtype=dtype)

  def get_duck_array(self):
    # special method for xarray to return an in-memory (computed) representation
    return np.asarray(self)

  # Work around the missing __copy__ and __deepcopy__ methods from TensorStore,
  # which are needed for Xarray:
  # https://github.com/google/tensorstore/issues/109
  # TensorStore objects are immutable, so there's no need to actually copy them.

  def __copy__(self) -> _TensorStoreAdapter:
    return type(self)(self.array, self.future)

  def __deepcopy__(self, memo) -> _TensorStoreAdapter:
    return self.__copy__()


def _read_tensorstore(
    array: indexing.ExplicitlyIndexed,
) -> indexing.ExplicitlyIndexed:
  """Starts async reading on a TensorStore array."""
  return array.read() if isinstance(array, _TensorStoreAdapter) else array


def read(xarraydata: XarrayData, /) -> XarrayData:
  """Starts async reads on all TensorStore arrays."""
  # pylint: disable=protected-access
  if isinstance(xarraydata, xarray.Dataset):
    data = {
        name: _read_tensorstore(var.variable._data)
        for name, var in xarraydata.data_vars.items()
    }
  elif isinstance(xarraydata, xarray.DataArray):
    data = _read_tensorstore(xarraydata.variable._data)
  else:
    raise TypeError(f'argument is not a DataArray or Dataset: {xarraydata}')
  # pylint: enable=protected-access
  return xarraydata.copy(data=data)


_DEFAULT_STORAGE_DRIVER = 'file'


def _zarr_spec_from_path(path: str) -> ...:
  if re.match(r'\w+\://', path):  # path is a URI
    kv_store = path
  else:
    kv_store = {'driver': _DEFAULT_STORAGE_DRIVER, 'path': path}
  return {'driver': 'zarr', 'kvstore': kv_store}


def _raise_if_mask_and_scale_used_for_data_vars(ds: xarray.Dataset):
  """Check a dataset for data variables that would need masking or scaling."""
  advice = (
      'Consider re-opening with xarray_tensorstore.open_zarr(..., '
      'mask_and_scale=False), or falling back to use xarray.open_zarr().'
  )
  for k in ds:
    encoding = ds[k].encoding
    for attr in ['_FillValue', 'missing_value']:
      fill_value = encoding.get(attr, np.nan)
      if fill_value == fill_value:  # pylint: disable=comparison-with-itself
        raise ValueError(
            f'variable {k} has non-NaN fill value, which is not supported by'
            f' xarray-tensorstore: {fill_value}. {advice}'
        )
    for attr in ['scale_factor', 'add_offset']:
      if attr in encoding:
        raise ValueError(
            f'variable {k} uses scale/offset encoding, which is not supported'
            f' by xarray-tensorstore: {encoding}. {advice}'
        )


def open_zarr(
    path: str,
    *,
    context: tensorstore.Context | None = None,
    mask_and_scale: bool = True,
    write: bool = False,
) -> xarray.Dataset:
  """Open an xarray.Dataset from Zarr using TensorStore.

  For best performance, explicitly call `read()` to asynchronously load data
  in parallel. Otherwise, xarray's `.compute()` method will load each variable's
  data in sequence.

  Example usage:

    import xarray_tensorstore

    ds = xarray_tensorstore.open_zarr(path)

    # indexing & transposing is lazy
    example = ds.sel(time='2020-01-01').transpose('longitude', 'latitude', ...)

    # start reading data asynchronously
    read_example = xarray_tensorstore.read(example)

    # blocking conversion of the data into NumPy arrays
    numpy_example = read_example.compute()

  Args:
    path: path or URI to Zarr group to open.
    context: TensorStore configuration options to use when opening arrays.
    mask_and_scale: if True (default), attempt to apply masking and scaling like
      xarray.open_zarr(). This is only supported for coordinate variables and
      otherwise will raise an error.
    write: Allow write access. Defaults to False.

  Returns:
    Dataset with all data variables opened via TensorStore.
  """
  # We use xarray.open_zarr (which uses Zarr Python internally) to open the
  # initial version of the dataset for a few reasons:
  # 1. TensorStore does not support Zarr groups or array attributes, which we
  #    need to open in the xarray.Dataset. We use Zarr Python instead of
  #    parsing the raw Zarr metadata files ourselves.
  # 2. TensorStore doesn't support non-standard Zarr dtypes like UTF-8 strings.
  # 3. Xarray's open_zarr machinery does some pre-processing (e.g., from numeric
  #    to datetime64 dtypes) that we would otherwise need to invoke explicitly
  #    via xarray.decode_cf().
  #
  # Fortunately (2) and (3) are most commonly encountered on small coordinate
  # arrays, for which the performance advantages of TensorStore are irrelevant.

  if context is None:
    context = tensorstore.Context()

  # chunks=None means avoid using dask
  ds = xarray.open_zarr(path, chunks=None, mask_and_scale=mask_and_scale)

  if mask_and_scale:
    # Data variables get replaced below with _TensorStoreAdapter arrays, which
    # don't get masked or scaled. Raising an error avoids surprising users with
    # incorrect data values.
    _raise_if_mask_and_scale_used_for_data_vars(ds)

  specs = {k: _zarr_spec_from_path(os.path.join(path, k)) for k in ds}
  array_futures = {
      k: tensorstore.open(spec, read=True, write=write, context=context)
      for k, spec in specs.items()
  }
  arrays = {k: v.result() for k, v in array_futures.items()}
  new_data = {k: _TensorStoreAdapter(v) for k, v in arrays.items()}

  return ds.copy(data=new_data)
