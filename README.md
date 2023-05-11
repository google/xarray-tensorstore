# Xarray-TensorStore

Xarray-TensorStore is a small library that allows opening Zarr arrays into
Xarray via TensorStore, instead of the standard Zarr-Python library. In some
cases, we've found it to be considerably faster.

**Warning**: Xarray-TensorStore relies upon internal Xarray APIs that will
likely change in
[future versions of Xarray](https://github.com/pydata/xarray/issues/3981),
precisely to accommodate these sorts of use-cases. Expect that the current
version of Xarray-TensorStore will break at some point in the future and require
updates for a new Xarray release.

## Installation

Xarray-TensorStore is available on pypi:
```
pip install xarray-tensorstore
```

## Usage

Open your Zarr files into an `xarray.Dataset` using `open_zarr()`, and then use
`read()` to start reading data in the background:

```python
import xarray_tensorstore

ds = xarray_tensorstore.open_zarr(path)

# As with xarray.open_zarr(), indexing & transposing is lazy
example = ds.sel(time='2020-01-01').transpose('longitude', 'latitude', ...)

# Optional: start reading data in all arrays asynchronously
read_example = xarray_tensorstore.read(example)

# Blocking conversion of the data into NumPy arrays. This happens sequentially,
# one array at a time, unless you call read() first.
numpy_example = read_example.compute()
```

## Limitations

- Xarray-TensorStore still uses Zarr-Python under the covers to open Zarr
  groups and read coordinate data (TensorStore does not yet support Zarr
  groups).
- Unlike `xarray.open_zarr`, decoding of data arrays according to CF Conventions
  (e.g., `scale` and `add_offset` attributes) is not supported.
