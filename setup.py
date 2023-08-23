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
# ==============================================================================
"""Setup Xarray-Tensorstore."""
import setuptools


setuptools.setup(
    name='xarray-tensorstore',
    version='0.1.1',  # keep in sync with xarray_tensorstore.py
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=['numpy', 'xarray', 'zarr', 'tensorstore'],
    extras_require={
        'tests': ['absl-py', 'dask', 'pandas', 'pytest'],
    },
    url='https://github.com/google/xarray-tensorstore',
    py_modules=['xarray_tensorstore'],
    python_requires='>=3.10',
)
