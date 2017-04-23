# K-SVD

An K-SVD implementaion written in Python.

[![Build Status](https://travis-ci.org/nel215/ksvd.svg?branch=master)](https://travis-ci.org/nel215/ksvd)
[![PyPI](https://img.shields.io/pypi/v/ksvd.svg)](https://pypi.python.org/pypi/ksvd)

## Installation

```
pip install ksvd
```

## Usage

```python
import numpy as np
from ksvd import ApproximateKSVD


# X ~ gamma.dot(dictionary)
X = np.random.randn(1000, 20)
aksvd = ApproximateKSVD(n_components=128)
dictionary = aksvd.fit(X).components_
gamma = aksvd.transform(X)
```

## Feature

- Approximate K-SVD

## Example

- [nel215/image-noise-reduction](https://github.com/nel215/image-noise-reduction)

## License

Licensed under the Apache License 2.0.

## References

- [Rubinstein, R., Zibulevsky, M. and Elad, M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit Technical Report - CS Technion, April 2008](http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf)
