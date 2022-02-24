![CI](https://github.com/timsainb/automutualinformation/actions/workflows/python-package.yml/badge.svg)
![package](https://github.com/timsainb/automutualinformation/actions/workflows/python-publish.yml/badge.svg)


AutoMutualInformation
==============================

Auto Mutual Information (Sequential Mutual Information) for temporal data. 

Auto mutual information can be treated as the equivalent of autocorrelation for symbolic data.

### Installation

The python package is installable via pip. 

`pip install automutualinformation`

### Quick Start

```python
from automutualinformation import sequential_mutual_information as smi
(MI, MI_var), (shuff_MI, shuff_MI_var) = smi(
    [signal], distances=np.arange(1,100)
)
```

### Documentation

Documentation and usage information is currently available in jupyter notebooks in the notebooks folder. 

### Citation

If you use this package, please cite the following paper:

```
@article {NBC2020,
    author = {Sainburg, Tim and Mai, Anna and Gentner, Timothy Q.},
    title = {Long-range sequential dependencies precede complex syntactic production in language acquisition},
    journal = {Proceedings of the Royal Society B},
    doi = {https://dx.doi.org/10.1098/rspb.2021.2657},
    year = 2022,
    }
```

### TODO

- make pypi package
- create tests/travisci
- add additional parameters example


For more info references see:

- [Mutual information functions versus correlation functions. W Li. (1990). Journal of Statistical Physics](https://doi.org/10.1007/BF01025996)
- [Critical Behavior in Physics and Probabilistic Formal Languages. HW Lin, M Tegmark (2017) Entropy](https://doi.org/10.3390/e19070299)
- [Parallels in the sequential organization of birdsong and human speech. T Sainburg, B Thielman, M Thielk, TQ Gentner, (2019) Nature Communications](https://doi.org/10.1038/s41467-019-11605-y)
- [Long-range sequential dependencies precede complex syntactic production in language acquisition. T Sainburg, A Mai, TQ Gentner. Proceedings of the Royal Society B](https://dx.doi.org/10.1098/rspb.2021.2657)
