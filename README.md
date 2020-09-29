# mlportopt - Overview

mlportopt is a Python library for portfolio optimisation.

Using machine learning clustering techniques and non-correlation based dependence measures, the aim is to achieve better out of sample risk adjusted returns than naive risk parity and Markowitz based portfolio optimisation techniques.

## Usage

All relevant code may be imported as follows. For full documentation please refer to the Appendix of the dissertation.

```python
from mlportopt.main import *
from mlportopt.util import *
```

## Clustering

### Divisive

- Dirichlet Process Gaussian Mixture Model
- Gaussian Process Variance Function based Clustering

### Agglomerative

- Single, Average, Complete linkage
- Ward Clustering
- Bayesian Hierarcical Clustering

## Dependence Measures

- Correlation
- Mutual Information
- Variation of Information
- Copula Entropy
- Optimal Transport between empricial and reference copulas
- Wasserstein distance between fitted mixture models

## Mixture Models

- Gaussian Mixture Models
- Gauss-Gamma Mixture Models
- PENDING: Additional extreme value - Gauss MM

## Preprocessing

- Dimensionality reducing AutoEncoder
- PCA
- Self Organising (Kohonen) Map, implemented in TensorFlow
- Random Matrix Theory based similarity matrix denoising (Targeted Shrinkage and Fixed Eigenvalue cleaning of Marcenko-Pastur implied random eigenvalues)
- Beta-adjusted returns
- Detoning

## Risk Metrics

- Volatility/Sharpe
- Probabilistic Sharpe (Lopez de Prado)
- VaR and CVaR under normality, student t and fitted Gaussian/Gauss-Gamma mixture models

## Portfolio Allocation

- Risk Parity
- Markowitz Optimisation (Maximum Sharpe)
- Hierarchical Risk Parity (Lopez de Prado)
- Hierarchical Equal Risk Contribution (Thomas Raffinot)

Any other variant is possible using the above risk metrics and investing inversely proportionately. User can specify both intra and inter cluster allocation methods.


## License
[MIT](https://choosealicense.com/licenses/mit/)