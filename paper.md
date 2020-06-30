---
title: "Trendfitter: A latent variable modelling package made for process control"
tags:
  - Multivariate Statistics
  - Explorative Multivariate Analysis
  - Statistical Process Control
  - Latent Variable Modelling
  - Principal Component Analysis
  - Multi-block Principal Components Analysis
  - Partial Least Squares
  - Multi-block Partial Least Squares
authors:
 - name: Daniel de Araújo Costa Rodrigues
   orcid: 0000-0002-8610-4374
   affiliation: 1
affiliations:
- name: Université Laval, Québec, Canada
  index: 1
date: 29/06/2020
bibliography: paper.bib
---

# Summary

Trendfitter is a latent variable modelling package made for multivariate statistical process control. Some of the methods implemented in this library are already available in other packages, but not with all the tools here available for investigation, exploration and prediction applied to industrial production processes. Additionally, the methods here follow an object-oriented approach similar to scikit-learn so that one can explore combining tools existing in that package without having the need to adapt the code.

This first version contains:
- Principal Component Analysis (PCA)
- Multi-Block Principal Component Analysis (MB-PCA)
- Partial Least Squares or Projection to Latent Structures (PLS)
- Multiblock PLS (MB-PLS)
- Sequential MBPLS (SMB-PLS)
- Sequential Orthogonal PLS (SO-PLS)
- Dynamic PCA (DyPCA)
- Dynamic PLS (DyPLS)

The models have functions for calculation of scores, loadings, weights, contributions, Hotelling's T², and Squared Prediction Errors for both models with single blocks and multiblocks with multiple levels. Functions to evaluate the confidence intervals are also available together with multiple functions to deal with missing data where available (PCA, MB-PCA, PLS, MB-PLS), a frequent problem in industrial process data.


The hoggorm package provides access to an extended repertoire of interpretation tools that are integrated in PCA, PCR, PLS1 and PLS2. These including scores, loadings, correlation loadings, explained variances for calibrated and validated models (both for individual variables as well as all variables together). Scores are the objects' coordinates in the compressed data representation and can for instance be used to search for patterns or groups among the objects. Loadings are the variables' representations in the compressed space showing their contribution to the components. Finally, correlation loadings show how each variable correlates to the score vectors/components and how much of the variation in each variable is explained across components. Note that models trained with hoggorm may also be applied for prediction purposes, both for continuous and categorical variables, where appropriate.

# Acknowledgements


# References
