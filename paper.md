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


The models have functions for calculation of scores, loadings, weights, contributions, Hotelling's T², and Squared Prediction Errors for both models with single blocks and multiblocks. Functions to evaluate the confidence intervals are also available together with multiple functions to deal with missing data where available (PCA, MB-PCA, PLS, MB-PLS), a frequent problem in industrial process data.



# Acknowledgements


# References
