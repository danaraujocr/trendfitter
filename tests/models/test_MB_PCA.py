import pandas as pd
from numpy import array, max
import pytest
import sys
import os
sys.path.append(".")
from trendfitter.models import MB_PCA

TESTDATA1_FILENAME = os.path.join(os.path.dirname(__file__), 'pca_test_dataset_complete.csv')
TESTDATA2_FILENAME = os.path.join(os.path.dirname(__file__), 'pca_test_dataset_incomplete.csv')

class TestMBPCA(object):

    def test_fit_fulldataset(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = MB_PCA()
        test_model.fit(test_data, [2, 5])

        b_loadings = array([[-0.45753343,  0.4787455,  -0.53238767,  0.50447688, -0.15340262],
                            [-0.37043885,  0.35674997,  0.19766103, -0.22123992,  0.8046661 ]])
        assert test_model.block_loadings[:2, :] == pytest.approx(b_loadings), 'Block loadings are not as expected'

        splevel_loadings = array([[ 0.43853409,  0.56146591,  0.3402806,  -0.3402806, ],
                                  [ 0.3402806,  -0.3402806,   0.26449548,  0.73550452,]])
        assert test_model.superlevel_loadings == pytest.approx(splevel_loadings), 'superlevel loadings are not as expected'

        splevel_scores = array([[ 0.57628739,  0.80692377,  0.44244208, -1.06184817],
                                [-0.90296265, -1.89151428, -0.68209329,  1.03581792],
                                [ 0.33565122, -0.57320763,  0.26868383,  0.59277439],
                                [ 0.38422145,  1.55511912,  0.29561357, -1.43735021],
                                [ 0.71067313,  0.55726393,  0.54628867,  0.11717591]])

        assert test_model.superlevel_training_scores[:5] == pytest.approx(splevel_scores), 'superlevel scores are not as expected'

 
 
    def test_predict_b_fulldata(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = MB_PCA()
        test_model.fit(test_data, [2, 5])

        expected = array([[-0.42756848,  0.43373619],
                          [ 0.66580945, -0.67562606],
                          [-0.25310258,  0.25654446],
                          [-0.28530091,  0.28940442],
                          [-0.52752326,  0.53512003]])

        assert expected == pytest.approx(test_model.predict_b(test_data.iloc[:5, :2], 0)), 'Prediction of block X values unsuccessful'
    
    
    def test_predict_b_missingdata(self):
        
        test_data = pd.read_csv(TESTDATA2_FILENAME, index_col = 0, delimiter = ';')
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = MB_PCA()
        test_model.fit(test_data, [2, 5])

        expected = array([[-0.40119088,  0.39775356],
                          [ 0.62157011, -0.61701447],
                          [-0.240606,    0.23778596],
                          [-0.26787863,  0.26554006],
                          [-0.49517005,  0.49088111]])

        assert expected == pytest.approx(test_model.predict_b(test_data.iloc[:5, :2], 0)), 'Prediction of X values with missing data unsuccessful'
    
    
    def test_transform_b_fulldata(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = MB_PCA()
        test_model.fit(test_data, [2, 5])
        
        expected = array([[ 0.57628739,  0.44244208],
                          [-0.90296265, -0.68209329],
                          [ 0.33565122,  0.26868383],
                          [ 0.38422145,  0.29561357],
                          [ 0.71067313,  0.54628867]])

        assert expected == pytest.approx(test_model.transform_b(test_data.iloc[:5, :2], 0)), "Transformation to block scores is malfunctioning "

    def test_transform_b_missingdata(self):

        test_data = pd.read_csv(TESTDATA2_FILENAME, index_col = 0, delimiter = ';')
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = MB_PCA()
        test_model.fit(test_data, [2, 5])

        expected = array([[ 0.58309938,  0.38480606],
                          [-0.91347032, -0.58249535],
                          [ 0.33978214,  0.24426879],
                          [ 0.38877249,  0.25771076],
                          [ 0.71908363,  0.47577261]])

        assert expected == pytest.approx(test_model.transform_b(test_data.iloc[:5, :2], 0))

    
    def test_score_b(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = MB_PCA()
        test_model.fit(test_data, [2, 5])
        
        expected = 0.7977328443400561
        
        assert expected == pytest.approx(test_model.score_b(test_data.iloc[:, :2], 0)), "block r² Score calculation malfunctioning"
        
