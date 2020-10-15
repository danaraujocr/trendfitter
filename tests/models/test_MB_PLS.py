import pandas as pd
from numpy import array, max
import pytest
import sys
import os
sys.path.append(".")
from trendfitter.models import MB_PLS

TESTDATA1_FILENAME = os.path.join(os.path.dirname(__file__), 'pls_dataset_complete.csv')
#TESTDATA2_FILENAME = os.path.join(os.path.dirname(__file__), 'pls_test_dataset_incomplete.csv')

class TestMBPCA(object):

    def test_fit_fulldataset(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, delimiter=';', index_col=[0]).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = MB_PLS()
        block_divs = [10, 21]
        test_model.fit(X_test_data, block_divs, Y_test_data, latent_variables=2)

        block_p_loadings = array([[-0.48853912, -0.07306347, -0.38223245,  0.25388191, -0.2861673,  -0.43993095,
                                   -0.37076898, -0.58510196,  0.12397263, -0.13425598,  0.,          0.,
                                    0.,          0.,          0.,          0.,          0.,          0.,
                                    0.,          0.,          0.,        ],
                                  [ 0.16566012,  0.17496217,  0.67461113,  0.29920494,  0.41527515,  0.68441669,
                                   -0.4200502,   0.41768707, -0.06280092,  0.42851025,  0.,          0.,
                                    0.,          0.,          0.,          0.,          0.,          0.,
                                    0.,          0.,          0.,        ],
                                  [ 0.,          0.,          0.,          0.,          0.,          0.,
                                    0.,          0.,          0.,          0.,         -0.66496878, -0.83493718,
                                    0.10712982, -0.12658335, -0.84771659, -0.05145963, -0.15193708,  0.24404461,
                                   -0.62518209, -0.17670201,  0.41900848],
                                  [ 0.,          0.,          0.,          0.,          0.,          0.,
                                    0.,          0.,          0.,          0.,          0.66586915,  0.71232674,
                                   -0.41013788,  0.02956861,  0.42072848, -0.28288597,  0.01078709, -0.42706078,
                                    0.14352652,  0.38603257, -0.39322403]])

        assert test_model.block_p_loadings == pytest.approx(block_p_loadings), 'Block loadings are not as expected'

        block_weights = array([[-0.38391965, -0.09236524, -0.26024297,  0.23208795, -0.19929187, -0.33394883,
                                -0.34659546, -0.48359965,  0.10296577, -0.14243939,  0.,          0.,
                                 0.,          0.,          0.,          0.,          0.,          0.,
                                 0.,          0.,          0.,        ],
                               [ 0.02867525, -0.04920307,  0.38022892,  0.2348004,   0.24956067,  0.30055631,
                                -0.48663128,  0.14915648,  0.00389374,  0.23637791,  0.,          0.,
                                 0.,          0.,          0.,          0.,          0.,          0.,
                                 0.,          0.,          0.,        ],
                               [ 0.,          0.,          0.,          0.,          0.,          0.,
                                 0.,          0.,          0.,          0.,         -0.18986493, -0.29474327,
                                -0.03496983, -0.05700808, -0.42094606, -0.0704986,  -0.03821775,  0.04818109,
                                -0.30362093,  0.01200999,  0.13949998],
                               [ 0.,          0.,          0.,          0.,          0.,          0.,
                                 0.,          0.,          0.,          0.,          0.37235114,  0.38134085,
                                -0.10644904, -0.03923513,  0.27707641, -0.059204,    0.10436763, -0.30675226,
                                 0.05878066,  0.22503696, -0.19624219]])
        assert test_model.block_weights == pytest.approx(block_weights), 'Block weights are not as expected'

        superlevel_weights = array([[0.81364389, 0.4251329,  0.,         0.,        ],
                                    [0.,         0.,         0.67051044, 0.57481081]])
        assert test_model.superlevel_weights == pytest.approx(superlevel_weights), 'superlevel weights are not as expected'

        splevel_scores = array([[ 0.60370323,  0.03092709,  1.29055187,  0.3291472 ],
                                [ 1.71641593,  0.39253751,  1.69022119,  0.51452702],
                                [-0.68002489, -0.48640048,  0.73747276,  0.31994962],
                                [-0.95209031, -0.74867712,  1.39195383,  0.68879973],
                                [-0.13324219, -0.38829141,  1.24327356,  0.35976336]])

        assert test_model.training_sl_scores[:5] == pytest.approx(splevel_scores), 'superlevel scores are not as expected'

 

    def test_transform_b_fulldata(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, delimiter=';', index_col=[0]).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = MB_PLS()
        block_divs = [10, 21]
        test_model.fit(X_test_data, block_divs, Y_test_data, latent_variables=2)

        expected = array([[ 0.60370323,  1.29055187],
                          [ 1.71641593,  1.69022119],
                          [-0.68002489,  0.73747276],
                          [-0.95209031,  1.39195383],
                          [-0.13324219,  1.24327356]])

        assert expected == pytest.approx(test_model.transform_b(X_test_data.iloc[:5, :10], 0)), 'Prediction of block X values unsuccessful'
    
    def test_score_b(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, delimiter=';', index_col=[0]).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = MB_PLS()
        block_divs = [10, 21]
        test_model.fit(X_test_data, block_divs, Y_test_data, latent_variables=2)
        
        expected = 0.38871942011418226
        
        assert expected == pytest.approx(test_model.score_b(X_test_data.iloc[:,:10], 0)), "block r² Score calculation malfunctioning"

