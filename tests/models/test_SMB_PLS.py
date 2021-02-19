import pandas as pd
from numpy import array, max, sqrt
import pytest
import sys
import os
sys.path.append(".")
from trendfitter.models import SMB_PLS

TESTDATA1_FILENAME = os.path.join(os.path.dirname(__file__), 'smb_pls_dataset.csv')

class TestSMBPLS(object):
    
    def test_fit_missingdataset(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';')
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        assert test_model.latent_variables == [1, 2], 'Number of principal components is not as expected'

        block_p_loadings = array([[ 0.72187055,  0.53149537,  0.38586444,  0.18720989, -0.14682118, -0.01077914,
                                   -0.19177375,  0.73084692,  0.57978842, -0.30484587],
                                  [ 0.,          0.,          0.,          0.,          0.,          0.,
                                    0.46434217,  0.8654855,   0.21667575, -0.01441398],
                                  [ 0.,          0.,          0.,          0.,          0.,          0.,
                                   -0.31419272,  0.27049653, -0.89933897,  0.14857099]])

        assert test_model.block_p_loadings == pytest.approx(block_p_loadings), 'block p loadings are not as expected'
        
        superlevel_p_loadings = array([[ 0.71366597,  0.52545454,  0.38147881,  0.18508211, -0.14515245, -0.01065663,
                                        -0.02883137,  0.10987593,  0.08716571, -0.0458307 ],
                                       [ 0.,          0.,          0.,          0.,          0.,          0.,
                                         0.46434217,  0.8654855,   0.21667575, -0.01441398],
                                       [ 0.,          0.,          0.,          0.,          0.,          0.,
                                        -0.31419272,  0.27049653, -0.89933897,  0.14857099]])  

        assert test_model.superlevel_p_loadings == pytest.approx(superlevel_p_loadings), 'superlevel p loadings are not as expected'
        
        x_weights = array([[ 0.29286555,  0.2441086,   0.11274504,  0.05083408, -0.01934472, -0.01542236,
                             0.36462124,  0.82820732,  0.13915657, -0.02219426],
                           [ 0.,          0.,          0.,          0.,          0.,          0.,
                             0.43009356,  0.89499372,  0.11828009, -0.0039479 ],
                           [ 0.,          0.,          0.,          0.,          0.,          0.,
                            -0.31481819,  0.27124384, -0.9044671,   0.09620574]])

        assert test_model.x_weights == pytest.approx(x_weights), 'x_weights are not as expected'

        block_weights = array([[ 0.7292824,   0.60786974,  0.28075331,  0.12658505, -0.04817148, -0.03840416,
                                -0.19177375,  0.73084692,  0.57978842, -0.30484587],
                               [ 0.,          0.,          0.,          0.,          0.,          0.,
                                 0.43009356,  0.89499372,  0.11828009, -0.0039479 ],
                               [ 0.,          0.,          0.,          0.,          0.,          0.,
                                -0.31481819,  0.27124384, -0.9044671,   0.09620574]])

        assert test_model.block_weights == pytest.approx(block_weights), 'block_weights are not as expected'

        x_weights_star = array([[ 7.15396313e-01,  5.26728552e-01,  3.82403741e-01,  1.85530860e-01,
                                 -1.45504383e-01, -1.06824664e-02, -8.49794608e-02,  3.63885135e-02,
                                  3.41552710e-02, -3.92117499e-02],
                                [-6.88328818e-02, -5.06799427e-02, -3.67935242e-02, -1.78511176e-02,
                                  1.39999128e-02,  1.02782881e-03,  4.38418356e-01,  8.91300738e-01,
                                  1.15817980e-01,  5.47696385e-03],
                                [ 2.56501258e-02,  1.88855511e-02,  1.37108675e-02,  6.65210291e-03,
                                 -5.21697646e-03, -3.83013724e-04, -2.69695768e-01,  3.68129549e-01,
                                 -8.82709237e-01,  1.46939644e-01]])

        assert test_model.x_weights_star == pytest.approx(x_weights_star), 'x_weights_star are not as expected'

        superlevel_weights = array([[0.98863427, 0.15034055],
                                    [0.,         1.        ],
                                    [0.,         1.        ]])

        assert test_model.superlevel_weights == pytest.approx(superlevel_weights), 'superlevel weights is not as expected'

        c_loadings = array([[1.11716365],
                            [1.65955812],
                            [0.19872565]])

        assert test_model.c_loadings == pytest.approx(c_loadings), 'c_loadings is not as expected'

        expected_q2 = array([0.270841981393722, 0.8939788875287843, 0.9537011992581154])

        assert test_model.q2y == pytest.approx(expected_q2), 'Q2 results are not the expected'
    
    def test_fit_fulldataset(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        assert test_model.latent_variables == [2, 2], 'Number of principal components is not as expected'

        block_p_loadings = array([[ 0.72434131,  0.54728295,  0.39237194,  0.16893039, -0.16190754, -0.02024584,
                                   -0.04310647,  0.76757319,  0.59608966, -0.23162544],
                                  [-0.05372319,  0.42436286, -0.5471112,  -0.36867179,  0.62294278, -0.09768928,
                                   -0.55470883, -0.27699636,  0.20925143, -0.75616465],
                                  [ 0.,          0.,          0.,          0.,          0.,          0.,
                                    0.46655164,  0.86575441,  0.21084529, -0.01300101],
                                  [ 0.,          0.,          0.,          0.,          0.,          0.,
                                   -0.3170945,   0.26587084, -0.89498554,  0.17547119]])

        assert test_model.block_p_loadings == pytest.approx(block_p_loadings), 'block p loadings are not as expected'
        
        superlevel_p_loadings = array([[ 0.71675602,  0.54155181,  0.38826303,  0.16716135, -0.16021205, -0.02003382,
                                        -0.00622203,  0.11079223,  0.08604014, -0.03343303],
                                      [ -0.05359194,  0.42332612, -0.54577458, -0.3677711,   0.6214209,  -0.09745062,
                                        -0.03875089, -0.01935043,  0.0146179,  -0.05282421],
                                      [  0.,          0.,          0.,          0.,          0.,          0.,
                                         0.46655164,  0.86575441,  0.21084529, -0.01300101],
                                      [  0.,          0.,          0.,          0.,          0.,          0.,
                                        -0.3170945,   0.26587084, -0.89498554,  0.17547119]])

        assert test_model.superlevel_p_loadings == pytest.approx(superlevel_p_loadings), 'superlevel p loadings are not as expected'
        
        x_weights = array([[ 2.89028752e-01,  2.43097117e-01,  1.13597963e-01,  4.38808090e-02,
                            -1.78012004e-02, -1.82581150e-02,  3.73370763e-01,  8.27117856e-01,
                             1.34396517e-01, -1.71408897e-02],
                           [ 1.14588159e-03,  2.90540134e-02, -4.83776317e-02, -2.65612009e-02,
                             5.31168645e-02, -1.16404001e-02,  4.28628544e-01,  8.92442460e-01,
                             1.13832881e-01, -4.22621652e-03],
                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                             0.00000000e+00,  0.00000000e+00,  4.32168914e-01,  8.94722052e-01,
                             1.12705172e-01,  1.56026869e-04],
                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                             0.00000000e+00,  0.00000000e+00, -3.16172025e-01,  2.66376764e-01,
                            -9.02463659e-01,  1.20987665e-01]])

        assert test_model.x_weights == pytest.approx(x_weights), 'x_weights are not as expected'

        block_weights = array([[ 7.26868350e-01,  6.11356478e-01,  2.85683564e-01,  1.10354320e-01,
                                -4.47676195e-02, -4.59166979e-02, -4.31064677e-02,  7.67573194e-01,
                                 5.96089659e-01, -2.31625438e-01],
                               [ 1.38468251e-02,  3.51088493e-01, -5.84594961e-01, -3.20965364e-01,
                                 6.41863817e-01, -1.40662512e-01, -5.54708825e-01, -2.76996362e-01,
                                 2.09251432e-01, -7.56164646e-01],
                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                 0.00000000e+00,  0.00000000e+00,  4.32168914e-01,  8.94722052e-01,
                                 1.12705172e-01,  1.56026869e-04],
                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                 0.00000000e+00,  0.00000000e+00, -3.16172025e-01,  2.66376764e-01,
                                -9.02463659e-01,  1.20987665e-01]])

        assert test_model.block_weights == pytest.approx(block_weights), 'block_weights are not as expected'

        x_weights_star = array([[ 7.15972705e-01,  6.22556924e-01,  2.96930764e-01,  1.04477543e-01,
                                 -5.28212129e-02, -3.74194041e-02, -7.42995455e-02,  3.24027112e-02,
                                  2.92518072e-02, -3.41649738e-02],
                                [ 7.16271143e-02,  5.25517020e-01, -4.86389517e-01, -3.44374029e-01,
                                  6.03391924e-01, -1.02567751e-01, -4.19187192e-02,  1.93774854e-02,
                                  9.78692444e-03, -5.51939484e-02],
                                [-7.35578111e-02, -4.88075861e-02, -4.73880913e-02, -2.23405274e-02,
                                  2.53371106e-02,  6.11775123e-04,  4.38662547e-01,  8.91746733e-01,
                                  1.11027040e-01,  7.83957779e-03],
                                [ 2.98308771e-02,  3.46056564e-02,  2.71565682e-03, -2.28560350e-03,
                                  9.18728002e-03, -3.40803144e-03, -2.73024921e-01,  3.63706701e-01,
                                 -8.78599894e-01,  1.73202743e-01]])

        assert test_model.x_weights_star == pytest.approx(x_weights_star), 'x_weights_star are not as expected'

        superlevel_weights = array([[0.98952802, 0.14434094],
                                    [0.99755694, 0.06985808],
                                    [0.,         1.        ],
                                    [0.,         1.        ]])

        assert test_model.superlevel_weights == pytest.approx(superlevel_weights), 'superlevel weights is not as expected'

        c_loadings = array([[1.13247757],
                            [0.22444752],
                            [1.66790302],
                            [0.19780693]])

        assert test_model.c_loadings == pytest.approx(c_loadings), 'c_loadings is not as expected'

        expected_q2 = array([0.28885562322442215, 0.32110795878481263, 0.9385592222790314, 0.9856995899828982])

        assert test_model.q2y == pytest.approx(expected_q2), 'Q2 results are not the expected'
    
    def test_transform_missingdataset(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';')
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        scores = array([[ 0.31413744,  0.39130973, -0.11099514],
                        [-0.2680073,  -0.66810093,  0.4425192 ],
                        [-0.28107877, -1.03087832, -0.15124386],
                        [-0.28268446, -0.35992011, -0.31941046],
                        [-0.3982549,   0.19327937, -0.57926287]])

        assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function malfunctions for datasets with missing values'

    def test_transform_fulldataset(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        scores = array([[ 0.28812785, -0.02374087,  0.38154197, -0.1253679 ],
                        [-0.23870014,  0.0836608,  -0.65833823,  0.47291029],
                        [-0.22974921,  0.32751755, -1.02398531, -0.1502275 ],
                        [-0.31065476, -0.21057959, -0.36306849, -0.29460576],
                        [-0.42839992, -0.31055099,  0.18820115, -0.58135756]])

        assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'

    def test_predict(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        predictions = array([[ 0.93254611],
                             [-1.25604448],
                             [-1.92429957],
                             [-1.06291171],
                             [-0.35595099]])

        assert test_model.predict(X_test_data.iloc[:5]) == pytest.approx(predictions), 'The predict function for perfect datasets malfunctions'
  
    def test_score(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        assert test_model.score(X_test_data.iloc[:5], Y_test_data.iloc[:5]) == pytest.approx(0.9934766558278576), 'The score function for perfect datasets malfunctions'
 
    def test_Hotellings_T2(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        T2s = array([[1.81687576],
                     [5.34981048],
                     [9.13485717],
                     [2.66708715],
                     [4.84775396]])

        assert test_model.Hotellings_T2(X_test_data.iloc[:5]) == pytest.approx(T2s), 'The Hotellings_T2 function for perfect datasets malfunctions'
"""
    def test_Hotellings_T2_blocks(self): #To Do

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)  

        #assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'

    def test_SPEs_X(self): # To Do
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        #assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'

    def test_SPEs_Y(self): # To Do

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        #assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'

    def test_contributions_scores(self): # To Do

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        #assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'
    
    def test_contributions_scores_block(self): # To Do
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        #assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'

    def test_contributions_SPE_X(self): # To Do
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        #assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'

    def test_contributions_SPE_X_block(self): # To Do
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        #assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'
    """
    

    

