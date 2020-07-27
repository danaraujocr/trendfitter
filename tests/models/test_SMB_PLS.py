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

        assert test_model.latent_variables == [1, 1], 'Number of principal components is not as expected'

        block_p_loadings = array([[ 0.72187055,  0.53149537,  0.38586444,  0.18720989, -0.14682118, -0.01077914,
                                   -0.19177375,  0.73084692,  0.57978842, -0.30484587],
                                  [ 0.        ,  0.        ,  0.        ,  0.        ,  0.         , 0.        ,
                                    0.46434217,  0.8654855 ,  0.21667575, -0.01441398]])

        assert test_model.block_p_loadings == pytest.approx(block_p_loadings), 'block p loadings are not as expected'
        
        superlevel_p_loadings = array([[ 0.71366597,  0.52545454,  0.38147881,  0.18508211, -0.14515245, -0.01065663,
                                        -0.02883137,  0.10987593,  0.08716571, -0.0458307 ],
                                       [ 0.        ,  0.        ,  0.        ,  0.         , 0.         , 0.        ,
                                         0.46434217,  0.8654855 ,  0.21667575, -0.01441398]])

        assert test_model.superlevel_p_loadings == pytest.approx(superlevel_p_loadings), 'superlevel p loadings are not as expected'
        
        x_weights = array([[ 0.29286555,  0.2441086 ,  0.11274504,  0.05083408, -0.01934472, -0.01542236,
                             0.36462124,  0.82820732,  0.13915657, -0.02219426],
                           [ 0.        ,  0.        ,  0.        ,  0.       ,   0.        , 0.         ,
                             0.43009356,  0.89499372,  0.11828009, -0.0039479 ]])

        assert test_model.x_weights == pytest.approx(x_weights), 'x_weights are not as expected'

        block_weights = array([[ 0.7292824 ,  0.60786974,  0.28075331,  0.12658505, -0.04817148, -0.03840416,
                                -0.19177375,  0.73084692,  0.57978842, -0.30484587],
                               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.         ,
                                 0.43009356,  0.89499372,  0.11828009, -0.0039479 ]])

        assert test_model.block_weights == pytest.approx(block_weights), 'block_weights are not as expected'

        x_weights_star = array([[ 0.71448381,  0.5260567 ,  0.38191598,  0.18529421, -0.14531879, -0.01066884,
                                 -0.07538501,  0.02329228,  0.06555772, -0.04443914],
                                [-0.07149937, -0.05264321, -0.03821885, -0.01854264,  0.01454225,  0.001067645,
                                  0.46645486,  0.85303144,  0.20758095, -0.0097983 ]])

        assert test_model.x_weights_star == pytest.approx(x_weights_star), 'x_weights_star are not as expected'

        superlevel_weights = array([[0.98863427, 0.15034055],
                                    [0.        , 1.        ]])

        assert test_model.superlevel_weights == pytest.approx(superlevel_weights), 'superlevel weights is not as expected'

        c_loadings = array([[1.11716365],
                            [1.65955812]])

        assert test_model.c_loadings == pytest.approx(c_loadings), 'c_loadings is not as expected'

        expected_q2 = array([0.270841981393722, 0.9249127776764678])

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

        assert test_model.latent_variables == [1, 2], 'Number of principal components is not as expected'

        block_p_loadings = array([[ 0.72434131,  0.54728295,  0.39237194,  0.16893039, -0.16190754, -0.02024584,
                                   -0.04310647,  0.76757319,  0.59608966, -0.23162544],
                                  [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                    0.46522025,  0.86612955,  0.21192985, -0.01607839],
                                  [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                   -0.32421554,  0.2704348 , -0.89330915,  0.1636237 ]])

        assert test_model.block_p_loadings == pytest.approx(block_p_loadings), 'block p loadings are not as expected'
        
        superlevel_p_loadings = array([[ 0.71675602,  0.54155181,  0.38826303,  0.16716135, -0.16021205, -0.02003382,
                                        -0.00622203,  0.11079223,  0.08604014, -0.03343303],
                                       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                         0.46522025,  0.86612955,  0.21192985, -0.01607839],
                                       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                        -0.32421554,  0.2704348 , -0.89330915,  0.1636237 ]])

        assert test_model.superlevel_p_loadings == pytest.approx(superlevel_p_loadings), 'superlevel p loadings are not as expected'
        
        x_weights = array([[ 0.28902875,  0.24309712,  0.11359796,  0.04388081, -0.0178012,  -0.01825811,
                             0.37337076,  0.82711786,  0.13439652, -0.01714089],
                           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.       ,   0.        ,
                             0.4301038 ,  0.89551408,  0.11422467, -0.00424076],
                           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.       ,   0.        ,
                            -0.32350558,  0.27070105, -0.90009589,  0.10905257]])

        assert test_model.x_weights == pytest.approx(x_weights), 'x_weights are not as expected'

        block_weights = array([[ 0.72686835,  0.61135648,  0.28568356,  0.11035432, -0.04476762, -0.0459167,
                                -0.04310647,  0.76757319,  0.59608966, -0.23162544],
                               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.       ,
                                 0.4301038 ,  0.89551408,  0.11422467, -0.00424076],
                               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.       ,
                                -0.32350558,  0.27070105, -0.90009589,  0.10905257]])

        assert test_model.block_weights == pytest.approx(block_weights), 'block_weights are not as expected'

        x_weights_star = array([[ 7.03552842e-01,  5.31576028e-01,  3.81110933e-01,  1.64082113e-01,
                                 -1.57260822e-01, -1.96647870e-02, -6.69998387e-02,  2.85750481e-02,
                                  2.84020306e-02, -2.49247641e-02],
                                [-7.48473945e-02, -5.65516593e-02, -4.05444462e-02, -1.74558582e-02,
                                  1.67301759e-02,  2.09203628e-03,  4.37245350e-01,  8.92324884e-01,
                                  1.12228140e-01,  4.28092118e-03],
                                [ 2.72179697e-02,  2.05647954e-02,  1.47438334e-02,  6.34775631e-03,
                                 -6.08386470e-03, -7.60761019e-04, -2.79281481e-01,  3.67623305e-01,
                                 -8.77043981e-01,  1.62342367e-01]])

        assert test_model.x_weights_star == pytest.approx(x_weights_star), 'x_weights_star are not as expected'

        superlevel_weights = array([[0.98952802, 0.14434094],
                                    [0.        , 1.        ],
                                    [0.        , 1.        ]])

        assert test_model.superlevel_weights == pytest.approx(superlevel_weights), 'superlevel weights is not as expected'

        c_loadings = array([[1.13247757],
                            [1.66200662],
                            [0.19718822]])

        assert test_model.c_loadings == pytest.approx(c_loadings), 'c_loadings is not as expected'

        expected_q2 = array([0.28885562322442215, 0.9301799477729965, 0.9413869033563421])

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

        scores = array([[ 0.3180861 ,  0.40284835],
                        [-0.28374995, -0.71410348],
                        [-0.27569826, -1.01515561],
                        [-0.27132141, -0.32671546],
                        [-0.37764758,  0.25349725]])

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

        scores = array([[ 0.29239167,  0.38514696, -0.1151795 ],
                        [-0.2536207 , -0.66350758,  0.45991813],
                        [-0.28589818, -1.02426442, -0.14611273],
                        [-0.27384959, -0.36088289, -0.29182376],
                        [-0.37422809,  0.19158321, -0.57872799]])

        assert test_model.transform(X_test_data.iloc[:5]) == pytest.approx(scores), 'The transform function for perfect datasets malfunctions'

    def test_transform_b_fulldataset(self):#To Do
        
        a=1 

    def test_predict(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        predictions = array([[ 0.94853176],
                             [-1.29928331],
                             [-2.05491923],
                             [-0.96746247],
                             [-0.2195107 ]])

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

        assert test_model.score(X_test_data.iloc[:5], Y_test_data.iloc[:5]) == pytest.approx(0.9706446197457211), 'The score function for perfect datasets malfunctions'

    def test_Hotellings_T2(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';').dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'y')
        X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_test_data[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
        X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] = X_test_data[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
        Y_test_data = test_data['y']
        test_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
        test_model.fit(X_test_data, [6,10], Y_test_data, random_state = 2)

        T2s = array([[1.61028906],
                     [4.66663225],
                     [7.50870743],
                     [1.89242678],
                     [3.33450584]])

        assert test_model.Hotellings_T2(X_test_data.iloc[:5]) == pytest.approx(T2s), 'The Hotellings_T2 function for perfect datasets malfunctions'

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

    

    

