import pandas as pd
from numpy import array, max
import pytest
import sys
import os
sys.path.append(".")
from trendfitter.models import PCA

TESTDATA1_FILENAME = os.path.join(os.path.dirname(__file__), 'pca_test_dataset_complete.csv')
TESTDATA2_FILENAME = os.path.join(os.path.dirname(__file__), 'pca_test_dataset_incomplete.csv')

class TestPCA(object):

    def test_fit_fulldataset(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        assert test_model.principal_components == 2, 'Number of principal components is not as expected'

        first_2_comps = array([[-0.45753346,  0.47874552, -0.53238765,  0.50447686, -0.15340256], 
                               [-0.37043881,  0.35674995,  0.19766106, -0.22123996,  0.80466611]])
        assert test_model.loadings[:2, :] == pytest.approx(first_2_comps), 'Loadings are not as expected'

        expected_VIPs = array([[0.52334216, 0.57299319, 0.70859153, 0.63624226, 0.05883086]])
        assert test_model.feature_importances_ == pytest.approx(expected_VIPs), 'VIPs are not as expected'

        expected_omega = array([[ 1.48529445e+02, -6.14879643e-06,  2.48236601e-14,  1.01018217e-13,   4.74717518e-15],
                                [-6.14879643e-06,  6.34895824e+01, -3.23822093e-07, -9.99561992e-07,  -2.73430106e-10],
                                [ 2.48236601e-14, -3.23822093e-07,  1.51924177e+01, -3.29969945e-06,   8.00928824e-13],
                                [ 1.01018217e-13, -9.99561992e-07, -3.29969945e-06,  1.18540840e+01,  -1.43787014e-06],
                                [ 4.74717518e-15, -2.73430106e-10,  8.00928824e-13, -1.43787014e-06,   5.93447064e+00]])
        assert test_model.omega == pytest.approx(expected_omega), 'Omega matrix is not as expected'

        expected_q2 = array([0.5567342018714448, 0.845616530550494])
        assert test_model.q2 == pytest.approx(expected_q2), 'Q2 results are not the expected'

        expected_chi2 = array([2.3793202060672445, 2.243469844496008, 1.132808505870134, 0.36465682712001724])
        assert test_model._chi2_params == pytest.approx(expected_chi2), 'Chi2 parameters are not as expected'   
    
    def test_fit_missingdataset(self):

        test_data = pd.read_csv(TESTDATA2_FILENAME, index_col = 0, delimiter = ';')
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        assert test_model.principal_components == 2, 'Number of principal components is not as expected'

        first_2_comps = array([[-0.4632267,   0.48424391, -0.53101657,  0.50288033, -0.12673451],
                               [-0.36939071,  0.32991466,  0.20386669, -0.25984463,  0.80350853]])
        assert test_model.loadings[:2, :] == pytest.approx(first_2_comps), 'Loadings are not as expected'

        expected_VIPs = array([[0.53644745, 0.5862304,  0.7049465,  0.63222156, 0.04015409]])
        assert test_model.feature_importances_ == pytest.approx(expected_VIPs), 'VIPs are not as expected'

        expected_omega = array([[ 1.47869724e+02, -4.00782741e+00, -1.61573682e+00,  1.11976354e+00,   8.86326026e-01],
                       [-4.00782741e+00,  6.45928321e+01, -3.80556246e-01,  2.28567059e-01,   5.84927728e-02],
                       [-1.61573682e+00, -3.80556246e-01,  1.38083114e+01,  2.07760322e-02,   4.33467397e-02],
                       [ 1.11976354e+00,  2.28567059e-01,  2.07760322e-02,  1.17336462e+01,  -1.19156376e-03],
                       [ 8.86326026e-01,  5.84927728e-02,  4.33467397e-02, -1.19156376e-03,   6.45202485e+00]])
        assert test_model.omega == pytest.approx(expected_omega), 'Omega matrix is not as expected'

        expected_q2 = array([0.5572977189894531, 0.8418633512116621])
        assert test_model.q2 == pytest.approx(expected_q2), 'Q2 results are not the expected'

        expected_chi2 = array([2.3031890323200703, 1.960885543811247, 1.0517067450883657, 0.4141006788413027])
        assert test_model._chi2_params == pytest.approx(expected_chi2), 'Chi2 parameters are not as expected'   

    
    def test_predict_fulldata(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        expected = array([[-0.40341328,  0.44123301, -0.858837,    0.8348354,  -0.71060328],
                          [ 1.14753328, -1.211652,    1.55766262, -1.488007,    0.71331028],
                          [-0.21042758,  0.19359614,  0.29674882, -0.31043066,  0.72962799],
                          [-0.46436954,  0.52113604, -1.25815783,  1.2309502,  -1.21621669],
                          [-0.82589664,  0.84371014, -0.54389296,  0.49286008,  0.3393626 ]])

        assert expected == pytest.approx(test_model.predict(test_data.iloc[:5])), 'Prediction of X values unsuccessful'
    
    
    def test_predict_missingdata(self):
        
        test_data = pd.read_csv(TESTDATA2_FILENAME, index_col = 0, delimiter = ';')
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        expected = array([[-0.6253456,   0.6455792,  -0.6260675,   0.5832297,  -0.04016798],
                          [ 1.1948117,  -1.26079601,  1.50100527, -1.43545535,  0.5162801 ],
                          [-0.11255508,  0.08932129,  0.1871146,  -0.21085459,  0.42507269],
                          [-0.68438244, 0.73650655, -1.01960558,  0.99060505, -0.52620305],
                          [-0.70410329,  0.72388327, -0.67142978,  0.62140642,  0.003059486]])

        assert expected == pytest.approx(test_model.predict(test_data.iloc[:5])), 'Prediction of X values with missing data unsuccessful'
    
    
    def test_transform_fulldata(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)
        
        expected = array([[ 1.38321112, -0.61940618],
                          [-2.79447691,  0.35372482],
                          [-0.23755635,  0.86145823],
                          [ 1.93934048, -1.14173678],
                          [ 1.26793711,  0.66346449]])

        assert expected == pytest.approx(test_model.transform(test_data.iloc[:5]))

    def test_transform_missingdata(self):

        test_data = pd.read_csv(TESTDATA2_FILENAME, index_col = 0, delimiter = ';')
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        expected = array([[ 1.23456343,  0.14473251],
                          [-2.74628248,  0.20937094],
                          [-0.1588916,   0.50395936],
                          [ 1.77623882, -0.37472197],
                          [ 1.34748053,  0.21634091]])

        assert expected == pytest.approx(test_model.transform(test_data.iloc[:5]))

    
    def test_score(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)
        
        expected = 0.8527862417379573
        
        assert expected == pytest.approx(test_model.score(test_data.iloc[:5]))
        
    
    def test_Hot_T2(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        expected = array([-0.56343754, -0.36184065,  1.35864715, -1.24669758,  1.54117051])
        expected_95_lim = 6.627785016879078

        assert expected == pytest.approx(test_model.Hotellings_T2(test_data.iloc[:5]))
        assert expected_95_lim == pytest.approx(test_model.T2_limit(0.95))

    def test_SPEs(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        expected = array([0.39903279, 1.46216746, 0.40595311, 0.5964136,  0.35905571])
        expected_95_lim = 8.618197023100464

        assert expected == pytest.approx(test_model.SPEs(test_data.iloc[:5]))
        assert expected_95_lim == pytest.approx(test_model.SPE_limit(0.95))
    
    def test_contributions_scores_ind(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)
        
        expected = array([[-0.04951269,  0.09041662, -0.08988002,  0.03864856, -0.21561601],
                          [ 0.09168162, -0.50549444,  0.53284368, -0.74637347,  0.03442284],
                          [-0.05482525,  0.00808005,  0.00728682, -0.02316333,  0.19126836],
                          [-0.08946416,  0.1431511,  -0.19203636,  0.40167076, -0.76786504],
                          [-0.06111031,  0.10338838, -0.02696328,  0.08253194,  0.11645877]])

        assert expected == pytest.approx(test_model.contributions_scores_ind(test_data.iloc[:5]))
    
    def test_contributions_spe(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col=0)
        test_data = (test_data - test_data.mean()) / test_data.std()
        test_model = PCA()
        test_model.fit(test_data)

        expected = array([[-1.40922213e-03,  1.16346779e-01,  7.69758438e-06, -1.96541047e-01,  -8.47280388e-02],
                          [-6.96758397e-01, -1.40997887e-01, -2.58981400e-02, -4.64770291e-01,  -1.33742743e-01],
                          [-1.75533104e-01, -8.83557269e-03, -6.97661053e-04, -1.56612937e-01,  -6.42738372e-02],
                          [ 2.22141917e-02, -3.97250224e-04,  1.61680615e-01,  3.89537193e-01,   2.25843449e-02],
                          [ 6.72448679e-02,  9.85292592e-03,  6.30157900e-02,  2.00243599e-01,   1.86985234e-02]])

        assert expected == pytest.approx(test_model.contributions_spe(test_data.iloc[:5]))