import pandas as pd
from numpy import array, max
import pytest
import sys
import os
sys.path.append(".")
from trendfitter.models import PLS

TESTDATA1_FILENAME = os.path.join(os.path.dirname(__file__), 'pls_dataset_complete.csv')

class TestPLS(object):

    def test_fit_missingdataset(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        assert test_model.latent_variables == 4, 'Number of principal components is not as expected'

        first_3_comps_p = array([[-0.38391965, -0.09236524, -0.26024297,  0.23208795, -0.19929187, -0.33394883, -0.34659546, -0.48359965,  
                                   0.10296577, -0.14243939, -0.18986493, -0.29474327, -0.03496983, -0.05700808, -0.42094606, -0.0704986,  
                                  -0.03821775,  0.04818109, -0.30362093,  0.01200999,  0.13949998],
                                 [ 0.02867525, -0.04920307,  0.38022892,  0.2348004,   0.24956067,  0.30055631, -0.48663128,  0.14915648, 
                                   0.00389374,  0.23637791,  0.37235114,  0.38134085, -0.10644904, -0.03923513,  0.27707641, -0.059204,    
                                   0.10436763, -0.30675226,  0.05878066,  0.22503696, -0.19624219],
                                 [ 0.35697262, -0.34331472,  0.15040837,  0.01330634,  0.03787924, -0.10852945,  0.03868765,  0.02556225,
                                  -0.00308889, -0.46519269, -0.33776066, -0.37454578,  0.42426537,  0.00158257,  0.12057003,  0.35663339,
                                   0.22890949,  0.16533393,  0.11915887, -0.31812768,  0.21223439]])

        assert test_model.p_loadings[:3, :] == pytest.approx(first_3_comps_p), 'p loadings are not as expected'

        first_3_comps_w = array([[-0.30758559, -0.21638924, -0.03264433,  0.294381,   -0.0990744,  -0.24625292,
                                  -0.56272695, -0.42592989,  0.07314031, -0.13499403, -0.06227673, -0.17043098,
                                  -0.06246633, -0.06938699, -0.2594048,  -0.04526347,  0.05427155, -0.05821118,
                                  -0.21573245,  0.07678774,  0.09477246],
                                 [ 0.15621484, -0.25381056,  0.46577226,  0.12748044,  0.20509137,  0.17946645,
                                  -0.44230514,  0.11801905, -0.06103671,  0.01523664,  0.26110454,  0.25440053,
                                  -0.05627059, -0.02533297,  0.33058824,  0.05164276,  0.18927595, -0.21772788,
                                   0.17986054,  0.13256528, -0.09153322],
                                 [ 0.25750013, -0.4130988,   0.17271044, -0.21667704, -0.08978272, -0.24447823,
                                   0.08949367, -0.06286591, -0.13109339, -0.44648022, -0.22460487, -0.2562902,
                                   0.10130938,  0.02806822,  0.10803941,  0.22379759,  0.17142834,  0.17973861,
                                   0.2444581,  -0.18669864,  0.21140552]])

        assert test_model.weights[:3, :] == pytest.approx(first_3_comps_w), 'Weights are not as expected'

        first_3_comps_w_star = array([[-0.30758559, -0.21638924, -0.03264433,  0.294381,   -0.0990744,  -0.24625292,
                                       -0.56272695, -0.42592989,  0.07314031, -0.13499403, -0.06227673, -0.17043098,
                                       -0.06246633, -0.06938699, -0.2594048,  -0.04526347,  0.05427155, -0.05821118,
                                       -0.21573245,  0.07678774,  0.09477246],
                                      [ 0.00591379, -0.35954871,  0.44982067,  0.27132911,  0.15667887,  0.05913547,
                                       -0.7172805,  -0.09011071, -0.02529685, -0.05072791,  0.23067314,  0.17111979,
                                       -0.08679464, -0.05923878,  0.20383062,  0.02952485,  0.21579562, -0.24617265,
                                        0.07444333,  0.17008745, -0.04522285],
                                      [ 0.26042922, -0.59118297,  0.39550623, -0.08228797, -0.0121798,  -0.21518848,
                                       -0.26577474, -0.10749767, -0.1436229,  -0.47160571, -0.11035265, -0.17153472,
                                        0.05832007, -0.001272693,  0.20899654,  0.23842122,  0.27831173,  0.05780951,
                                        0.28132982, -0.10245447,  0.18900668]])

        assert test_model.weights_star[:3, :] == pytest.approx(first_3_comps_w_star), 'Weights_star are not as expected'

        first_3_comps_c = array([[0.29056385],
                                 [0.34640027],
                                 [0.09379042]])

        assert test_model.c_loadings[:3, :] == pytest.approx(first_3_comps_c), 'c loadings are not as expected'

        expected_VIPs = array([0.7070244,  0.60158889, 0.79225345, 0.67058561, 0.23224286, 0.53446447,
                               2.63378319, 1.16720839, 0.07320984, 0.2074043,  0.30493998, 0.45483306,
                               0.16182822, 0.03284593, 0.80461928, 0.06646472, 0.16314725, 0.20477494,
                               0.44569794, 0.13521753, 0.10586574])
        assert test_model.feature_importances_ == pytest.approx(expected_VIPs), 'VIPs are not as expected'

        expected_omega = array([[ 6.53568591e+03, -1.20792265e-12,  1.70530257e-13],
                                [-1.20792265e-12,  2.67886350e+03, -1.35003120e-13],
                                [ 1.70530257e-13, -1.35003120e-13,  4.90047180e+03]])

        assert test_model.omega[:3,:3] == pytest.approx(expected_omega), 'Omega matrix is not as expected'

        expected_q2 = array([0.28124385593261386, 0.44299735550099845, 0.46414076569509394, 0.4777738478674691])
        assert test_model.q2y == pytest.approx(expected_q2), 'Q2 results are not the expected'

        expected_chi2_x = array([4.2260741947880405, 3.6569604105213234, 2.839838320613596, 2.5325678398904667, 
                                 2.339460442574692, 2.377990269594017, 1.8635322589489425, 2.1352872839523274, 
                                 2.126912496418633])
        assert test_model._x_chi2_params == pytest.approx(expected_chi2_x), 'x_Chi2 parameters are not as expected'  

        expected_chi2_y = array([0.769305530255242, 0.7417466343075413, 0.7559688556382609, 0.7592388820343167, 
                                 0.765427926544429, 0.7376254759409571, 0.7462417136569457, 0.7384581739778784, 
                                 0.7374879297830842])
        assert test_model._y_chi2_params == pytest.approx(expected_chi2_y), 'y_Chi2 parameters are not as expected'   
    
    def test_fit_fulldataset(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum'])
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        assert test_model.latent_variables == 3, 'Number of principal components is not as expected'

        first_3_comps_p = array([[-0.39106638, -0.09236706, -0.21387359,  0.22967692, -0.20702947, -0.3253848,
                                  -0.36559824, -0.47753691,  0.0987895,  -0.11846458, -0.16549252, -0.27461443,
                                  -0.05582826, -0.05015748, -0.4218769,  -0.08286477, -0.05102729,  0.04780678,
                                  -0.29685175,  0.03212617,  0.12879999],
                                 [ 0.07097837, -0.15222275,  0.31716597,  0.21546817,  0.24333866,  0.29032132,
                                  -0.46533026,  0.18816008, -0.00721616,  0.21747614,  0.38106322,  0.37921151,
                                  -0.10040177, -0.02221209,  0.31366937, -0.08337564,  0.14144098, -0.27578668,
                                   0.06784578,  0.20944708, -0.1909246, ],
                                 [ 0.31835697, -0.30350748,  0.13855747,  0.04005883,  0.02266223, -0.10458581,
                                   0.04580604, -0.00901923,  0.02508521, -0.4545722,  -0.34808134, -0.37021413,
                                   0.42139396, -0.00408376,  0.08162373,  0.34025094,  0.22393838,  0.19347418,
                                   0.10550796, -0.2909857,   0.19503255]])

        assert test_model.p_loadings[:3, :] == pytest.approx(first_3_comps_p), 'p loadings are not as expected'

        first_3_comps_w = array([[-0.2966795,  -0.23471954, -0.03260299,  0.30348731, -0.11325343, -0.23075466,
                                  -0.58210269, -0.41134708,  0.07810032, -0.11967569, -0.04806623, -0.15045568,
                                  -0.07334184, -0.06082079, -0.25160195, -0.06713384,  0.06545168, -0.04097898,
                                  -0.20930659,  0.0983964,   0.08879355],
                                 [ 0.1934994,  -0.29751353,  0.37018063,  0.15608748,  0.19447119,  0.19549362,
                                  -0.45805303,  0.13345899, -0.05391289, -0.0035831,   0.24359214,  0.25638508,
                                  -0.03705075, -0.02270366,  0.35160199,  0.03199279,  0.2428773,  -0.18476174,
                                   0.18004097,  0.1385851,  -0.09855732],
                                 [ 0.27357357, -0.32964806,  0.12651661, -0.12831027, -0.10387174, -0.20505985,
                                   0.0540954,  -0.11838545, -0.10314372, -0.48767909, -0.29906534, -0.26656296,
                                   0.13885521, -0.001512683,  0.08969401,  0.25505496,  0.22803857,  0.19775747,
                                   0.2505644,  -0.15394687,  0.18788584]])

        assert test_model.weights[:3, :] == pytest.approx(first_3_comps_w), 'Weights are not as expected'

        first_3_comps_w_star = array([[-0.2966795,  -0.23471954, -0.03260299,  0.30348731, -0.11325343, -0.23075466,
                                       -0.58210269, -0.41134708,  0.07810032, -0.11967569, -0.04806623, -0.15045568,
                                       -0.07334184, -0.06082079, -0.25160195, -0.06713384,  0.06545168, -0.04097898,
                                       -0.20930659,  0.0983964,   0.08879355],
                                      [ 0.06356431, -0.40031236,  0.35590168,  0.28900416,  0.14487021,  0.09443126,
                                       -0.71299335, -0.04669643, -0.01970772, -0.05599681,  0.22254084,  0.19049083,
                                       -0.06917187, -0.049341,    0.24140926,  0.002590553,  0.27154282, -0.20270908,
                                        0.08837211,  0.18167923, -0.0596689 ],
                                      [ 0.31358406, -0.50120572,  0.2882677,  -0.00957986, -0.03420769, -0.15364979,
                                       -0.24519197, -0.12371571, -0.11501786, -0.50835641, -0.19686601, -0.17490241,
                                        0.11046451, -0.02143884,  0.20818532,  0.25879028,  0.34799615,  0.10790673,
                                        0.29842179, -0.07577566,  0.15758113]])

        assert test_model.weights_star[:3, :] == pytest.approx(first_3_comps_w_star), 'Weights_star are not as expected'

        first_3_comps_c = array([[0.27990557],
                                 [0.32582029],
                                 [0.06882031]])

        assert test_model.c_loadings[:3, :] == pytest.approx(first_3_comps_c), 'c loadings are not as expected'

        expected_VIPs = array([0.73793148, 0.71789397, 0.5044633,  0.69651041, 0.22360899, 0.5010976,
                               2.97262732, 1.17697529, 0.05424181, 0.18039394, 0.26058425, 0.40994093,
                               0.04716521, 0.02607252, 0.86100397, 0.05686198, 0.25863264, 0.14771365,
                               0.42601949, 0.14093365, 0.09932758])
        assert test_model.feature_importances_ == pytest.approx(expected_VIPs), 'VIPs are not as expected'

        expected_omega = array([[15119.1667892,     63.75260406,    30.61279584],
                                [63.75260406,  6201.40217004,    29.70987033],
                                [30.61279584,    29.70987033, 13205.0743981 ]])

        assert test_model.omega[:3,:3] == pytest.approx(expected_omega), 'Omega matrix is not as expected'

        expected_q2 = array([0.2668460462257186, 0.4149487741167585, 0.4299748436733221])
        assert test_model.q2y == pytest.approx(expected_q2), 'Q2 results are not the expected'

        expected_chi2_x = array([4.070330841792376, 3.456808416735069, 2.704209315645609, 2.580675804525607, 2.3438153267219723, 2.1124115762540243, 1.8197774351953078])
        assert test_model._x_chi2_params == pytest.approx(expected_chi2_x), 'x_Chi2 parameters are not as expected'  

        expected_chi2_y = array([0.5399925133560206, 0.6020512439574859, 0.6178419025809769, 0.6412706478079183, 0.6174785762892275, 0.6157691570215392, 0.6252037067311925])
        assert test_model._y_chi2_params == pytest.approx(expected_chi2_y), 'y_Chi2 parameters are not as expected'
    
    def test_predict_fulldata(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([[ 0.11609397],
                          [ 1.18838183],
                          [-0.4517026 ],
                          [-0.36745363],
                          [-0.29454481]])

        assert expected == pytest.approx(test_model.predict(X_test_data.iloc[:5])), 'Prediction of X values with missing data unsuccessful'
    
    def test_predict_missingdata(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum'])
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([[ 0.25948147],
                          [ 0.68925751],
                          [ 1.08415729],
                          [-0.42101315],
                          [-0.34141955]])

        assert expected == pytest.approx(test_model.predict(X_test_data.iloc[:5])), 'Prediction of X values with missing data unsuccessful'
        
    def test_transform_fulldata(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns = 'Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)
        
        expected = array([[ 0.81657827,  0.77136905, -2.13631715, -1.45398157],
                          [ 2.47743698,  1.96468243, -1.95575112, -0.22123731],
                          [-0.9682347,  -0.06753595, -1.34948841, -0.15780255],
                          [-1.1467817,   0.57333892, -1.62808662, -0.61981112],
                          [-0.29097603,  0.32964276, -2.12061936, -0.9689491 ]])

        assert expected == pytest.approx(test_model.transform(X_test_data.iloc[:5]))

    def test_transform_missingdata(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum'])
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([[ 0.79860523,  0.56234314, -2.13999888],
                          [ 2.04400782,  0.84832038, -2.31431176],
                          [ 2.48474947,  1.6599049,  -2.21110043],
                          [-0.96870861, -0.14662799, -1.48345501],
                          [-1.2399239,   0.30345428, -1.35468006]])

        assert expected == pytest.approx(test_model.transform(X_test_data.iloc[:5]))

    def tranform_inv_fulldata(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = [[-7.58647916e-01,  1.47267698e+00, -9.05615714e-01,  1.28746214e+00,
                      4.06054629e-01,  8.31715822e-01, -3.59119796e-01,  1.28885273e-01,
                      7.95459835e-01,  1.19530022e+00,  1.60434905e-01,  1.93715894e-01,
                      5.44906477e-01, -1.67084981e-01, -6.39536902e-01, -3.74945744e-01,
                     -4.62674862e-01, -4.45433692e-01, -1.27375726e+00,  1.56675018e-01,
                     -5.13648217e-01],
                    [-1.25299661e+00,  1.47154224e-02, -2.86056085e-02,  1.52404870e+00,
                      1.37940693e-01,  7.22177291e-03, -2.27048923e+00, -9.73848589e-01,
                      4.80622945e-01,  5.39542702e-01,  4.83938969e-01,  2.24251255e-01,
                     -3.19520821e-01, -2.96061210e-01, -6.72009607e-01, -4.97814491e-01,
                      2.11219019e-03, -7.53785687e-01, -1.02271488e+00,  6.85811416e-01,
                     -2.46891186e-01],
                    [-4.92207366e-02,  1.10363203e+00, -5.73854298e-01, -1.20461170e-01,
                      1.38691591e-01,  5.82078712e-01,  9.61276716e-01,  5.97908247e-01,
                      1.73376974e-01,  8.07151153e-01,  1.71673340e-01,  3.66975795e-01,
                      7.31052192e-02,  6.55807906e-02, -5.96666392e-02, -2.63788962e-01,
                     -4.51261807e-01, -1.41976844e-02, -2.35926548e-01,  2.46302810e-02,
                     -3.60307676e-01]]

        assert expected == pytest.approx(test_model.tranform_inv(test_model.transform(X_test_data.iloc[:3])))

    def tranform_inv_missdata(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([[-0.87225439,  0.66001851, -0.4427906,   0.42538695,  0.0642264,   0.19763127,
                           -0.34110664, -0.09001187,  0.29742693,  0.9608191,   0.50805141,  0.36125749,
                           -0.33386343, -0.03043928, -0.51069214, -0.60596756, -0.53974108, -0.37763837,
                           -0.75608018,  0.34290577, -0.29986684],
                          [-1.2782236,   0.34058459, -0.43186344,  0.88766934, -0.02942729, -0.03596285,
                           -1.22721859, -0.59409347,  0.40910641,  0.88437395,  0.54614751,  0.25884338,
                           -0.46424052, -0.11655899, -0.79128968, -0.73394585, -0.44123254, -0.50545341,
                           -1.04349749,  0.53061369, -0.23381549],
                          [-1.32502938, -0.13948352, -0.12763615,  1.2549914,   0.034701,   -0.0768855,
                           -2.08773646, -0.82605823,  0.41566285,  0.7337136,   0.68525462,  0.3290788,
                           -0.54130295, -0.18562237, -0.68477004, -0.73472199, -0.15608816, -0.67689291,
                           -1.03322568,  0.71360823, -0.22684267]])
        assert expected == pytest.approx(test_model.tranform_inv(test_model.transform(X_test_data.iloc[:3])))

    def test_score(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)
        
        expected = 0.4951827330153371
        
        assert expected == pytest.approx(test_model.score(X_test_data, Y_test_data))
            
    def test_Hot_T2(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([5.02767792, 8.78391192, 1.76548568, 2.96267035, 3.56459586])
        expected_95_lim = 9.526113035898028

        assert expected == pytest.approx(test_model.Hotellings_T2(X_test_data.iloc[:5]))
        assert expected_95_lim == pytest.approx(test_model.T2_limit(0.95))

    def test_SPEs_Y(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([ 8.70231307,  7.12559627, 14.19338576, 13.17493506, 12.35086177])
        expected_95_lim = 5.9332452766668045

        assert expected == pytest.approx(test_model.SPEs_Y(X_test_data.iloc[:5], Y_test_data.iloc[:5]))
        assert expected_95_lim == pytest.approx(test_model.SPE_Y_limit(0.95))

    def test_SPEs_X(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([17.72518645, 20.89734763, 16.81212398, 20.49478103, 21.33611356])
        expected_95_lim = 19.791328565268596

        assert expected == pytest.approx(test_model.SPEs_X(X_test_data.iloc[:5]))
        assert expected_95_lim == pytest.approx(test_model.SPE_X_limit(0.95))
    
    def test_rmsee(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = [0.71124698]
        assert expected == pytest.approx(test_model.RMSEE(X_test_data, Y_test_data))
    
    def test_contributions_scores_ind(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)
        
        expected = array([[ 1.11252687e+00, -4.62942516e+00, -7.33401238e-01, -7.50251539e-01,
                           -9.48160094e-02, -1.60347239e+00,  1.74671715e-01,  2.55057138e-01,
                           -3.31612562e-01, -2.04254373e+00,  1.80111519e-02, -2.50946841e-01,
                            3.21165469e-01,  4.49492145e-03,  3.50094616e-01,  1.35828629e+00,
                            4.09259624e-01, -1.76923634e-03, -1.07336033e+00,  2.46038381e-01,
                            1.33630995e-01],
                          [ 1.23996571e+00, -1.31505444e+00, -4.01094173e-01,  3.06478259e-02,
                            4.84448214e-03, -6.64498616e-01, -1.23131262e+00,  2.60542349e-03,
                           -6.83652533e-02, -1.76463062e+00,  1.75517908e-03, -1.55125238e-01,
                            1.72197070e-01, -1.86372701e-02,  4.07337723e-01,  1.75430364e+00,
                            5.57176467e-01, -1.25660257e-01, -3.11599550e-01,  7.73681261e-02,
                            3.72162416e-01],
                          [ 3.33463401e-01, -8.62497125e-01, -3.60460957e-01,  1.24413993e-01,
                           -1.28198890e-02, -7.11969023e-01,  1.72885473e-02,  3.56442990e-02,
                           -1.41772328e-01, -1.37854076e+00, -1.41289772e-02, -2.32049306e-01,
                            3.51592865e-02,  3.69843054e-03,  1.37037237e-01,  6.49447995e-01,
                            4.24618165e-01,  3.92665819e-03, -1.07272258e+00,  1.21942316e-01,
                            6.47426681e-02],
                          [ 8.39320238e-02, -4.19522932e+00, -3.58979056e-01, -7.03722785e-02,
                           -4.48532888e-02, -1.04723650e+00,  2.76987804e-01, -6.82584646e-02,
                            2.52113823e-01, -1.90538590e+00, -5.72993269e-02, -4.02679289e-01,
                            1.57714762e-01,  4.48192301e-03,  2.22629770e-01,  7.77241215e-01,
                            2.15857339e-01, -8.56712416e-05, -1.66610048e+00,  1.92721602e-01,
                            2.23230991e-01],
                          [ 8.24758206e-01, -6.63191901e+00, -1.86278168e-01, -3.92126940e-01,
                           -7.01806151e-02, -1.34739677e+00,  3.78024256e-01,  7.44531385e-02,
                            3.45469694e-01, -2.49466021e+00, -8.00003540e-02, -3.85557145e-01,
                            2.31945117e-01,  7.21203047e-03,  3.19638970e-01,  1.18995117e+00,
                            2.79500695e-01, -7.01913099e-03, -1.74014059e+00,  2.57566472e-01,
                            1.81400432e-01]])

        assert expected == pytest.approx(test_model.contributions_scores_ind(X_test_data.iloc[:5]))
    
    def test_contributions_SPE_X(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([[-4.98353719e-01, -2.05472738e-01,  1.69990850e+00, -1.40950991e-02,
                            5.23150683e-02,  8.17079276e-01, -3.09394493e-02, -1.50798153e+00,
                           -2.90893382e-02, -1.95750690e-01, -7.63820808e-02,  2.79742817e-01,
                           -7.83554532e-01, -2.18884919e+00, -8.08031872e-03, -2.59895218e+00,
                           -1.89345614e-04, -3.28059324e-01,  4.29379016e+00, -2.07242721e+00,
                            4.41738920e-02],
                          [-4.37791690e-01,  1.32922546e-01,  1.21901639e-01, -7.02917528e-01,
                            3.88450246e-04,  1.75012730e+00,  1.55071654e-03, -2.71209398e-01,
                           -4.59048736e-02,  4.55055526e-02, -3.13862511e-01,  1.92700329e-01,
                           -6.14945492e-01, -2.68598655e+00, -3.17264700e-01, -5.91291624e+00,
                           -6.09640521e-01, -3.65332009e-01,  1.92254249e+00, -3.87589435e+00,
                           -5.76042735e-01],
                          [-3.54322550e-01, -6.00828942e-01,  7.76955840e-01, -5.21343811e-01,
                            1.57359702e-01,  1.47113166e+00, -9.77092748e-01, -7.00204392e-01,
                            3.61100686e-01,  1.28128485e-03, -8.95999470e-04,  4.25918296e-01,
                           -1.25243213e-01, -2.83015838e+00, -1.17271732e-01, -1.68150207e+00,
                           -8.89059752e-02, -1.29875703e-02,  3.70031091e+00, -1.89380751e+00,
                            1.35007029e-02],
                          [-7.29511821e-03, -1.52942809e-05,  6.08045828e-01, -1.40858822e-04,
                            1.26731605e-01,  7.46353023e-01, -1.90607203e+00, -1.86535085e-01,
                           -1.31286592e+00, -8.87906895e-03,  7.33555910e-02,  1.02476816e+00,
                           -4.12859628e-01, -4.01445452e+00, -3.43148295e-01, -1.59982208e+00,
                            9.03859253e-03,  2.76797538e-02,  5.12391070e+00, -2.93318623e+00,
                           -2.96236532e-02],
                          [-3.88430668e-01, -1.51847793e-02,  1.07200255e+00,  3.40451766e-02,
                            1.03505175e-01,  6.04202540e-01, -1.26289540e+00, -7.07673907e-01,
                           -1.88529559e+00, -9.26016869e-02,  9.31628943e-02,  5.32123462e-01,
                           -5.63653266e-01, -3.77562005e+00, -5.63786563e-02, -1.94623436e+00,
                            7.12808294e-02,  2.86202058e-01,  5.39927168e+00, -2.43130504e+00,
                            1.50437897e-02]])

        assert expected == pytest.approx(test_model.contributions_SPE_X(X_test_data.iloc[:5]))
        