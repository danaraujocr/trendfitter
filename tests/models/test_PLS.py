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

        assert test_model.omega[:3,:3] == pytest.approx(expected_omega, abs = 1e-10, rel= 1e-6), 'Omega matrix is not as expected'

        expected_q2 = array([0.28124385593261386, 0.44299735550099845, 0.46414076569509394, 0.4777738478674691])
        assert test_model.q2y == pytest.approx(expected_q2), 'Q2 results are not the expected'

        expected_chi2_x = array([4.2260741947880405, 3.6569604105213234, 2.839838320613596, 2.5325678398904667])
        assert test_model._x_chi2_params == pytest.approx(expected_chi2_x), 'x_Chi2 parameters are not as expected'  

        expected_chi2_y = array([0.769305530255242, 0.7417466343075413, 0.7559688556382609, 0.7592388820343167])
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

        expected_chi2_x = array([4.070330841792376, 3.456808416735069, 2.704209315645609])
        assert test_model._x_chi2_params == pytest.approx(expected_chi2_x), 'x_Chi2 parameters are not as expected'  

        expected_chi2_y = array([0.5399925133560206, 0.6020512439574859, 0.6178419025809769])
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
            
    def test_Hotellings_T2(self):
        
        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([4.43909685, 6.11206434, 1.01557983, 2.02847542, 2.76519804])
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

        expected = array([ 3.98374384, 1.00666111, 0.594293551, 0.00154030566, 0.0000143011233])
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

        expected = array([13.92958513, 18.65835615, 14.64250661, 17.51150011, 17.53666052])
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
        
        expected = array([[-0.55119265,  1.07492243,  0.28307943,  0.81076301,  0.22325817,  1.04610695,
                           -0.34450055, -0.43390115,  0.30051563,  0.53834116, -0.033709,    0.2251455,
                           -0.26477459, -0.13587385, -0.26030856, -0.69848554, -0.19342997, -0.20191149,
                            0.53416277, -0.39588554, -0.09041671],
                          [-0.99897904,  0.37455377,  0.28650682,  0.42245137,  0.04686227,  0.58661755,
                           -3.23260752, -0.90782582,  0.05761018,  0.46141375, -0.03175858,  0.27981257,
                           -0.23861951, -0.26306489, -0.67883752, -0.88728108, -0.38951948, -0.57437813,
                            0.17242,    -0.42121949, -0.27705274],
                          [-0.17582612,  0.16991294,  0.10392468, -0.15507551,  0.0346797,   0.41274011,
                           -0.01015317, -0.05818804,  0.10519918,  0.3423248,   0.01444717,  0.1747795,
                           -0.02882394, -0.06026152, -0.08993824, -0.31734461, -0.17777734, -0.00768407,
                            0.45518559, -0.13478157, -0.04100665],
                          [-0.04402044,  0.8968054,   0.13050063,  0.08145354,  0.12524269,  0.61024802,
                           -0.38190515,  0.10027805, -0.19849843,  0.47931868,  0.10443048,  0.35010962,
                           -0.12884703, -0.1170948,  -0.16585692, -0.38430132, -0.10101148, -0.00543567,
                            0.73204152, -0.29172952, -0.14425345],
                          [-0.36928141,  1.36935517,  0.05973933,  0.36549494,  0.14890257,  0.75334434,
                           -0.24112327, -0.08047015, -0.27946722,  0.62610887,  0.1035579,   0.28081339,
                           -0.18749611, -0.09591704, -0.18907477, -0.58897251, -0.12107046,  0.03420113,
                            0.75348764, -0.32688444, -0.11421919]])

        assert expected == pytest.approx(test_model.contributions_scores_ind(X_test_data.iloc[:5]))
    
    def test_contributions_SPE_X(self):

        test_data = pd.read_csv(TESTDATA1_FILENAME, index_col = 0, delimiter = ';',).drop(columns=['ObsNum']).dropna()
        test_data = (test_data - test_data.mean()) / test_data.std()
        X_test_data = test_data.drop(columns='Y-Kappa')
        Y_test_data = test_data['Y-Kappa']
        test_model = PLS()
        test_model.fit(X_test_data, Y_test_data, random_state = 2)

        expected = array([[-3.64328547e-01, -6.38937288e-04,  9.73736182e-02,  1.51182931e-02,
                            7.53956542e-02,  1.46803910e+00,  1.52906289e-02, -4.51815515e-01,
                           -1.00337293e-03, -1.63992452e-03, -3.98404905e-01,  1.13284446e-02,
                           -2.34267664e-03, -2.57429619e+00, -1.94856653e-01, -2.25107561e+00,
                           -1.78824587e-02, -2.24010935e-01,  2.57683640e+00, -3.07836421e+00,
                            1.09543046e-01],
                          [-1.23251781e-01, -9.76468033e-04,  2.14091056e-01, -1.86344336e-01,
                            2.97791854e-02,  1.70056432e+00, -1.24775744e-01, -2.76168190e-01,
                           -7.75524936e-03, -5.19458421e-02, -8.95938127e-01, -2.72155305e-03,
                           -7.83455641e-03, -2.95613852e+00, -2.66379082e-01, -3.99433051e+00,
                           -2.08046924e-01, -3.05150957e-01,  1.65589737e+00, -5.37043633e+00,
                           -2.79830056e-01],
                          [-3.06333466e-01, -7.48769923e-02,  6.19111482e-02, -4.36039087e-01,
                            1.33768255e-01,  1.71436212e+00, -1.24175627e-01, -4.25762021e-01,
                            6.54466481e-01,  1.49188545e-02, -1.90108226e-01,  7.85666212e-02,
                            2.95837155e-02, -2.81089627e+00, -4.08420726e-01, -1.42222331e+00,
                           -2.31367284e-01,  1.44681488e-02,  2.54863534e+00, -2.92933396e+00,
                            3.22889433e-02],
                          [-8.50045281e-03,  2.17047591e-01, -2.01780208e-02,  5.70738385e-03,
                            5.70880934e-02,  7.77003976e-01, -5.89509926e-01, -5.96395210e-02,
                           -8.99336190e-01,  1.45831824e-07, -7.23218607e-02,  2.01905939e-01,
                            5.11151596e-04, -4.16661047e+00, -1.12036857e+00, -1.31886241e+00,
                           -1.05110751e-02,  2.11386812e-01,  3.31487824e+00, -4.46013225e+00,
                           -2.84986307e-08],
                          [-2.99542639e-01,  2.45587573e-01,  5.22057268e-07,  1.24916710e-01,
                            1.02785830e-01,  1.05673720e+00, -2.39052144e-01, -1.46022300e-01,
                           -1.32773090e+00,  2.17513673e-04, -2.89518373e-02,  5.82666256e-02,
                            6.60330288e-03, -4.04158765e+00, -4.28687880e-01, -1.62725837e+00,
                            3.00489597e-03,  5.68344733e-01,  3.32397088e+00, -3.85501976e+00,
                            5.23712528e-02,]])

        assert expected == pytest.approx(test_model.contributions_SPE_X(X_test_data.iloc[:5]))
        