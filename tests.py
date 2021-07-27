from trendfitter.models.DiPLS import DiPLS
from trendfitter.models import PCA, PLS, SMB_PLS, MB_PCA, MB_PLS, MLSMB_PLS
import pandas as pd
from numpy import sqrt, mean
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

"""
#pca_data = pd.read_csv('pca_test_dataset_incomplete.csv', index_col=0, delimiter = ';')
pca_data = pd.read_csv('pca_test_dataset_complete.csv', index_col=0)
pca_data = (pca_data - pca_data.mean()) / pca_data.std()
#print(pca_data.head())


pca_model = PCA()
pca_model.fit(pca_data)
print(f'PCA Model fitted with : {pca_model.principal_components} principal components')
print(pca_model.loadings)
print('VIPs')
print(pca_model.feature_importances_)
print('OMEGA')
print(pca_model.omega)
print('q2')
print(pca_model.q2)
print('chi2_params')
print(pca_model._chi2_params)
print('predict 5 first lines')
print(pca_model.predict(pca_data.iloc[:5]))
print('scores')
print(pca_model.transform(pca_data.iloc[:5]))
print('r2')
print(pca_model.score(pca_data))
print('Hot T2')
print(pca_model.Hotellings_T2(pca_data.iloc[:5]))
print('T2 Limit')
print(pca_model.T2_limit(0.95))
print('SPEs')
print(pca_model.SPEs(pca_data.iloc[:5]))
print('SPE limit')
print(pca_model.SPE_limit(0.99))
print('contributions scores ind')
print(pca_model.contributions_scores_ind(pca_data.iloc[:5]))
print('contributions spe')
print(pca_model.contributions_spe(pca_data.iloc[:5]))


mbpca_model = MB_PCA()
mbpca_model.fit(pca_data, [2,5])
print('block loadings')
print(mbpca_model.block_loadings)
print('superlevel loadings')
print( mbpca_model.superlevel_loadings)
print('superlevel scores')
print(mbpca_model.superlevel_training_scores[:5])
print('transform b')
print(mbpca_model.transform_b(pca_data.iloc[:,:2], 0)[:5])
print('predict b')
print(mbpca_model.predict_b(pca_data.iloc[:,:2], 0)[:5])
print('score b')
print(mbpca_model.score_b(pca_data.iloc[:,:2], 0))
"""

"""
#from trendfitter.models.PLS import *

pls_data = pd.read_csv('pls_dataset_complete.csv', delimiter=';', index_col=[0]).drop(columns=['ObsNum']).dropna()
#pls_data.to_csv('pls_data_w_no_na.csv')
pls_data = (pls_data - pls_data.mean()) / pls_data.std()
#print(pls_data.head())

X_pls = pls_data.drop(columns='Y-Kappa')
Y_pls = pls_data['Y-Kappa']

#X_pls = X_pls.values
#Y_pls = np.array(pls_data['Y-Kappa'].values, ndmin = 2).T

pls_model = PLS()
pls_model.fit(X_pls, Y_pls, random_state = 2)
print(pls_model.score(X_pls, Y_pls))

print(f'PLS Model fitted with : {pls_model.latent_variables} latent variables')
print('r²')
print(pls_model.score(X_pls, Y_pls))

print('training scores')
print(pls_model.training_scores[:5])
print('predicted scores')
print(pls_model.transform(X_pls[:5]))


print('P')
print(pls_model.p_loadings[:3, :])
print('W')
print(pls_model.weights[:3, :])
print('W*')
print(pls_model.weights_star[:3, :])
print('C')
print(pls_model.c_loadings[:3, :])
print('coeffs')
print(pls_model.coefficients)
print('VIPs')
print(pls_model.feature_importances_)
print('scores')
print(pls_model.training_scores[:5,:3])
print('omega')
print(pls_model.omega[:3, :3])
print('x_chi2')
print(pls_model._x_chi2_params)
print('y_chi2')
print(pls_model._y_chi2_params)
print('q2')
print(pls_model.q2y)

print('predict')
print(pls_model.predict(X_pls[:5]))
print('transform')
print(pls_model.transform(X_pls[:5]))
print('transform_inv')
print(pls_model.transform_inv(pls_model.transform(X_pls[:3])))
print('T2')
print(pls_model.Hotellings_T2(X_pls[:5]))
print('T2_limit')
print(pls_model.T2_limit(0.95))
print('SPEs X')
print(pls_model.SPEs_X(X_pls[:5]))
print('SPEs X_limit')
print(pls_model.SPE_X_limit(0.95))
print('SPEs Y')
print(pls_model.SPEs_Y(X_pls[:5], Y_pls[:5]))
print('SPEs Y_limit')
print(pls_model.SPE_Y_limit(0.95))
print('RMSEE')
print(pls_model.RMSEE(X_pls, Y_pls))
print('contributions_scores_ind')
print(pls_model.contributions_scores_ind(X_pls[:5]))
print('contributions_SPE_X')
print(pls_model.contributions_SPE_X(X_pls[:5]))
"""



smbpls_data = pd.read_csv('smb_pls_dataset.csv', delimiter=';', index_col = 0).dropna()
smbpls_data_miss = pd.read_csv('smb_pls_dataset.csv', delimiter=';', index_col = 0)
smbpls_data = (smbpls_data - smbpls_data.mean()) / smbpls_data.std()
smbpls_data_miss = (smbpls_data_miss - smbpls_data_miss.mean()) / smbpls_data_miss.std()
X = smbpls_data.drop(columns=['y'])
X[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
X[['Var4', 'Var5', 'rand4', 'rand5']] = X[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
Y = smbpls_data['y']

X_miss = smbpls_data_miss.drop(columns=['y'])
X_miss[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] = X_miss[['Var1', 'Var2', 'Var3', 'rand1', 'rand2', 'rand3']] / sqrt(6)
X_miss[['Var4', 'Var5', 'rand4', 'rand5']] = X_miss[['Var4', 'Var5', 'rand4', 'rand5']] / sqrt(4)
Y_miss = smbpls_data_miss['y']

smbpls_model = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
#smbpls_model.fit(X, [3,5], Y, latent_variables = [2, 1])
smbpls_model.fit(X, [6,10], Y, random_state = 2)#, latent_variables = [1,1])

#smbpls_model.fit(X, [3,5], Y, latent_variables = [2, 1])
"""
print(f'SMBPLS Model fitted with : {smbpls_model.latent_variables} latent variables')
print('block_p_loadings')
print(smbpls_model.block_p_loadings)
print('superlevel_p_loadings')
print(smbpls_model.superlevel_p_loadings)
print('x_weights_star')
print(smbpls_model.x_weights_star)
print('x_weights')
print(smbpls_model.x_weights)
print('superlevel_weights')
print(smbpls_model.superlevel_weights)
print('c_loadings')
print(smbpls_model.c_loadings)
print('weights_block')
print(smbpls_model.block_weights)
print(f'score is {smbpls_model.score(X, Y) * 100:.4f}%')
print(smbpls_model.q2y)
for i in range(sum(smbpls_model.latent_variables)):
    print(smbpls_model.score(X, Y, latent_variables = i+1))
    
print('Hotellings_T2')
print(smbpls_model.Hotellings_T2(X.iloc[:5]))
"""

smbpls_model_miss = SMB_PLS(tol = 1e-8, cv_splits_number = 7)
smbpls_model_miss.fit(X_miss, [6,10], Y_miss, random_state = 2)
print('miss data section')
print(f'SMBPLS Model fitted with : {smbpls_model_miss.latent_variables} latent variables')
print('scores')
print(smbpls_model_miss.transform(X.iloc[:5]))
print('scores calculated with x_weights star (pinv expression)')
new_block_scores = X_miss.values @ smbpls_model_miss.x_weights_star.T
print(new_block_scores[:5])
print('scores calc with x_weights star2 (G matrix expression)')
new_block_scores = X_miss.values @ smbpls_model_miss.x_weights_star2.T
print(new_block_scores[:5])

print('block_p_loadings')
print(smbpls_model_miss.block_p_loadings)
print('superlevel_p_loadings')
print(smbpls_model_miss.superlevel_p_loadings)

print('x_weights_star')
print(smbpls_model_miss.x_weights_star)
print('x_weights')
print(smbpls_model_miss.x_weights)

print('superlevel_weights')
print(smbpls_model_miss.superlevel_weights)

print('weights_block')
print(smbpls_model_miss.block_weights)
print('c_loadings')
print(smbpls_model_miss.c_loadings)
print('Q2')
print(smbpls_model_miss.q2y)
print('scores')
print(smbpls_model_miss.transform(X_miss[:5]))
print('predictions')
print(smbpls_model_miss.predict(X_miss[:5]))
print(f'score is {smbpls_model_miss.score(X_miss[:5], Y_miss.iloc[:5])*100:.2f}%')
#print('Hotellings T²')
#print(smbpls_model_miss.Hotellings_T2(X_miss[:5]))

"""
mbpls_model = MB_PLS()
print(X_pls.shape)
block_divs = [10, 21]
mbpls_model.fit(X_pls, block_divs, Y_pls, latent_variables=2)
print('SL Weights')
print(mbpls_model.superlevel_weights)
print('SL Scores')
print(mbpls_model.training_sl_scores[:5,:])
print('block loadings')
print(mbpls_model.block_p_loadings)
print('block weights')
print(mbpls_model.block_weights)

print('transform_b')
print(mbpls_model.transform_b(X_pls.iloc[:,:10], 0)[:5,:])
print('score_b')
print(mbpls_model.score_b(X_pls.iloc[:,:10], 0))

dipls_model = DiPLS()
LV = 3
s = 3
dipls_model.fit(X_pls, Y_pls, latent_variables = LV, s = s)
print(f'DiPLS r² is {dipls_model.score(X_pls, Y_pls):.2f}')



q2_model = DiPLS()
q2_model.fit(X_pls.iloc[:-200], Y_pls.iloc[:-200], latent_variables = LV, s = s) # this model is trained with only a partition of the total dataset
q2_final = q2_model.score(X_pls.iloc[-200: :], Y_pls.iloc[-200:]) # its performance is registered in a list
print(f'DiPLS q² is {q2_final:.2f}')

kf = KFold(n_splits = 7 , random_state = 2)
pls_model = PLS()
pls_model.fit(X_pls, Y_pls)
print(f'PLS r² is {pls_model.score(X_pls, Y_pls):.2f}')
testq2 = []
for train_index, test_index in kf.split(X_pls):
    q2_model = PLS()
    q2_model.fit(X_pls.iloc[train_index], Y_pls.iloc[train_index], latent_variables = pls_model.latent_variables) # this model is trained with only a partition of the total dataset
    testq2.append(q2_model.score(X_pls.iloc[test_index], Y_pls.iloc[test_index])) # its performance is registered in a list
q2_final = mean(testq2)
print(f'PLS q² is {q2_final:.2f}')


from numpy.random import default_rng
rng = default_rng()

X1 = rng.standard_normal((1000, 2))
X2 = rng.standard_normal((1000, 2))
X3 = rng.standard_normal((1000, 2)) + X1 * 0.2 + X2 * 0.3
all_Xs = np.concatenate([X1, X2, X3], axis = 1)
second_level = [4, 6]
third_level = [2, 4]

#Y = all_Xs @ wstar.T
#Y = (Y - np.mean(Y)) / np.std(Y)
Y = np.array(np.mean(all_Xs, axis = 1), ndmin = 2 ).T
Y = (Y - np.mean(Y)) / np.std(Y)

model = MLSMB_PLS()
model.fit(all_Xs,third_level, second_level, Y, latent_variables = [1, 1])
"""