from numpy import zeros, array
from numpy.linalg import det, inv, pinv
from math import floor
from itertools import product
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:31:43 2020

auxiliary functions for the Trendfitter library

- Score calculation with missing values

@author: Daniel Rodrigues
"""

def scores_with_missing_values( omega, loadings, X_matrix, LVs = None, method = 'TSM', inverse_method = None, inverse_term = None ):
    """

    function to estimate missing values using different techniques. 
    
    TSR - Trimmed Score Regression  - As in Arteaga & Ferrer's " Dealing with missing data in MSPC: several methods, different interpretations, some examples" (2002)

    CMR - Conditional Mean Replacement - as in Nelston, et al.'s "Missing Data methods in PCA and PLS: Score calculations with incomplete observations" (1996)

    Parameters:

    Omega

    loadings

    X_matrix

    method:

    inverse_method:

    inverse_term:
    
    returns
    
    score: vector of predicted scores

    """
    if LVs == None : LVs = omega.shape[0]

    if method == 'TSR' : #estimate using the 'TSR' method - Trimmed Score Regression
        
        B_1 = omega[ :LVs, :LVs ] @ loadings[ :LVs ] @ loadings[ :LVs ].T  # OMEGA.P*'.P*
        B_2 = pinv( loadings[ :LVs ] @ loadings.T @ omega @ loadings @ loadings[ :LVs ].T ) # (P*'.P*.OMEGA.P'.P*)^-1 
        B_factor = B_1 @ B_2 @ loadings[ :LVs ]   # OMEGA.P*'.P*.(P*'.P*.OMEGA.P.P*')^-1.P*'
       # print(B_factor)
        scores = ( B_factor @ X_matrix.T ).reshape( X_matrix.shape[ 0 ], -1 )         

    elif method == 'CMR' : # estimate using the 'CMR' method - Conditional mean replacement method
        B_1 = omega[ :LVs, :LVs ] @ loadings[ :LVs ] # OMEGA.P*'
        B_2 = pinv( loadings.T @ omega @ loadings )  #(P*.OMEGA.P*')^-1
        B_factor = B_1 @ B_2 #OMEGA.P*'.(P*.OMEGA.P*')^-1
        #print(B_factor)
        scores = ( B_factor @ X_matrix.T ).reshape( X_matrix.shape[ 0 ], -1 )
    
    elif method == 'TSM' : # Trimmed score method - just substitute all nans for zeros
        scores = X_matrix @ loadings[ :LVs ].T

    elif method == 'PMP' : #Projection to the Model Plane method
        B_factor = pinv( loadings[ :LVs ] @ loadings[ :LVs ].T ) @ loadings[ :LVs ]
        #print(B_factor)
        scores = ( B_factor @ X_matrix.T ).reshape( X_matrix.shape[ 0 ], -1 )

    else : raise Exception('User inputted method "{}" not implemented'.format(method))

    return scores

    """
def subinverse( main_inverse_term, desired_variables ):
        
    
    Because of the necessity of calculating multiple matrix inverses, this functions uses a method 
    published in Juarez-Ruiz, E. et al.'s "Relationship between the Inverses of a Matrix and a Submatrix"
    to calculate such inverses using a method that uses the full-inverse as basis. (2016) 
    
    
    final_inv_num_total = zeros( main_inverse_term.shape[ 0 ] - len( desired_variables ) )
    i, j = 0, 0
    final_inv_den_matrix = zeros( len( desired_variables ) )
    
    i = 0
    for q, p in product( desired_variables, repeat = 2 ):
        final_inv_den_matrix[ i % len( desired_variables ) , floor( i / len ( desired_variables ) ) ] = main_inverse_term[ q, p ]
        i += 1

    final_inv_den = det( final_inv_den_matrix )
    
    for i in range( main_inverse_term.shape[ 0 ] - len( desired_variables ) ):
        
        for j in range( main_inverse_term.shape[ 0 ] - len( desired_variables ) ):
            
            final_inv_num = zeros( main_inverse_term.shape[ 0 ] - len( desired_variables ) )
            final_inv_num[ :-1, :-1 ] = final_inv_den_matrix
            
            k = 0
            for qp in desired_variables:
                final_inv_num[ -1, k ] = main_inverse_term[ i, qp ]
                final_inv_num[ k, -1 ] = main_inverse_term[ qp, j ]
                k += 1
            final_inv_num[ -1, -1 ] = main_inverse_term[ i, j ]
            
            final_inv_num_total[ i , j ] = final_inv_num
            
            j += 1
        
        i += 1
        
        inverse_matrix = det( final_inv_num_total ) / final_inv_den
        
    return inverse_matrix
"""