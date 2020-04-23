from numpy import zeros
from numpy.linalg import det, pinv
from math import floor
from itertools import product
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:31:43 2020

auxiliary functions for the Trendfitter library

- Score calculation with missing values

@author: Daniel Rodrigues
"""

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

def scores_with_missing_values( omega, loadings, X_matrix, method = 'TSR', inverse_method = None, inverse_term = None ):
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

    """


    if method == 'TSR' : #estimate using the 'TSR' method
        
        B = omega.dot( loadings ).dot( loadings.T ).dot( pinv( loadings.dot( loadings.T ).dot( omega ).dot( loadings ).dot( loadings.T ) ) ).dot( loadings )  # OMEGA.P*'.P*.(P*'.P*.OMEGA.P.P*')^-1         
        scores = B.dot( X_matrix.T )
        
    elif method == 'CMR' : # estimate using the 'CMR' method
        
        B = omega.dot( loadings ).dot( pinv( loadings.T.dot( omega ).dot( loadings ) ) )
        scores = B.dot( X_matrix )
        
    return scores.T