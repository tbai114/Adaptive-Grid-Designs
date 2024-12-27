# -*- coding: utf-8 -*-

import pandas as pd
import math
import copy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from hyperopt import fmin, tpe, hp
from itertools import product
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri,r
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import numpy2ri
from sklearn.gaussian_process.kernels import RBF
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
####py documents for generate PALC design###########
from active_learning import ActiveLearningMachine
from partitioned_gp import PGP
from strategy_lib import pimse
#########Active learning for SVM####################
from skactiveml.pool import UncertaintySampling,ContrastiveAL
from skactiveml.utils import MISSING_LABEL,labeled_indices
from skactiveml.classifier import SklearnClassifier
####################################################


##############State Function####################

def f_p(x, p):
    '''
    y(x) used in Section 4 and Bachoc et al. 
    Parameters
    ----------
    x : list
        Design point x with p dimensions.
    p : integer
        A positive integer represents dimension.
    Returns
    -------
    f : float
        The function value of given point x.
    '''
    f = 0
    #for i in range(1, p+1):
        #f = f+np.arctan(5*(1-i/(p+1))*x[i-1])

    f=x[0]**4+x[1]**3+15*max(x[0]-0.5,0)**1.3+3*max(x[1]-0.2,0)**0.4
    #f=np.arctan(5*x[0])+np.arctan(2*x[1])+x[2]+2*x[3]**2+2/(1+math.exp(-10*(x[4]-0.5)))
    return f

def f_state(x, p, q):
    '''
    State function f(), generate binary response with -1 and 1.
    
    Parameters
    ----------
    x : list
        Design point x with p dimensions.
    p : integer
        A positive integer represents dimension.
    q : float
        A tunning parameter in [0,1], define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).

    Returns
    -------
    integer
        Binary response, with -1 represents negative repsonse and 1 represents positive response
    '''
    a = [1 for i in range(0, p)]
    if f_p(x, p) > q*f_p(a, p):
        return 1
    else:
        return -1

##############Basis Fucntions used for Adaptive Grid Designs####################
def generate_p_center_grid(p, l):
    '''
    Generate center grid {1/2^l,3/2^l,\cdots,(2^l-1)/2^l}^p.
    Used for compute V(U) for grid-based designs.
    
    Parameters
    ----------
    p : integer
        A positive integer represents dimension.
    l : integer
        A positive integer represents generation.
    Returns
    -------
    result : list
        p-dimensional center grid {1/2^l,3/2^l,\cdots,(2^l-1)/2^l}^p, 
        with each element to be one point.
    '''
    init = []
    for i in range(0, 2**(l-1)):
        q = 2*i+1
        init.append(q/2**l)
    result = list(product(init, repeat=p))
    return result


def generate_p_grid(p, l):  
    '''
    Generate p-dimension grid  D_{\text{SG},p,(2^l+1)^p} for GG and AG.
    Parameters
    ----------
    p : integer
        A positive integer represents dimension.
    l : integer
        A non-negative integer represents generation.
    Returns
    -------
    result : list
        p-dimensional grid design D_{\text{SG},p,(2^l+1)^p}, 
        with each element to be one point.
    '''
    init = []
    for i in range(0, 2**l+1):
        init.append(i/2**l)
    result = list(product(init, repeat=p))
    return result


def generate_p_noboundary_grid(p, l):
    '''
    Generate p-dimension inner grid  D_{\text{SI},p,(2^l-1)^p} for GI and AI.
    Parameters
    ----------
    p : integer
        A positive integer represents dimension.
    l : integer
        A positive integer represents generation.
    Returns
    -------
    result : list
        p-dimensional grid design D_{\text{SG},p,(2^l+1)^p}, 
        with each element to be one point.
    '''
    init = []
    for i in range(1, 2**l):
        init.append(i/2**l)
    result = list(product(init, repeat=p))
    return result


def FindIndex(searchX, AGridData, p):
    '''
    Find if x is already in the Design.

    Parameters
    ----------
    searchX : list
        List of a given point x represents its p-dimension.
    AGridData : np.array
        Design matrix generate in grid-based designs.
    p : inter
        A positive integer represents dimension.

    Returns
    -------
    int
       If larger than 0: the index of given x in the design matrix D
       0: not in the design matrix D.
    '''
    number = np.arange(1, len(AGridData)+1, 1)
    LookData = np.column_stack((AGridData, number))
    del_index = []
    for jj in range(0, p):
        for k in range(0, len(LookData)):
            if LookData[k, 3+jj] != searchX[jj]:
                del_index.append(k)
    LookData = np.delete(LookData, del_index, 0)
    if len(LookData) > 0:
        return LookData[0, 3+p]
    if len(LookData) == 0:
        return 0


def FindResponse(searchX, Grid, p):
    '''
    Find the response of x if x can be derived by monotonic information in the data set.
    Parameters
    ----------
    searchX :  list
        List of a given point x represents its p-dimension.
    Grid :  np.array
        Design matrix generate in grid-based designs.
    p : inter
        A positive integer represents dimension.

    Returns
    -------
    resp : integer
        Response of given x and design matrix.
        -1: f(x)=-1 and do not need evaluation
        1: f(x)=1 and do not need evaluation
        0: x is in the uncertain area U(D) and needs evaluation
    '''
    AGrid = copy.deepcopy(Grid)
    searchP1 = copy.deepcopy(searchX)
    AGrid = AGrid[AGrid[:, 1] == 3]
    resp = 0
    for grids in AGrid:
        if grids[0] == 1:
            data = grids[3:3+p]
            if (np.array(searchP1-data) >= 0).all() == True:
                resp = 1
                break
        if grids[0] == -1:
            data = grids[3:3+p]
            if (np.array(searchP1-data) <= 0).all() == True:
                resp = -1
                break
    return resp

################Grid-based designs, SG, GG, and AG######################
def SG(p, m):
    '''
    Static grid design SG
    ----------
    p : int
        Dimension, a postive integer.
    m: int
        Index for generation, each dimension is equally divided into m portions.
        Remark that we finally generate D_{\text{SG},p,(m+1)^p}.
    Returns
    -------
    result : list
        (m+1)^p length design list, with each element represents one design point.
    '''
    init = []
    for i in range(0, m+1):
        init.append(i/m)
    result = list(product(init, repeat=p))
    return result

def GG(p, n, q):
    '''
    Group-adaptive grid design GG
    Parameters
    ----------
    p : int
        Dimension, a postive integer.
    n : int
        Sample size, a postive integer.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, 2 represents skipped by monotonic information;
        Generation:the points belongs to which generation grid points, e.g., (0.75,0.5) belongs to generation 2, see parameter l for more information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    l : integer
        The generation index after n sample size GG, represents the GG generate points in D_{\text{SG},p,(2^l+1)^p}\setminus D_{\text{SG},p,(2^{l-1}+1)^p}.
        Remark that we use l to compute the volume of uncertain area V(U).
    '''
    Data = np.empty([0, 3+p])
    l = 0
    while np.sum(Data[:, 1] == 3) < n:
        if l > 0:
            grid = list(set(generate_p_grid(p, l)) -
                        set(generate_p_grid(p, l-1)))
        else:
            grid = generate_p_grid(p, l)
        K = []
        for g in grid:
            resp = FindResponse(g, Data, p)
            if resp != 0:
                ls = [resp, 2, l]
                Data = np.row_stack((Data, np.hstack((ls, g))))
                K.append(g)
        grid1 = list(set(grid)-set(K))
        length = len(grid1)
        for i in range(0, length):
            IndexNext = i
            g = grid1[IndexNext]
            res = f_state(g, p, q)
            ls = [res, 3, l]
            Data = np.row_stack((Data, np.hstack((ls, g))))
            if np.sum(Data[:, 1] == 3) == n:
                break
        if np.sum(Data[:, 1] == 3) == n:
            break
        l = l+1
    return Data, l


def AG(p, n1, q):
    '''
    Fully-adaptive grid design AG with order that sequentially maximize min{card(A_x),card(B_x)} and max{card(A_x),card(B_x)}
    Parameters
    ----------
    p : int
        Dimension, a postive integer.
    n : int
        Sample size, a postive integer.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, we do not record the skipped points here. One may derive them use D_{\text{SG},p,(2^l+1)^p}\setminus D_{\text{AG},p,m_{\text{AG}}(l)};
        Generation:the points belongs to which generation grid points, e.g., (0.75,0.5) belongs to generation 2, see parameter maxg for more information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    maxg : integer
        The generation index after n sample size AG, represents the AG generate points in D_{\text{SG},p,(2^l+1)^p}\setminus D_{\text{SG},p,(2^{l-1}+1)^p}.
        Remark that we use maxg to compute the volume of uncertain area V(U).
    '''
    Data = np.empty([0, 3+p])
    maxg = 0
    while np.sum(Data[:, 1] == 3) < n1:
        FData = copy.deepcopy(Data)
        grid = []
        grid1 = generate_p_grid(p, maxg)
        if maxg != 0:
            grid2 = generate_p_grid(p, maxg-1)
            for x in grid1:
                if x not in grid2:
                    grid.append(x)
        else:
            grid = grid1

        FData = FData[FData[:, 1] == 3]
        Alist = FData[FData[:, 0] == -1][:, 3:3+p]
        Blist = FData[FData[:, 0] == 1][:, 3:3+p]
        U = []
        for xk in grid:
            q1 = 0
            q2 = 0
            for yk in Alist:
                if all(x <= y for x, y in zip(xk, yk)):
                    q1 = 1
                    break
            if q1 == 0:
                for yk in Blist:
                    if all(x >= y for x, y in zip(xk, yk)):
                        q2 = 1
                        break
            if q1 == 0 and q2 == 0:
                U.append(xk)
        ul = len(U)
        Umatrix = np.eye(ul) * 2
        # Save the information of Ax, Bx, and status: 1: uncertain, 0: certain in this generation 2:certain in past generation
        Ordermatrix = np.zeros((ul, 3))
        Ordermatrix[:, 2] = 1
        for i in range(0, ul):
            xi = U[i]
            for j in range(0, ul):
                xj = U[j]
                if j != i:
                    if all(x <= y for x, y in zip(xi, xj)):
                        Umatrix[i, j] = -1
                    elif all(x >= y for x, y in zip(xi, xj)):
                        Umatrix[i, j] = 1
                    else:
                        Umatrix[i, j] = 0
        for i in range(0, ul):
            Ordermatrix[i, 0] = np.sum(Umatrix[i] == 1)
            Ordermatrix[i, 1] = np.sum(Umatrix[i] == -1)
        while np.sum(Ordermatrix[:, 2] == 1) > 0:
            cmin = []
            cmax = []
            for i in range(0, ul):
                cmin.append(min(Ordermatrix[i, 0], Ordermatrix[i, 1]))
                cmax.append(max(Ordermatrix[i, 0], Ordermatrix[i, 1]))
            max_value = max(cmin)
            max_indices1 = [index for index,
                            value in enumerate(cmin) if value == max_value]
            max_indices = [x for x in max_indices1 if Ordermatrix[x, 2] == 1]
            if len(max_indices) > 1:
                maxnumber = 0
                IndexNext = max_indices[0]
                for i in range(0, len(max_indices)):
                    maxnum = cmax[max_indices[i]]
                    if maxnum >= maxnumber:
                        maxnumber = maxnum
                        IndexNext = max_indices[i]
            else:
                IndexNext = max_indices[0]

            g = U[IndexNext]
            gen = maxg
            res = f_state(g, p, q)
            ls = [res, 3, gen]
            Data = np.row_stack((Data, np.hstack((ls, g))))
            ##Update Ordermatrix according to Umatrix##
            Umatrix[IndexNext, IndexNext] = res
            # Update Ak and Bk
            for i in range(0, ul):
                if Ordermatrix[i, 2] == 1:
                    if i != IndexNext:
                        if res == -1:
                            if Umatrix[i, IndexNext] == -1:
                                Umatrix[i, i] = -1
                                Ordermatrix[i, :] = 0
                            if Umatrix[i, IndexNext] == 1:
                                Ordermatrix[i, 0] = Ordermatrix[i,
                                                                0]-Ordermatrix[IndexNext, 0]-1
                        if res == 1:
                            if Umatrix[i, IndexNext] == 1:
                                Umatrix[i, i] = 1
                                Ordermatrix[i, :] = 0
                            if Umatrix[i, IndexNext] == -1:
                                Ordermatrix[i, 1] = Ordermatrix[i,
                                                                1]-Ordermatrix[IndexNext, 1]-1
            for i in range(0, ul):
                if Umatrix[i, IndexNext] == 0:
                    if Ordermatrix[i, 2] == 1:
                        for j in range(0, ul):
                            if Ordermatrix[j, 2] == 0:
                                if res == -1:
                                    if Umatrix[i, j] == 1:
                                        Ordermatrix[i, 0] = Ordermatrix[i, 0]-1
                                else:
                                    if Umatrix[i, j] == -1:
                                        Ordermatrix[i, 1] = Ordermatrix[i, 1]-1
            Ordermatrix[IndexNext, :] = 0
            Ordermatrix[:, 2][Ordermatrix[:, 2] == 0] = 2
            if np.sum(Data[:, 1] == 3) == n1:
                break
        if np.sum(Data[:, 1] == 3) == n1:
            break
        maxg = maxg+1

    return Data, maxg

################Inner-grid-based designs, SI, GI, and AI######################
def SI(p, m):
    '''
    Static inner grid design SI
    ----------
    p : int
        Dimension, a postive integer.
    m: int
        Index for generation, each dimension is equally divided into m portions.
        Remark that we finally generate D_{\text{SI},p,(m-1)^p}.
    Returns
    -------
    result : list
        (m-1)^p length design list, with each element represents one design point.
    '''
    init = []
    for i in range(1, m):
        init.append(i/m)
    result = list(product(init, repeat=p))
    return result

def GI(p, n, q):
    '''
    Group-adaptive inner grid design GI
    Parameters
    ----------
    p : int
        Dimension, a postive integer.
    n : int
        Sample size, a postive integer.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, 2 represents skipped by monotonic information;
        Generation:the points belongs to which generation grid points, e.g., (0.75,0.5) belongs to generation 2, see parameter l for more information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    l : integer
        The generation index after n sample size GI, represents the GI generate points in D_{\text{SI},p,(2^l-1)^p}\setminus D_{\text{SI},p,(2^{l-1}-1)^p}.
        Remark that we use l to compute the volume of uncertain area V(U).
    '''
    Data = np.empty([0, 3+p])
    l = 1
    while np.sum(Data[:, 1] == 3) < n:
        grid = list(set(generate_p_noboundary_grid(p, l)) -
                    set(generate_p_noboundary_grid(p, l-1)))
        K = []
        for g in grid:
            resp = FindResponse(g, Data, p)
            if resp != 0:
                ls = [resp, 2, l]
                Data = np.row_stack((Data, np.hstack((ls, g))))
                K.append(g)
        grid1 = list(set(grid)-set(K))
        length = len(grid1)
        for i in range(0, length):
            IndexNext = i
            g = grid1[IndexNext]
            res = f_state(g, p, q)
            ls = [res, 3, l]
            Data = np.row_stack((Data, np.hstack((ls, g))))
            if np.sum(Data[:, 1] == 3) == n:
                break
        if np.sum(Data[:, 1] == 3) == n:
            break
        l = l+1
    return Data, l

def AI(p, n1, q):
    '''
    Fully-adaptive inner grid design AI with order that sequentially maximize min{card(A_x),card(B_x)} and max{card(A_x),card(B_x)}
    Parameters
    ----------
    p : int
        Dimension, a postive integer.
    n : int
        Sample size, a postive integer.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, we do not record the skipped points here. One may derive them use D_{\text{SI},p,(2^l-1)^p}\setminus D_{\text{AI},p,m_{\text{AI}}(l)};
        Generation:the points belongs to which generation grid points, e.g., (0.75,0.5) belongs to generation 2, see parameter maxg for more information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    maxg : integer
        The generation index after n sample size AI, represents the AI generate points in D_{\text{SI},p,(2^l-1)^p}\setminus D_{\text{SI},p,(2^{l-1}-1)^p}.
        Remark that we use l to compute the volume of uncertain area V(U).
    '''

    Data = np.empty([0, 3+p])
    maxg = 1
    while np.sum(Data[:, 1] == 3) < n1:
        FData = copy.deepcopy(Data)
        grid = []
        grid1 = generate_p_noboundary_grid(p, maxg)
        grid2 = generate_p_noboundary_grid(p, maxg-1)
        for x in grid1:
            if x not in grid2:
                grid.append(x)

        FData = FData[FData[:, 1] == 3]
        Alist = FData[FData[:, 0] == -1][:, 3:3+p]
        Blist = FData[FData[:, 0] == 1][:, 3:3+p]
        U = []
        for xk in grid:
            q1 = 0
            q2 = 0
            for yk in Alist:
                if all(x <= y for x, y in zip(xk, yk)):
                    q1 = 1
                    break
            if q1 == 0:
                for yk in Blist:
                    if all(x >= y for x, y in zip(xk, yk)):
                        q2 = 1
                        break
            if q1 == 0 and q2 == 0:
                U.append(xk)
        ul = len(U)
        if len(U) == 0:
            maxg = maxg+1
            grid = generate_p_grid(p, maxg)
            for xk in grid:
                q1 = 0
                q2 = 0
                for yk in Alist:
                    if all(x <= y for x, y in zip(xk, yk)):
                        q1 = 1
                        break
                if q1 == 0:
                    for yk in Blist:
                        if all(x >= y for x, y in zip(xk, yk)):
                            q2 = 1
                            break
                if q1 == 0 and q2 == 0:
                    U.append(xk)
        m = 1/2**maxg
        gen_all = []
        gen_sum = []
        for g in U:
            gens = [0 for i in range(0, p)]
            for j in range(0, p):
                if g[j] != 0:
                    for gg in range(0, maxg+2):
                        if g[j]/m/2**gg != math.floor(g[j]/m/2**gg):
                            break
                if g[j] == 0:
                    gg = maxg+1
                gens[j] = maxg-gg+1
            gen = max(gens)  # generation
            gensum = sum(gens)
            gen_all.append(gen)
            gen_sum.append(gensum)

        gen_all1 = [maxg*p*i for i in gen_all]

        if ul <= 30000:  # if ul<=30000, memory is sufficient, use the same order with AG.
            Umatrix = np.eye(ul) * 2
            # Save the information of Ax, Bx, and status: 1: uncertain, 0: certain in this generation 2:certain in past generation
            Ordermatrix = np.zeros((ul, 3))
            Ordermatrix[:, 2] = 1
            for i in range(0, ul):
                xi = U[i]
                for j in range(0, ul):
                    xj = U[j]
                    if j != i:
                        if all(x <= y for x, y in zip(xi, xj)):
                            Umatrix[i, j] = -1
                        elif all(x >= y for x, y in zip(xi, xj)):
                            Umatrix[i, j] = 1
                        else:
                            Umatrix[i, j] = 0
            for i in range(0, ul):
                Ordermatrix[i, 0] = np.sum(Umatrix[i] == 1)
                Ordermatrix[i, 1] = np.sum(Umatrix[i] == -1)
            while np.sum(Ordermatrix[:, 2] == 1) > 0:
                cmin = []
                cmax = []
                for i in range(0, ul):
                    cmin.append(min(Ordermatrix[i, 0], Ordermatrix[i, 1]))
                    cmax.append(max(Ordermatrix[i, 0], Ordermatrix[i, 1]))
                max_value = max(cmin)

                max_indices1 = [index for index,
                                value in enumerate(cmin) if value == max_value]
                max_indices = [
                    x for x in max_indices1 if Ordermatrix[x, 2] == 1]
                if len(max_indices) > 1:
                    maxnumber = 0
                    IndexNext = max_indices[0]
                    for i in range(0, len(max_indices)):
                        maxnum = cmax[max_indices[i]]
                        if maxnum >= maxnumber:
                            maxnumber = maxnum
                            IndexNext = max_indices[i]
                else:
                    IndexNext = max_indices[0]

                g = U[IndexNext]
                gen = maxg
                res = f_state(g, p, q)
                ls = [res, 3, gen]
                Data = np.row_stack((Data, np.hstack((ls, g))))
                ##Update Ordermatrix according to Umatrix##
                Umatrix[IndexNext, IndexNext] = res
                # Update Ak and Bk
                for i in range(0, ul):
                    if Ordermatrix[i, 2] == 1:
                        if i != IndexNext:
                            if res == -1:
                                if Umatrix[i, IndexNext] == -1:
                                    Umatrix[i, i] = -1
                                    Ordermatrix[i, :] = 0
                                if Umatrix[i, IndexNext] == 1:
                                    Ordermatrix[i, 0] = Ordermatrix[i,
                                                                    0]-Ordermatrix[IndexNext, 0]-1
                            if res == 1:
                                if Umatrix[i, IndexNext] == 1:
                                    Umatrix[i, i] = 1
                                    Ordermatrix[i, :] = 0
                                if Umatrix[i, IndexNext] == -1:
                                    Ordermatrix[i, 1] = Ordermatrix[i,
                                                                    1]-Ordermatrix[IndexNext, 1]-1
                for i in range(0, ul):
                    if Umatrix[i, IndexNext] == 0:
                        if Ordermatrix[i, 2] == 1:
                            for j in range(0, ul):
                                if Ordermatrix[j, 2] == 0:
                                    if res == -1:
                                        if Umatrix[i, j] == 1:
                                            Ordermatrix[i,
                                                        0] = Ordermatrix[i, 0]-1
                                    else:
                                        if Umatrix[i, j] == -1:
                                            Ordermatrix[i,
                                                        1] = Ordermatrix[i, 1]-1
                Ordermatrix[IndexNext, :] = 0
                Ordermatrix[:, 2][Ordermatrix[:, 2] == 0] = 2
                if np.sum(Data[:, 1] == 3) == n1:
                    break
        else: 
            #####################################################################
            #if ul>30000, memory is insufficient, we use another order instead of 
            #maximize min{card(A_x),card(B_x)} and max{card(A_x),card(B_x)},
            #until ul<=30000. This is used for large p such as p=6.
            #We check the ul every 50 points .
            #####################################################################
            n = 0
            while n < 50:
                IndexNext = np.argsort(
                    np.sum([gen_all1, gen_sum], axis=0).tolist())[ul-n-1]
                gen = gen_all[IndexNext]
                g = U[IndexNext]
                Index = int(FindIndex(g, Data, p))
                if Index == 0:
                    resp = FindResponse(g, Data, p)
                    if resp == 0:
                        res = f_state(g, p, q)
                        ls = [res, 3, gen]
                    else:
                        ls = [resp, 2, gen]
                    Data = np.row_stack((Data, np.hstack((ls, g))))
                n = n+1
                if np.sum(Data[:, 1] == 3) == n1:  
                    break
            if np.sum(Data[:, 1] == 3) == n1:  
                break
        if np.sum(Data[:, 1] == 3) == n1:
            break
    return Data, maxg

############Adaptive Monte Carlo design AMC####################
def AMC(p, n, q):
    '''
    Adaptive Monte Carlo design AMC proposed in de Rocquigny

    Parameters
    ----------
    p : integer
        A postive integer represents dimension.
    n : integer
        A postive integer represents sample size.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).

    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, ;
        Index:the order in evaluation, 2 represents skipped by monotonic information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    '''
    Data = np.empty([0, 3+p])  
    i = 0
    while(i < n):
        i = i+1
        if_u = 0
        x_candidate = np.random.uniform(low=[0]*p, high=[1]*p)

        while(if_u < 1):
            resp = FindResponse(x_candidate, Data, p)
            if resp == 0:
                if_u = 2
                res = f_state(x_candidate, p, q)
                ls = [res, 3, i]
            else:
                ########If you wants to record the skipped points, run the following codes####
                #ls=[resp,2,i]
                #Data = np.row_stack((Data, np.hstack((ls, x_candidate))))
                ##############################################################################
                x_candidate = np.random.uniform(low=[0]*p, high=[1]*p)
        Data = np.row_stack((Data, np.hstack((ls, x_candidate))))
    return Data

############Minimum Energy design in uncertain area##############
def Med(n,p,n_init,q):
    '''
    Minimum energy design proposed by V. Roshan Joseph and generate by R package Mined

    Parameters
    ----------
    p : integer
        A postive integer represents dimension.
    n : integer
        A postive integer represents sample size.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    n_init:
        A postive integer represents initial sample size. The initial design is n_init mmLHD design generate by R package SLHD.
        The recommend value is 10p.
        
    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, ;
        Index:the order in evaluation, 2 represents skipped by monotonic information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    '''
    pandas2ri.activate()
    med=importr('mined')
    slhd=importr('SLHD')
    init_1=slhd.maximinSLHD(t = 1, m =n_init, k = p)
    init=np.array(init_1.rx2('StandDesign'))
    Data = np.empty([0, 3+p])
    for x in range(0,len(init)):
        res=f_state(init[x],p,q)
        ls=[res,3,x]
        Data = np.row_stack((Data, np.hstack((ls, init[x]))))
    j=0
    while(j<n):
        cand=np.random.uniform(low=[0] * p, high=[1] * p, size=(10000, p))
        cand_df = pd.DataFrame(cand, columns=[f'dim_{i+1}' for i in range(p)])
        candlf=[]
        Data1=copy.deepcopy(Data)
        for i in range(0,len(cand)):
            resp=FindResponse(np.array(cand[i]), Data1, p)
            if resp==0:
                candlf.append(10)
            else:
                candlf.append(0)
        candlf_df = pd.DataFrame(candlf, columns=['candlf'])
        sumdf=pd.Series()
        for i in range(0,p):
            sumdf=pd.concat([sumdf,cand_df['dim_%d'%(i+1)]])

        sumlist=sumdf.tolist()
        robjects.r.matrix(robjects.IntVector(range(10)), nrow=5)
        cand_r = robjects.r.matrix(robjects.FloatVector(sumlist), nrow=len(cand))
        candlf_r = robjects.r.matrix(robjects.FloatVector(candlf_df['candlf'].values), nrow=candlf_df.shape[0], ncol=1)
        res=med.SelectMinED(cand_r,candlf_r,1,1,2)

        res1=res.rx2('points')
        for k in range(0,1):
            res2=f_state(res1[0],p,q)
            ls=[res2,3,n_init+j]
            Data = np.row_stack((Data, np.hstack((ls, res1[0]))))
        j=j+1
    return Data


##########PALC method from Lee et al. 2023########
def palc(n,p,q,n_init):
    '''
    Partitioned active learning Cohn design proposed by Lee et al. 2023
    Need active_learning.py, partitioned_gp.py, and strategy_lib.py
    Can be download from https://github.com/cheolheil/ALIEN?tab=readme-ov-file
    Generate palc design with default setting:
        kernel: RBF
        n_partitions:2
        X_pool: 10,000 p dimension uniform points
        X_ref: 10,000 p dimension uniform points
    Parameters
    ----------
    p : integer
        A postive integer represents dimension.
    n : integer
        A postive integer represents sample size.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    n_init:
        A postive integer represents initial sample size. The initial design is n_init mmLHD design generate by R package SLHD.
        The recommend value is 10p.
        
    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, ;
        Index:the order in evaluation, 2 represents skipped by monotonic information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    '''
    slhd=importr('SLHD')
    init_1=slhd.maximinSLHD(t = 1, m =n_init, k = p)
    init=np.array(init_1.rx2('StandDesign'))
    Data = np.empty([0, 3+p])
    for x in range(0,len(init)):
        res=f_state(init[x],p,q)
        ls=[res,3,x]
        Data = np.row_stack((Data, np.hstack((ls, init[x]))))
    
    X_init=Data[:,3:3+p]
    y_init=Data[:,0]
    kernel = 1.0 * RBF(1.0)
    j=0
    model = PGP(kernel=kernel,n_partitions=2)
    al_machine = ActiveLearningMachine(model, pimse)
    # Initial fit
    al_machine.init_fit(X_init, y_init)
    while j<n:
        X_pool=np.random.uniform(low=[0] * p, high=[1] * p, size=(10000, p))
        X_ref=np.random.uniform(low=[0] * p, high=[1] * p, size=(10000, p))
        x_new = al_machine.query(X_pool,False,X_ref)
        y_new = f_state(x_new[0],p,q)  
        al_machine.update(x_new, y_new)
        ls=[y_new,3,n_init+j]
        Data = np.row_stack((Data, np.hstack((ls, x_new[0]))))
        j=j+1
    return Data


##########Active leraning for SVM########


def ActiveLearningSVM(n,p,q,n_init,method='ContrastiveAL'):
    '''
    Active learning methods using package skactiveml. 
    Different options including CAL, ALE, and ALM.
    The sample pool is randomly generated by 10,000 MC points.
    For more details, please see https://github.com/scikit-activeml/scikit-activeml
    
    Parameters
    ----------
    n : integer
        A postive integer represents sample size.
    p : integer
        A postive integer represents dimension.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    n_init : integer
        A postive integer represents initial sample size. The initial design is n_init mmLHD design generate by R package SLHD.
        The recommend value is 10p.
    method : string, optional
        Method used for active learning. 
        Options:
                  - 'ContrastiveAL' (default): Contrastive Active Learning (CAL)
                  - 'entropy': Active learning with uncertainty using entropy (ALE)
                  - 'margin_sampling': Active learning with uncertainty using margin sampling (ALM)
        ''
        The default is 'ContrastiveAL'.
    
    Returns
    -------
    Data : np.array
        Design matrix, with each column represents:
        Response:-1 or 1;
        Status: 3 represents evaluated, ;
        Index:the order in evaluation, 2 represents skipped by monotonic information;
        4:4+p columns: x_1,\cdots,x_p, the p-dimensional coordinates of specific design points x.
    '''
    warnings.filterwarnings("ignore")

    slhd=importr('SLHD')
    Data = np.empty([0, 3+p])
    while sum(Data[:,0])==0 or sum(Data[:,0])==len(Data):
        init_1=slhd.maximinSLHD(t = 1, m =n_init, k = p)
        init=np.array(init_1.rx2('StandDesign'))
        Data = np.empty([0, 3+p])
        for x in range(0,len(init)):
            res=f_state(init[x],p,q)
            ls=[res,3,x]
            Data = np.row_stack((Data, np.hstack((ls, init[x]))))
        
    X_pool=np.random.uniform(low=[0] * p, high=[1] * p, size=(10000, p))
    y_true=[]
    for i in range(len(X_pool)):
        res=f_state(X_pool[i],p,q)
        y_true.append(res)
    y_true=np.array(y_true)
    X_init=Data[:,3:3+p]
    y_init=Data[:,0]
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
    clf = SklearnClassifier(
        SVC(probability=True),
        classes=np.unique(y_true)
    )
    clf.fit(X_init,y_init)
    if method=='ContrastiveAL':
        qs=ContrastiveAL()
    if method=='entropy':
        qs=UncertaintySampling(method='entropy')
    if method=='margin_sampling':
        qs=UncertaintySampling(method='margin_sampling')
    for c in range(n-n_init):
        query_idx = qs.query(X=X_pool, y=y, clf=clf)
        y[query_idx] = y_true[query_idx]
        clf.fit(np.vstack((X_init,X_pool)),np.concatenate((y_init,y)))
    lbld_idx=labeled_indices(y)
    
    X_idx = X_pool[lbld_idx, :] 
    numbers = np.arange(n_init,n).reshape(-1, 1) 
    status=np.array([3]*(n-n_init)).reshape(-1,1)
    result = np.hstack((y[lbld_idx].reshape(-1,1),status, numbers, X_idx))
    Data = np.row_stack((Data, result))

    return Data

###########Basis Functions for Simulations#########
def trainset(design, p, q):
    '''
    Genreate 0 and 1 response for SVC classification

    Parameters
    ----------
    design : np.array
        nxp design matrix that ONLY contains design points x, with each columns to be p-dimensional coordinates.
    p : integer
        A positive integer represents dimension.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    Returns
    -------
    Result : list
        A list contains 0-1 responses with length n.
    '''
    Result = []
    a = [1 for i in range(0, p)]
    for x in design:
        if f_p(x, p) > q*f_p(a, p):
            Result.append(1)
        else:
            Result.append(0)
    return Result


def volume(data, Design, p, q):
    '''
    MC methods for computing V(U) for designs without response information, e.g., MC, AMC, and LHD.

    Parameters
    ----------
    data : np.array
        nxp matrix, usually the 10^6 independent test samples in [0,1]^p.
    Design : np.array
        nxp design matrix that ONLY contains design points x, with each columns to be p-dimensional coordinates.
    p : integer
        A positive integer represents dimension.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).

    Returns
    -------
    v : float
        The volume of uncertain area V(U) given design D.

    '''
    a = [1 for i in range(0, p)]
    Alist = []
    Blist = []
    for x in Design:
        if f_p(x, p) > q*f_p(a, p):
            Blist.append(x)
        else:
            Alist.append(x)
    negative = []
    positive = []
    randomsample = copy.deepcopy(data)
    for xk in randomsample:
        for yk in Alist:
            if all(x <= y for x, y in zip(xk, yk)):
                negative.append(xk)
                break
        for yk in Blist:
            if all(x >= y for x, y in zip(xk, yk)):
                positive.append(xk)
                break
    v = 1-(len(negative)+len(positive))/len(randomsample)
    return v

def volume_grid(Design, p, q, m):
    '''
    Exact V(U) for grid-based designs, e.g., AG, AI, and GG

    Parameters
    ----------
    Design : np.array
        nxp design matrix that ONLY contains design points x, with each columns to be p-dimensional coordinates.
    p : integer
        A positive integer represents dimension.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    m : integer
        The generation of a given design D.

    Returns
    -------
    v : float
        The exact volume of uncertain area V(U) given grid-based design D.

    '''
    a = [1 for i in range(0, p)]
    Alist = []
    Blist = []
    for x in Design:
        if f_p(x, p) > q*f_p(a, p):
            Blist.append(x)
        else:
            Alist.append(x)
    negative = []
    positive = []
    randomsample = generate_p_center_grid(p, m+1)
    for xk in randomsample:
        for yk in Alist:
            if all(x <= y for x, y in zip(xk, yk)):
                negative.append(xk)
                break
        for yk in Blist:
            if all(x >= y for x, y in zip(xk, yk)):
                positive.append(xk)
                break
    v = 1-(len(negative)+len(positive))/len(randomsample)
    return v


def Compare_trained(data, design, p, q, k):
    '''
    SVC classification given design and test set data.
    Tunning SVC using Hyperopt and 5-fold CV.
    Parameters
    ----------
    data : np.array
        nxp matrix, usually the 10^6 independent test samples in [0,1]^p.
    Design : np.array
        nxp design matrix that ONLY contains design points x, with each columns to be p-dimensional coordinates.
    p : integer
        A positive integer represents dimension.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).

    k : integer
        An indicator for p=2 classification problems. If k>0, for p=2, repeat the classification process 10 times and 
        use the best SVC as the true model. See more details below.

    Returns
    -------
    inter
        A postive integer indicates the number of misclassification among 10^6 test samples.

    '''
    def train_SVC(params):
        clf = SVC(kernel='rbf', **params)
        metric = cross_val_score(clf, x_train, y_train, cv=5).mean()

        return -metric
    FN_test_svc = 0
    FP_test_svc = 0

    y_train = trainset(design, p, q)
    x_train = design
    y_test = trainset(data, p, q)
    x_test = data
    space4svc = {
        'C': hp.uniform('C', 0, 1000),
        'gamma': hp.uniform('gamma', 0, 20),
    }
    if sum(y_train) < 5:  # All test data are predicted to be negative
        y_svc = [0]*len(y_test)
    elif sum(y_train) > (len(x_train) - 5):
        y_svc = [1]*len(y_test)
    else:
        ##############Training SVC###############
        if k > 0 and p == 2:
            ###################################################################################
            #When p=2, due to the lack of monotonicity constraints in SVC,
            #the classifier may wrongly determine the classification boundary,
            #e.g., misclassifying the decision boundary as an ellipse,
            #leads to violations of the monotonicity assumption and large classification error.
            #We solve this by replication and outlier detection in the results for all designs.
            #This may be solved by using a monotonic classifier.
            ###################################################################################
            error1 = 4000
            repeat = 0
            y_svc = [1]*len(y_test)
            while repeat < 10:
                print('repeat error', error1, repeat)
                best_svc = fmin(train_SVC, space4svc,
                                algo=tpe.suggest, max_evals=100)
                svc_best = SVC(kernel='rbf', **best_svc)
                svc_best.fit(x_train, y_train)
                y_svc1 = svc_best.predict(x_test)
                confmat_svc1 = confusion_matrix(y_true=y_test, y_pred=y_svc1)
                error = confmat_svc1[0, 1]+confmat_svc1[1, 0]
                if error < error1:
                    error1 = error
                    y_svc = y_svc1
                repeat = repeat+1
                if error1 <= 200:
                    break
        else:
            best_svc = fmin(train_SVC, space4svc,
                            algo=tpe.suggest, max_evals=100)
            svc_best = SVC(kernel='rbf', **best_svc)
            svc_best.fit(x_train, y_train)
            y_svc = svc_best.predict(x_test)
    confmat_svc = confusion_matrix(y_true=y_test, y_pred=y_svc)
    FN_test_svc = FN_test_svc+confmat_svc[0, 1]
    FP_test_svc = FP_test_svc+confmat_svc[1, 0]
    SVC_error = FN_test_svc+FP_test_svc
    return SVC_error





def percentage(p, q):
    '''
    Compute the percentage of V(A) for a given f_state() through MC method.
    Used to determine \mu in Section 4.
    
    Parameters
    ----------
    p : integer
        A positive integer represents dimension.
    q : float
        A tunning parameter in [0,1] used for f_state function, define the propotion of negative repsonse.
        A higher value refers to larger V(A). If f_state does not depend on q, any value within the specified range is acceptable.
        Remark that q is not equal to V(A).
    Returns
    -------
    vn : float
        Percentage of V(A).
    vp : float
        Percentage of V(B).

    '''
    data = np.random.uniform(low=[0] * p, high=[1] * p, size=(100000, p))
    result = trainset(data, p, q)
    positive = np.sum(result)
    vp = positive/len(data)
    vn = 1-vp
    return vn, vp

######################Simulations in Section 4.1#############################################

t=100
##########################################################################################
#Simulation for MC, OLH, MED, ALE, PALC, AMC, CAL, GG, GI, AG, and AI
#Sample size n and \mu for different p, with q \in [ql,qh] to ensure V(A)\approx 10\%-90\%
#Since AG is deterministic, for each \mu, we only generate AG once, the same to grid designs.
#n1:sample size
#t:repeat times
#p:dimension, from 2-6.
#May needs long time for large n and p.
#output: Results, sum of errors and V(U) with t replications
#For active learning methods (ALE, CAL), default choice is ALE.
##########################################################################################
'''
for p in range(2,7):
    if p==2:
        n1=[20,30,40,60,80,100]
        ql=0.4
        qh=0.91
    if p==3:
        n1=[30,60,100,150,200,250]
        ql=0.45
        qh=0.87
    if p==4:
        n1=[40,80,120,200,280,360]
        ql=0.48
        qh=0.84
    if p==5:
        n1=[50,100,150,240,330,420]
        ql=0.5
        qh=0.83   
    if p==6:
        n1=[60,120,180,300,420,540]
        ql=0.52
        qh=0.82
    
    mc_error = pd.DataFrame(columns=['mc', 'vmc'])
    amc_error=pd.DataFrame(columns=['amc','vamc'])
    ag_error=pd.DataFrame(columns=['ag','vag'])
    ai_error=pd.DataFrame(columns=['ai','vai'])
    gg_error=pd.DataFrame(columns=['gg','vgg'])
    gi_error=pd.DataFrame(columns=['gi','vgi'])
    med_error = pd.DataFrame(columns=['med', 'vmed'])
    olhs_error = pd.DataFrame(columns=['olhs', 'volhs'])
    palc_error = pd.DataFrame(columns=['palc', 'vpalc'])
    al_error= pd.DataFrame(columns=['al', 'val'])
    for n in range(0,len(n1)):
        ag_error.loc[n]=[0,0]
        ai_error.loc[n]=[0,0]
        gg_error.loc[n]=[0,0]
        gi_error.loc[n]=[0,0]
        amc_error.loc[n]=[0,0]
        mc_error.loc[n]=[0,0]
        med_error.loc[n]=[0,0]
        olhs_error.loc[n]=[0,0]
        palc_error.loc[n]=[0,0]
        al_error.loc[n]=[0,0]
    for i in range(0, t):
        print('p=', p)
        print('Running %d times' % i)
        q = np.random.uniform(low=ql, high=qh)
        print('q=',q)
        max_n=max(n1)
        print('Running AG')
        agmax, agmaxg = AG(p, max_n, q)
        agall = agmax[agmax[:, 1] == 3]
        print('Running AI')
        aimax, aimaxg = AI(p, max_n, q)
        aiall = aimax[aimax[:, 1] == 3]
        print('Running GI')
        gimax,gimaxg=GI(p,max_n,q)
        giall=gimax[gimax[:, 1] == 3]
        print('Running GG')
        ggmax,ggmaxg=GG(p,max_n,q)
        ggall=ggmax[ggmax[:, 1] == 3]
        print('Running AMC')
        amc_design=AMC(p,max_n,q)
        print('Running MED')
        med_design = Med(max_n-10*p,p,10*p,q)
        print('Running PALC')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            palc_design=palc(max_n-10*p,p,q,10*p)
        print('Runing AL methods')
        al_design=ActiveLearningSVM(max_n,p,q,10*p,method='entropy')
        for n in range(0,len(n1)):
            print('try %d times' % n)
            k=n1[n]
            data = np.random.uniform(low=[0] * p, high=[1] * p, size=(100000, p))
            data1=np.random.uniform(low=[0] * p, high=[1] * p, size=(100000, p))
            
            MC = np.random.uniform(low=[0] * p, high=[1] * p, size=(k, p))
            vmc = volume(data1, MC, p, q)
            SVC_mc = Compare_trained(data, MC, p, q, n)
            mc_error.loc[n] = mc_error.loc[n] + [SVC_mc, vmc]
            
            ag = agall[0:k, :]
            agmaxg = int(max(ag[:, 2]))
            Edesign = ag[:, 3:]
            vag = volume_grid(Edesign, p, q, agmaxg)
            SVC_ag = Compare_trained(data, Edesign, p, q, n)
            ag_error.loc[n] = ag_error.loc[n] + [SVC_ag, vag]
            
            ai = aiall[0:k, :]
            aimaxg = int(max(ai[:, 2]))
            Edesign = ai[:, 3:]
            vai = volume_grid(Edesign, p, q, aimaxg)
            SVC_ai= Compare_trained(data, Edesign, p, q, n)
            ai_error.loc[n] = ai_error.loc[n] + [SVC_ai, vai]


            gi = giall[0:k, :]
            Edesign = gi[:, 3:]
            gimaxg = int(max(gi[:, 2]))
            vgi = volume_grid(Edesign, p, q, gimaxg)
            SVC_gi = Compare_trained(data, Edesign, p, q, n)
            gi_error.loc[n] = gi_error.loc[n] + [SVC_gi, vgi]
            
            gg= ggall[0:k, :]
            ggmaxg = int(max(gg[:, 2]))
            Edesign = gg[:, 3:]
            vgg = volume_grid(Edesign, p, q, ggmaxg)
            SVC_gg= Compare_trained(data, Edesign, p, q, n)
            gg_error.loc[n] = gg_error.loc[n] + [SVC_gg, vgg]
            
            amc=amc_design[0:k,3:3+p]
            vamc=volume(data1,amc,p,q)
            SVC_amc=Compare_trained(data,amc,p,q,n)
            amc_error.loc[n]=amc_error.loc[n]+[SVC_amc,vamc]
            
            med=med_design[0:k,3:3+p]
            vmed=volume(data1,med,p,q)
            SVC_med=Compare_trained(data,med,p,q,n)
            med_error.loc[n]=med_error.loc[n]+[SVC_med,vmed]
            
            alc=palc_design[0:k,3:3+p]
            vpalc=volume(data1,alc,p,q)
            SVC_palc=Compare_trained(data,alc,p,q,n)
            palc_error.loc[n]=palc_error.loc[n]+[SVC_palc,vpalc]
            
            init_1=slhd.maximinSLHD(t = 1, m =k, k = p)
            olhs_design=np.array(init_1.rx2('StandDesign'))
            volhs=volume(data1,olhs_design,p,q)
            SVC_olhs=Compare_trained(data,olhs_design,p,q,n)
            olhs_error.loc[n]=olhs_error.loc[n]+[SVC_olhs,volhs]
            
            
            al=al_design[0:k,3:3+p]
            val=volume(data1,al,p,q)
            SVC_al=Compare_trained(data,al,p,q,n)
            al_error.loc[n]=al_error.loc[n]+[SVC_al,val]
            
        result1=pd.concat([mc_error,olhs_error,med_error,palc_error,amc_error,al_error,gg_error,gi_error,ag_error,ai_error],axis=1)
        print(result1)
        #result1.to_csv('...\\%dresult%d%0.4f.csv'%(i,p,q))

##############
#Simulation for SG and SI
#Generation m and \mu for different p, with q \in [ql,qh] to ensure V(A)\approx 10\%-90\%
#Remark that for p>4, we do not generate SG and SI due to the sample size.
##############  
for p in range(2,5):
    if p==2:
        m=[5,6,7,8,9,10]
        ql=0.4
        qh=0.91
    if p==3:
        m=[3,4,5,6]
        ql=0.45
        qh=0.87
    if p==4:
        m=[2,3,4]
        ql=0.48
        qh=0.84

    sg_error=pd.DataFrame(columns=['sg','vsg'])
    si_error=pd.DataFrame(columns=['si','vsi'])
    for n in range(0,len(m)):
        sg_error.loc[n]=[0,0]
        si_error.loc[n]=[0,0]
    for i in range(0, t):
        print('p=', p)
        print('Running %d times' % i)
        q = np.random.uniform(low=ql, high=qh)
        print('q=',q)
        for n in range(0,len(m)):
            print('try %d times' % n)
            k=m[n]
            data = np.random.uniform(low=[0] * p, high=[1] * p, size=(100000, p))
            data1=np.random.uniform(low=[0] * p, high=[1] * p, size=(100000, p))
            sg=SG(p,k-1)
            si=SI(p,k+1)
            
            vsg=volume(data1,sg,p,q)
            vsi=volume(data1,si,p,q)
            SVC_sg=Compare_trained(data,sg,p,q,n)
            SVC_si=Compare_trained(data,si,p,q,n)
            sg_error.loc[n]=sg_error.loc[n]+[SVC_sg,vsg]
            si_error.loc[n]=si_error.loc[n]+[SVC_si,vsi]
        result1=pd.concat([sg_error,si_error],axis=1)
        #result1.to_csv('...\\%dresult%d%0.4f.csv'%(i,p,q))

'''

##############
#Simulation for extreme cases except for SG
#sample size n and \mu for different p, with q  to ensure V(A)=0.5
##############  

'''
for p in (2,4):
    if p == 2:
        n1 = [ 30, 40,50, 60, 80, 100]
        q=0.3
    if p==4:
        n1=[40,80,120,200,280,360]
        q=0.42
    if p==6:
        n1=[60,120,180,300,420,540]
        q=0.47
    positive5 = pd.DataFrame(columns=['mc','ag','ai','gg','amc','ale'])
    for n in range(0,len(n1)):
        positive5.loc[n] = [0,0,0,0,0,0]
    print('Running AI')
    aimax, aimaxg = AI(p, n1[5], q)
    aiall = aimax[aimax[:, 1] == 3]
    print('Running AG')
    agmax, agmaxg = AG(p, n1[5], q)
    agall = agmax[agmax[:, 1] == 3]
    print('Running GG')
    ggmax,ggmaxg=GG(p,n1[5],q)
    ggall = ggmax[ggmax[:, 1] == 3]
    for i in range(0, t):
        print('p=', p)
        print('Running %d times' % i)
        print('q=', q)
        max_n=max(n1)
        print('Running AMC')
        amc_design=AMC(p,max_n,q)
        print('Runing ALE')
        al_design=ActiveLearningSVM(max_n,p,q,10*p,method='entropy')
        for n in range(0,len(n1)):
            print('try %d times' % n)
            k=n1[n]
            amc=amc_design[0:k,3:3+p]
            reamc=trainset(amc,p,q)
            vamc=len(reamc)-np.sum(reamc)

            mc = np.random.uniform(low=[0] * p, high=[1] * p, size=(k, p))
            reran=trainset(mc,p,q)
            vran=len(reran)-np.sum(reran)
            
            al=al_design[0:k,3:3+p]
            real=trainset(al,p,q)
            val=len(real)-np.sum(real)
            
            ai = aiall[0:k, :]
            ag = agall[0:k, :]
            Bdesign =ai[:, 3:]
            Edesign =ag[:, 3:]
            gg=ggall[0:k,:]
            Ddesign=gg[:,3:]
            regg=trainset(Ddesign,p,q)
            reai=trainset(Bdesign,p,q)
            reag=trainset(Edesign,p,q)
            vai=len(reai)-np.sum(reai)
            vag=len(reag)-np.sum(reag)
            vgg=len(regg)-np.sum(regg)
            positive5.loc[n] = positive5.loc[n]+[ vran,  vag, vai,vgg, vamc, val]
        #positive5.to_csv('...\\p=%d\\negative.csv' % p)
'''
'''
slhd=importr('SLHD')
for p in (2,4):
    if p == 2:
        n1 = [ 30, 40,50, 60, 80, 100]
        q=0.3
    if p==4:
        n1=[40,80,120,200,280,360]
        q=0.42
    
    al_error= pd.DataFrame(columns=['al'])
    for n in range(0,len(n1)):
        al_error.loc[n]=[0]
    for i in range(0,t):
        print('p=', p)
        print('Running %d times' % i)

        max_n=max(n1)
        al_design=ActiveLearningSVM(max_n,p,q,10*p,method='entropy')

        for n in range(0,len(n1)):
            print('try %d times' % n)
            k=n1[n]

            al=al_design[0:k,3:3+p]
            real=trainset(al,p,q)
            val=len(real)-np.sum(real)
            print(val)
            
            al_error.loc[n]=al_error.loc[n]+[val]
            
            
        result1=pd.concat([al_error],axis=1)
        print(result1)
        result1.to_csv('extreme/%dresult%d.csv'%(i,p))
'''       
##############
#Simulation for extreme cases for SG
#Generation m and \mu for different p, with q to ensure V(A)=5%
##############  

'''
for p in (2,4):
    if p == 2:
        m=[5,6,7,8,9,10]
        q=0.3
    if p==4:
        m=[2,3,4]
        q=0.42
    positive5 = pd.DataFrame(columns=['sg'])

    for n in range(0,len(m)):
        positive5.loc[n] = [0]
    for i in range(0, t):
        print('p=', p)
        print('Running %d times' % i)
        print('q=', q)
        for n in range(0,len(m)):
            print('try %d times' % n)
            k1=m[n]
            sg=SG(p,k1-1)
            print(len(sg))
            resg=trainset(sg,p,q)
            vsg=len(resg)-np.sum(resg)
            positive5.loc[n] = positive5.loc[n]+[vsg]
    #positive5.to_csv('...\\p=%d\\negative.csv' % p)
'''