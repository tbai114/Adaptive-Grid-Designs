# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from hyperopt import fmin, tpe, hp
from itertools import product
import numpy as np
from mlxtend.plotting import plot_decision_regions




#Basis Functions
def to_percent(temp, position):
  return '%0.01f'%(100*temp) + '%'

def generate_p_center_grid(p,l):##generate center grid for generation l
    init=[]
    for i in range(0,2**(l-1)):
        q=2*i+1
        init.append(q/2**l)
    result = list(product(init, repeat=p))
    return result



def FindIndex(searchX,AGridData,p):
    number=np.arange(1,len(AGridData)+1,1)
    LookData=np.column_stack((AGridData,number))
    del_index=[]
    for jj in range(0,p):
        for k in range(0,len(LookData)):
            if LookData[k,3+jj]!=searchX[jj]:
                del_index.append(k)
    LookData=np.delete(LookData,del_index,0)
    if len(LookData)>0:
        return LookData[0,3+p]
    if len(LookData)==0:
        return 0


def FindResponse(searchX,Grid,p):# Find the response of node not contained in the data set.
    AGrid=copy.deepcopy(Grid)
    searchP1=copy.deepcopy(searchX)
    AGrid=AGrid[AGrid[:,1] ==3]
    resp=0
    for grids in AGrid:
        if grids[0]==1:
            data=grids[3:3+p]
            if (np.array(searchP1-data)>=0).all()==True:
                resp=1
                break;
        if grids[0]==-1:
            data=grids[3:3+p]
            if (np.array(searchP1-data)<=0).all()==True:
                resp=-1
                break;
    return resp



def generate_p_noboundary_grid(p,l):
    init=[]
    for i in range(1,2**l):
        init.append(i/2**l)
    result = list(product(init, repeat=p))
    return result
def generate_p_grid(p,l):
    init=[]
    for i in range(0,2**l+1):
        init.append(i/2**l)
    result = list(product(init, repeat=p))
    return result






def volume_grid_case(Design,p,q,m):#Compute V(U) for GG, GI, AG, and AI
    Alist=[]
    Blist=[]
    for x in Design:
        if x[0]>0:
            Blist.append(x[3:3+p])
        if x[0]<0:
            Alist.append(x[3:3+p])
    negative=[]
    positive=[]
    randomsample=generate_p_center_grid(p, m+1)
    for xk in randomsample:
        for yk in Alist:
            if all(x <= y for x, y in zip(xk,yk)):
                negative.append(xk)
                break
        for yk in Blist:    
            if all(x>=y for x,y in zip(xk,yk)):
                positive.append(xk)
                break
    v=1-(len(negative)+len(positive))/len(randomsample)
    
    return v


def volume(Design,p,q,m):#Compute V(U) for SG and SI
    Alist=[]
    Blist=[]
    for x in Design:
        if f_state(x,3,1)>0:
            Blist.append(x)
        if f_state(x,3,1)<0:
            Alist.append(x)
    negative=[]
    positive=[]
    randomsample=generate_p_center_grid(p, m+1)
    for xk in randomsample:
        for yk in Alist:
            if all(x <= y for x, y in zip(xk,yk)):
                negative.append(xk)
                break
        for yk in Blist:    
            if all(x>=y for x,y in zip(xk,yk)):
                positive.append(xk)
                break
    v=1-(len(negative)+len(positive))/len(randomsample)
    
    return v

#########Grid-based designs##########
def SG(p,m):
    init=[]
    for i in range(0,m+1):
        init.append(i/m)
    result=list(product(init,repeat=p))
    return result
def GG(p,n,q):
    Data=np.empty([0,3+p])
    l=0
    while np.sum(Data[:,1]==3)<n:
        if l>0:
            grid=list(set(generate_p_grid(p,l))-set(generate_p_grid(p,l-1)))
        else:
            grid=generate_p_grid(p,l)
        K=[]
        for g in grid:
            resp=FindResponse(g,Data,p)
            if resp!=0:
                ls=[resp,2,l]
                Data =np.row_stack((Data,np.hstack((ls,g))))
                K.append(g)
        grid1=list(set(grid)-set(K))
        length=len(grid1)
        for i in range(0,length):
            IndexNext=i
            g=grid1[IndexNext]
            res=f_state(g,p,q)
            ls=[res,3,l]
            Data =np.row_stack((Data,np.hstack((ls,g))))
            if np.sum(Data[:,1]==3)==n:
                break
  
        if np.sum(Data[:,1]==3)==n:
            break
        l=l+1
    return Data,l

def AG(p,n1,q):
    Data=np.empty([0,3+p])
    maxg=0
    while np.sum(Data[:,1]==3)<n1:
        FData=copy.deepcopy(Data)
        grid=[]
        grid1=generate_p_grid(p, maxg)
        if maxg!=0:
            grid2=generate_p_grid(p, maxg-1)
            for x in grid1:
                if x not in grid2:
                    grid.append(x)
        else:
            grid=grid1

        FData=FData[FData[:,1]==3]
        Alist= FData[FData[:,0]==-1][:,3:3+p]
        Blist=FData[FData[:,0]==1][:,3:3+p]
        U=[]
        for xk in grid:
            q1=0
            q2=0
            for yk in Alist:
                if all(x <= y for x, y in zip(xk,yk)):
                    q1=1
                    break
            if q1==0:
                for yk in Blist:
                    if all(x>=y for x,y in zip(xk,yk)):
                        q2=1
                        break
            if q1==0 and q2==0:
                U.append(xk)
        ul=len(U)
        Umatrix=np.eye(ul) * 2
        Ordermatrix = np.zeros((ul, 3))#####Save the information of Ax, Bx, and status: 1: uncertain
        Ordermatrix[:, 2] = 1
        for i in range(0,ul):
            xi=U[i]
            for j in range(0,ul):
                xj=U[j]
                if j!=i:
                    if all(x <= y for x, y in zip(xi,xj)):
                        Umatrix[i,j]=-1
                    elif all(x>=y for x,y in zip(xi,xj)):
                        Umatrix[i,j]=1
                    else:
                        Umatrix[i,j]=0
        for i in range(0,ul):
            Ordermatrix[i,0]=np.sum(Umatrix[i] == 1)
            Ordermatrix[i,1]=np.sum(Umatrix[i] == -1)
        while np.sum(Ordermatrix[:,2]==1)>0:
            cmin=[]
            cmax=[]
            for i in range(0,ul):
                cmin.append(min(Ordermatrix[i,0],Ordermatrix[i,1]))
                cmax.append(max(Ordermatrix[i,0],Ordermatrix[i,1]))
            max_value = max(cmin)
        
            max_indices1 = [index for index, value in enumerate(cmin) if value == max_value]
            max_indices=[x for  x in max_indices1 if Ordermatrix[x,2]==1]
            if len(max_indices)>1:
                maxnumber=0
                IndexNext=max_indices[0]
                for i in range(0,len(max_indices)):
                    maxnum=cmax[max_indices[i]]
                    if maxnum>=maxnumber:
                        maxnumber=maxnum
                        IndexNext=max_indices[i]
            else:
                IndexNext=max_indices[0]

            g=U[IndexNext]
            gen=maxg
            res=f_state(g,p,q)
            ls=[res,3,gen]
            Data =np.row_stack((Data,np.hstack((ls,g))))
            ##Update Ordermatrix according to Umatrix##

            Umatrix[IndexNext,IndexNext]=res
            #Update Ak and Bk
            for i in range(0,ul):
                if Ordermatrix[i,2]==1:
                    if i != IndexNext:
                        if res==-1:
                            if Umatrix[i,IndexNext]==-1:
                                Umatrix[i,i]=-1
                                Ordermatrix[i,:]=0
                            if Umatrix[i,IndexNext]==1:
                                Ordermatrix[i,0]=Ordermatrix[i,0]-Ordermatrix[IndexNext,0]-1
                        if res==1:
                            if Umatrix[i,IndexNext]==1:
                                Umatrix[i,i]=1
                                Ordermatrix[i,:]=0
                            if Umatrix[i,IndexNext]==-1:
                                Ordermatrix[i,1]=Ordermatrix[i,1]-Ordermatrix[IndexNext,1]-1
            for i in range(0,ul):
                if Umatrix[i,IndexNext]==0:
                    if Ordermatrix[i,2]==1:
                        for j in range(0,ul):
                            if Ordermatrix[j,2]==0:
                                if res==-1:
                                    if Umatrix[i,j]==1:
                                        Ordermatrix[i,0]=Ordermatrix[i,0]-1
                                else:
                                    if Umatrix[i,j]==-1:
                                        Ordermatrix[i,1]=Ordermatrix[i,1]-1
            Ordermatrix[IndexNext,:]=0
            Ordermatrix[:, 2][Ordermatrix[:, 2] == 0] = 2

            if np.sum(Data[:,1]==3)==n1:
                break
        
        if np.sum(Data[:,1]==3)==n1:
            break
        maxg=maxg+1

    return Data,maxg
######Inner-grid-based designs#######
def SI(p,m):
    init=[]
    for i in range(1,m):
        init.append(i/m)
    result = list(product(init, repeat=p))
    return result
def GI(p,n,q):
    Data=np.empty([0,3+p])
    l=1
    while np.sum(Data[:,1]==3)<n:
        grid=list(set(generate_p_noboundary_grid(p,l))-set(generate_p_noboundary_grid(p,l-1)))
        K=[]
        for g in grid:
            resp=FindResponse(g,Data,p)
            if resp!=0:
                ls=[resp,2,l]
                Data =np.row_stack((Data,np.hstack((ls,g))))
                K.append(g)
        grid1=list(set(grid)-set(K))
        length=len(grid1)
        for i in range(0,length):
            IndexNext=i
            g=grid1[IndexNext]
            res=f_state(g,p,q)
            ls=[res,3,l]
            Data =np.row_stack((Data,np.hstack((ls,g))))
            if np.sum(Data[:,1]==3)==n:
                break
        if np.sum(Data[:,1]==3)==n:
            break
        l=l+1
    return Data,l
def AI(p,n1,q):
    Data=np.empty([0,3+p])
    maxg=1
    while np.sum(Data[:,1]==3)<n1:
        FData=copy.deepcopy(Data)
        grid=[]
        grid1=generate_p_noboundary_grid(p, maxg)
        grid2=generate_p_noboundary_grid(p, maxg-1)
        for x in grid1:
            if x not in grid2:
                grid.append(x)

        FData=FData[FData[:,1]==3]
        Alist= FData[FData[:,0]==-1][:,3:3+p]
        Blist=FData[FData[:,0]==1][:,3:3+p]
        U=[]
        for xk in grid:
            q1=0
            q2=0
            for yk in Alist:
                if all(x <= y for x, y in zip(xk,yk)):
                    q1=1
                    break
            if q1==0:
                for yk in Blist:
                    if all(x>=y for x,y in zip(xk,yk)):
                        q2=1
                        break
            if q1==0 and q2==0:
                U.append(xk)
        ul=len(U)
        Umatrix=np.eye(ul) * 2
        Ordermatrix = np.zeros((ul, 3))
        Ordermatrix[:, 2] = 1
        for i in range(0,ul):
            xi=U[i]
            for j in range(0,ul):
                xj=U[j]
                if j!=i:
                    if all(x <= y for x, y in zip(xi,xj)):
                        Umatrix[i,j]=-1
                    elif all(x>=y for x,y in zip(xi,xj)):
                        Umatrix[i,j]=1
                    else:
                        Umatrix[i,j]=0
        for i in range(0,ul):
            Ordermatrix[i,0]=np.sum(Umatrix[i] == 1)
            Ordermatrix[i,1]=np.sum(Umatrix[i] == -1)
        while np.sum(Ordermatrix[:,2]==1)>0:
            cmin=[]
            cmax=[]
            for i in range(0,ul):
                cmin.append(min(Ordermatrix[i,0],Ordermatrix[i,1]))
                cmax.append(max(Ordermatrix[i,0],Ordermatrix[i,1]))
            max_value = max(cmin)
            max_indices1 = [index for index, value in enumerate(cmin) if value == max_value]
            max_indices=[x for  x in max_indices1 if Ordermatrix[x,2]==1]
            if len(max_indices)>1:
                maxnumber=0
                IndexNext=max_indices[0]
                for i in range(0,len(max_indices)):
                    maxnum=cmax[max_indices[i]]
                    if maxnum>=maxnumber:
                        maxnumber=maxnum
                        IndexNext=max_indices[i]
            else:
                IndexNext=max_indices[0]

            g=U[IndexNext]
            gen=maxg
            res=f_state(g,p,q)
            ls=[res,3,gen]
            Data =np.row_stack((Data,np.hstack((ls,g))))
            Umatrix[IndexNext,IndexNext]=res
            for i in range(0,ul):
                if Ordermatrix[i,2]==1:
                    if i != IndexNext:
                        if res==-1:
                            if Umatrix[i,IndexNext]==-1:
                                Umatrix[i,i]=-1
                                Ordermatrix[i,:]=0
                            if Umatrix[i,IndexNext]==1:
                                Ordermatrix[i,0]=Ordermatrix[i,0]-Ordermatrix[IndexNext,0]-1
                        if res==1:
                            if Umatrix[i,IndexNext]==1:
                                Umatrix[i,i]=1
                                Ordermatrix[i,:]=0
                            if Umatrix[i,IndexNext]==-1:
                                Ordermatrix[i,1]=Ordermatrix[i,1]-Ordermatrix[IndexNext,1]-1
            for i in range(0,ul):
                if Umatrix[i,IndexNext]==0:
                    if Ordermatrix[i,2]==1:
                        for j in range(0,ul):
                            if Ordermatrix[j,2]==0:
                                if res==-1:
                                    if Umatrix[i,j]==1:
                                        Ordermatrix[i,0]=Ordermatrix[i,0]-1
                                else:
                                    if Umatrix[i,j]==-1:
                                        Ordermatrix[i,1]=Ordermatrix[i,1]-1
            Ordermatrix[IndexNext,:]=0
            Ordermatrix[:, 2][Ordermatrix[:, 2] == 0] = 2
            if np.sum(Data[:,1]==3)==n1:
                break
        if np.sum(Data[:,1]==3)==n1:
            break
        maxg=maxg+1

    return Data,maxg




#####ice-breaking dynamics######

def f_state(x,p,q):
    result = pd.read_csv('fgd.csv', index_col=0)
    f=0
    for i in range(0,len(result)):
       if result['Result'].iloc[i]==-1:
           if result.iloc[i,0]>=x[0]:
               if result.iloc[i,1]>=x[1]:
                   if result.iloc[i,2]>=x[2]:
                       f=-1
                       break;
       if result['Result'].iloc[i]==1:
           if result.iloc[i,0]<=x[0]:
               if result.iloc[i,1]<=x[1]:
                   if result.iloc[i,2]<=x[2]:
                       f=1
                       break;
    return f



'''
nag=[5,7,9,11,13,15,17,19,21,23,25,27,29]
ngi=[1,3,5,7,9,11,13,15,17,19,20]
#GI
for i in range(0,11):
    ag,maxg=GI(3,ngi[i],1)
    ag=ag[ag[:,1]==3]
    v=volume_grid_case(ag, 3, 1, maxg)
    print(v)
#GG
for i in range(2,16):
    ag,maxg=GG(3,4*i,1)
    ag=ag[ag[:,1]==3]
    v=volume_grid_case(ag, 3, 1, maxg)
    print(v)
#AG
for i in range(0,13):
    ag,maxg=AG(3,nag[i],1)
    ag=ag[ag[:,1]==3]
    v=volume_grid_case(ag, 3, 1, maxg)
    print(v)
#AI
for i in range(1,9):
    ag,maxg=AI(3,i,1)
    ag=ag[ag[:,1]==3]
    v=volume_grid_case(ag, 3, 1, maxg)
    print(v)

#SI
for i in (1,2):
    ag=SI(3,2**i)
    v=volume(ag,3,1,i)
    print(v)
#SG
for i in (0,1,2):
    ag=SG(3,2**i)
    v=volume(ag,3,1,i)
    print(v)

'''

#Plot for Fig.13
grid=generate_p_grid(3, 2)
ag,maxg=AG(3,29,1)
grid1=ag[:,3:6]
skip=[]
for x in grid:
    if np.any(np.all(grid1 == x, axis=1)):
        continue
    else:
        skip.append(x)

for i in range(0,len(skip)): 
    x=skip[i]
    res=f_state(x,2,1)
    ag=np.row_stack((ag,[res,2,4,x[0],x[1],x[2]]))

train=ag
xe=0
#xe represents X_e,  value in (0,0.25,0.5,0.75,1), represents (5,4,3,2,1) in original space respectively
train=train[train[:,5]==xe]
x_train=train[:,3:5]
y_train=train[:,0]
def train_SVC(params):
    clf=SVC(kernel='linear',**params)
    metric = cross_val_score(clf,x_train,y_train,cv=5).mean()

    return -metric
FN_test_svc=0
FP_test_svc=0

space4svc = {
    'C': hp.uniform('C', 0, 1000),
    'gamma': hp.uniform('gamma', 0, 20),
}
   
best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
svc_best=SVC(kernel='rbf',**best_svc,probability=True)
svc_best.fit(x_train,y_train)
y_train=y_train.astype(np.int_)

def plot_2dexample(n,q):
    x=np.linspace(0,1)
    y=np.linspace(0,1)
    X,Y=np.meshgrid(x,y)
    plt.figure(figsize=(15,15))

    contourf_kwargs = {'alpha': 0}       
    contour_kwargs={'colors': 'purple',        
                           'linestyles': '--',      
                               'linewidths': 5}
    scatter_kwargs = {
        's': 0,             
        'alpha': 0,         
        'marker': 'o',       
        'edgecolor': 'k',     
        'linewidths': 1       
    }
    plot_decision_regions(x_train, y_train ,clf=svc_best,scatter_kwargs=scatter_kwargs,legend=0,
                          hide_spines=False,
                 feature_index=[0,1],                           
                ax=None,
                  contour_kwargs=contour_kwargs,
                  contourf_kwargs=contourf_kwargs)

    matrix = x_train
    for i in range(0,len(matrix)):
        if train[i,2]==4:
            marker='x'
        else:
            marker='.'

        s=1200
        if train[i,0]==1:
            color='blue'
        else:
            color='red'
        
        plt.scatter(matrix[i,0],matrix[i,1],color=color,marker=marker,s=s)
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel(r'$x_{v}$',fontsize=45,fontproperties='Times New Roman')
    plt.ylabel(r'$x_{t}$',fontsize=45,fontproperties='Times New Roman')
    font = {'family': 'serif',
            'fontname': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size':15,
        }
    ax = plt.gca()
    x_ticks = np.linspace(0,1,9)
    x_ticklabels = np.round(x_ticks *35+5,1)  
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontdict=font)
    y_ticks = np.linspace(0,1,9)
    y_ticklabels = np.round(-(y_ticks * 10-15) ,1) 
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontdict=font)
    plt.xticks(fontsize=40,fontproperties='Times New Roman')
    plt.yticks(fontsize=40,fontproperties='Times New Roman')
    plt.savefig('E=5.png',dpi=300)
    plt.show()
plot_2dexample(29,0)







