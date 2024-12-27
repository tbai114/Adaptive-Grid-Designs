
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from hyperopt import fmin, tpe, hp
from itertools import product
import numpy as np
import random
from skactiveml.pool import UncertaintySampling,ContrastiveAL
from skactiveml.utils import MISSING_LABEL,labeled_indices
from skactiveml.classifier import SklearnClassifier
import warnings



###########Basis Functions############
def trainset(design,p,q):
    Result=[]
    for x in design:
        if f_state(x,p,q)==1:
            Result.append(1)
        else:
            Result.append(0)
    return Result

def to_percent(temp, position):
  return '%0.01f'%(100*temp) + '%'


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


def FindResponse(searchX,Grid,p):
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
########Metrics##########
def generate_center():
    init1=[]
    init2=[]
    for i in range(0,66):
        init1.append(round(0.05+0.1*i,8))
    for i in range(0,14):
        init2.append(round(-10.05+0.5*i,8))
    init11=[round(x/6.6,8) for x in init1]
    init22=[round((x+10.3)/7,8) for x in init2]
    result=list(product(init11, init22))
    return result

def volume_grid(Design):
    '''
    Compute exact V(U) for given design

    Parameters
    ----------
    Design : np.array
        nxp design matrix contains ONLY x coordinates.

    Returns
    -------
    v : float
        Exact volume of given design D in original space.

    '''
    centergrid=generate_center()
    Alist=[]
    Blist=[]
    negative=[]
    positive=[]
    for x in Design:
        if x[0]==1:
            Blist.append(x[3:5])
        if x[0]==-1:
            Alist.append(x[3:5])

    for xk in centergrid:
        for yk in Alist:
            if all(x <= y for x, y in zip(xk,yk)):
                negative.append(xk)
                break
        for yk in Blist:    
            if all(x>=y for x,y in zip(xk,yk)):
                positive.append(xk)
                break
    v=1-(len(negative)+len(positive))/len(centergrid)
    return v



########Grid-based designs#############
def SG(p,n,q):
    Data=np.empty([0,3+p])
    l=0
    Data1=Data[Data[:,0]!=0]
    while np.sum(Data1[:,1]==3)<n:
        if l>0:
            grid=list(set(generate_p_grid(p,l))-set(generate_p_grid(p,l-1)))
        else:
            grid=generate_p_grid(p,l)
        for g in grid:
            res=f_state(g,p,q)
            ls=[res,3,0]
            Data =np.row_stack((Data,np.hstack((ls,g))))
            Data1 =Data[Data[:,0]!=0]
            if np.sum(Data1[:,1]==3)==n:
                break
        if np.sum(Data1[:,1]==3)==n:
            break
        if l==7:
            break
        l=l+1
    return Data,l

def GG(p,n,q):
    Data=np.empty([0,3+p])
    l=0
    Data1=Data[Data[:,0]!=0]
    while np.sum(Data1[:,1]==3)<n:
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
            Data1=Data[Data[:,0]!=0]
            if np.sum(Data1[:,1]==3)==n:
                break
  
        if np.sum(Data1[:,1]==3)==n:
            break
        if l==7:
            break
        l=l+1
    return Data,l


def AG(p,n1,q):
    Data=np.empty([0,3+p])
    maxg=0
    Responsedata=Data[Data[:,0]!=0]
    while np.sum( Responsedata[:,1]==3)<n1:
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
            Responsedata=Data[Data[:,0]!=0]
            if np.sum(Responsedata[:,1]==3)==n1:
                break
        if maxg==7:
            break
        if np.sum(Responsedata[:,1]==3)==n1:
            break
        maxg=maxg+1

    return Data,maxg
#########Inner-grid-based designs#######
def SI(p,n1,q):
    Data=np.empty([0,3+p])
    maxg=1
    Responsedata=Data[Data[:,0]!=0]

    while np.sum(Responsedata[:,1]==3)<n1:
        grid=[]
        if maxg<8:
            grid1=generate_p_noboundary_grid(p, maxg)
        else:
            init=[]
            for i in range(1,2**6):
                init.append(i/2**6)
            init1=[1/256,255/256]
            grid1 = list(product(init1, init))
        grid2=generate_p_noboundary_grid(p, maxg-1)
        for x in grid1:
            if x not in grid2:
                grid.append(x)
        for g in grid:
            res=f_state(g,p,q)
            ls=[res,3,0]
            Data =np.row_stack((Data,np.hstack((ls,g))))
            Responsedata=Data[Data[:,0]!=0]
            if np.sum(Responsedata[:,1]==3)==n1:
                break
        if maxg==8:
            break
        if np.sum(Responsedata[:,1]==3)==n1:
            break
        maxg=maxg+1
    return Data,maxg
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
    Responsedata=Data[Data[:,0]!=0]

    while np.sum(Responsedata[:,1]==3)<n1:
        FData=copy.deepcopy(Data)
        grid=[]
        if maxg<8:
            grid1=generate_p_noboundary_grid(p, maxg)
        else:
            init=[]
            for i in range(1,2**6):
                init.append(i/2**6)
            init1=[1/256,255/256]
            grid1 = list(product(init1, init))
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
            Responsedata=Data[Data[:,0]!=0]
            if np.sum(Responsedata[:,1]==3)==n1:
                break
        if maxg==8:
            break
        if np.sum(Responsedata[:,1]==3)==n1:
            break
        maxg=maxg+1

    return Data,maxg



########MC and AMC design sampling from given data set#########

def MC(n,data):
    Data=np.empty([0,5])
    index=list(data.index)
    for i in range(0,n):
        sampleindex = random.choice(index)
        sample=data.loc[sampleindex]
        index.remove(sampleindex)
        Data =np.row_stack((Data,[2*(sample['crash']-1/2),3,i,sample['eoff'],sample['acc']]))
    return Data

def AMC(p,n,data):
    Data=np.empty([0,3+p])
    i=0
    index=list(data.index)
    
    while(i<n):
        i=i+1
        if_u=0
        if len(index)==0:
            break
        sampleindex=random.choice(index)
        sample=data.loc[sampleindex]
        x_candidate=sample[['eoff','acc']]
        index.remove(sampleindex)
        while(if_u<1):
            resp=FindResponse(x_candidate,Data,p)
            if resp==0:
                if_u=2
                res=2*(sample['crash']-1/2)
                ls=[res,3,i]
                Data =np.row_stack((Data,np.hstack((ls,x_candidate))))
            else:
                res=2*(sample['crash']-1/2)
                ls=[res,2,i]
                Data =np.row_stack((Data,np.hstack((ls,x_candidate))))
                if len(index)==0:
                    break
                sampleindex=random.choice(index)
                sample=data.loc[sampleindex]
                x_candidate=sample[['eoff','acc']]
                index.remove(sampleindex)
    return Data
########ALE design sampling from given data set#########
def ActiveLearningSVM(n,data,p,method='ContrastiveAL'):
    warnings.filterwarnings("ignore")
    Data=np.empty([0,3+p])
    i=0
    index=list(data.index)
    while np.sum(Data[:,0]==1)<1 or np.sum(Data[:,0]==-1)<1:
        sampleindex=random.choice(index)
        sample=data.loc[sampleindex]
        x_candidate=sample[['eoff','acc']]
        index.remove(sampleindex)
        res=2*(sample['crash']-1/2)
        ls=[res,3,i]
        Data =np.row_stack((Data,np.hstack((ls,x_candidate))))
    n_init=len(Data)
    pool=data.loc[index]
    X_pool=np.array(pool[['eoff','acc']]    )
    y_true=np.array(2*(pool['crash']-1/2))
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

##############Simulation################
#Full data: crashData.csv
#Total of 44 cases, where 4 cases contains only 1 category, we skip these 4 cases.
#For 17th crash occation, please let k=17.
#For different designs, f_state is different.
########################################
'''
#MC
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,7)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),7)

result=pd.DataFrame(columns=['10svc','10v','20svc','20v','40svc','40v','60svc','60v','80svc','80v','100svc','100v','200svc','200v','300svc','300v','400svc','400v','500svc','500v','600svc','600v','800svc','800v','1005svc','1005v'])
for k in range(1,45):
    print('Running MC %d case'%k)
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=odata[odata['caseID']==k]
    data.index=range(0,len(data))
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000001:
                if abs(data['acc'][i]-x2)<0.000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f

    n=[10,20,40,60,80,100,200,300,400,500,600,800,1005]
    ag_error=[0]*26
    for t in range(0,100):
        ag1=MC(1005,data)
        ag1=ag1[ag1[:,1]==3]
        for j in range(0,13):
            x_test1=np.array(data[['eoff','acc']])
            y_test1=np.array(data['crash'])
            ag=ag1[0:n[j],:]
            x_ag=ag[:,3:5]
            y_ag=ag[:,0]
            y_ag=(y_ag+1)/2
            vag=volume_grid(ag)
            if j==12:
                ag_error2=0
                ag_error[2*j]=ag_error[2*j]+ag_error2
                ag_error[2*j+1]=ag_error[2*j+1]+vag
            else:
                Alist=ag[ag[:,0]==-1][:,3:5]
                Blist=ag[ag[:,0]==1][:,3:5]
                Uindex=[]
                for i in range(0,len(x_test1)):
                    xk=x_test1[i]
                    n1=0
                    n2=0
                    for yk in Alist:
                        if all(round(x,7)<=y for x, y in zip(xk,yk)):
                            n1=1
                            break
                    for yk in Blist:
                        if all(round(x,7)>=y for x,y in zip(xk,yk)):
                            n2=1
                            break
                    if n1==0 and n2==0:
                        Uindex.append(i)
                x_test=x_test1[Uindex]
                y_test=y_test1[Uindex]
                if len(x_test)==0:
                    ag_error2=0     
                    for m in range(j,13):
                        ag_error[2*m]=ag_error[2*m]+ag_error2
                        ag_error[2*m+1]=ag_error[2*m+1]+vag
                        break
                else:
        
                    if sum(y_ag)<5:
                        y_ag_pre=[0]*len(y_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    elif sum(y_ag)>len(y_ag)-5:
                        y_ag_pre=[1]*len(y_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    else:
                        def train_SVC(params):
                            clf=SVC(kernel='rbf',**params)
                            metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                            return -metric
                        space4svc = {
                            'C': hp.uniform('C', 0, 1000),
                            'gamma': hp.uniform('gamma', 0, 20),
                        }
                       
                        ag_error2=1000
                        best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                        svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                        svc_ag.fit(x_ag,y_ag)
                        y_ag_pre=svc_ag.predict(x_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    ag_error[2*j]=ag_error[2*j]+ag_error2
                    ag_error[2*j+1]=ag_error[2*j+1]+vag
        ag_error1=[x/100 for x in ag_error]
    result.loc[k-1]=ag_error1
   # result.to_csv('...//MCresult%d.csv'%k)

'''
'''
#AMC
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,7)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),7)
result=pd.DataFrame(columns=['10svc','10v','15svc','15v','20svc','20v','25svc','25v','30svc','30v','35svc','35v','40svc','40v','45svc','45v','50svc','50v','55svc','55v','60svc','60v','65svc','65v','70svc','70v','75svc','75v','80svc','80v','85svc','85v','90svc','90v','95svc','95v'])
for k in range(1,45):
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=odata[odata['caseID']==k]
    data.index=range(0,len(data))
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000001:
                if abs(data['acc'][i]-x2)<0.000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f

    n=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    ag_error=[0]*36
    for t in range(0,100):
        ag1=AMC(2,95,data)
        ag1=ag1[ag1[:,1]==3]
        print(len(ag1))
        for j in range(0,18):
            x_test1=np.array(data[['eoff','acc']])
            y_test1=np.array(data['crash'])
            if len(ag1)>=n[j]:
                ag=ag1[0:n[j],:]
                x_ag=ag[:,3:5]
                y_ag=ag[:,0]
                y_ag=(y_ag+1)/2
                Alist=ag[ag[:,0]==-1][:,3:5]
                Blist=ag[ag[:,0]==1][:,3:5]
                Uindex=[]
                for i in range(0,len(x_test1)):
                    xk=x_test1[i]
                    n1=0
                    n2=0
                    for yk in Alist:
                        if all(round(x,7)<=y for x, y in zip(xk,yk)):
                            n1=1
                            break
                    for yk in Blist:
                        if all(round(x,7)>=y for x,y in zip(xk,yk)):
                            n2=1
                            break
                    if n1==0 and n2==0:
                        Uindex.append(i)
                x_test=x_test1[Uindex]
                y_test=y_test1[Uindex]
                print(len(x_test))
                if len(x_test)==0:
                    ag_error2=0     
                else:
        
                    if sum(y_ag)<5:
                        y_ag_pre=[0]*len(y_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    elif sum(y_ag)>len(y_ag)-5:
                        y_ag_pre=[1]*len(y_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    else:
                        def train_SVC(params):
                            clf=SVC(kernel='rbf',**params)
                            metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                            return -metric
                        space4svc = {
                            'C': hp.uniform('C', 0, 1000),
                            'gamma': hp.uniform('gamma', 0, 20),
                        }
                       
                        ag_error2=1000
                        
                        best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                        svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                        svc_ag.fit(x_ag,y_ag)
                        y_ag_pre=svc_ag.predict(x_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                
                vag=volume_grid(ag)
                ag_error[2*j]=ag_error[2*j]+ag_error2
                ag_error[2*j+1]=ag_error[2*j+1]+vag
            else:
                ag=ag1
                ag_error2=0
                vag=volume_grid(ag)
                for m in range(j,18):
                    ag_error[2*m]=ag_error[2*m]+ag_error2
                    ag_error[2*m+1]=ag_error[2*m+1]+vag
                break
        ag_error1=[x/100 for x in ag_error]
    result.loc[k-1]=ag_error1
    #result.to_csv('...//AMCresult%d.csv'%k)
'''
'''
#ALE
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,7)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),7)

result=pd.DataFrame(columns=['10svc','10v','20svc','20v','40svc','40v','60svc','60v','80svc','80v','100svc','100v','200svc','200v','300svc','300v','400svc','400v','500svc','500v','600svc','600v','800svc','800v','1005svc','1005v'])
for k in range(36,40):
    print('Running ALE %d case'%k)
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=odata[odata['caseID']==k]
    data.index=range(0,len(data))
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000001:
                if abs(data['acc'][i]-x2)<0.000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f

    n=[10,20,40,60,80,100,200,300,400,500,600,800,1005]
    ag_error=[0]*26
    for t in range(0,100):
        ag1=ActiveLearningSVM(1005,data,2,method='entropy')
        ag1=ag1[ag1[:,1]==3]
        for j in range(0,13):
            x_test1=np.array(data[['eoff','acc']])
            y_test1=np.array(data['crash'])
            ag=ag1[0:n[j],:]
            x_ag=ag[:,3:5]
            y_ag=ag[:,0]
            y_ag=(y_ag+1)/2
            vag=volume_grid(ag)
            if j==12:
                ag_error2=0
                ag_error[2*j]=ag_error[2*j]+ag_error2
                ag_error[2*j+1]=ag_error[2*j+1]+vag
            else:
                Alist=ag[ag[:,0]==-1][:,3:5]
                Blist=ag[ag[:,0]==1][:,3:5]
                Uindex=[]
                for i in range(0,len(x_test1)):
                    xk=x_test1[i]
                    n1=0
                    n2=0
                    for yk in Alist:
                        if all(round(x,7)<=y for x, y in zip(xk,yk)):
                            n1=1
                            break
                    for yk in Blist:
                        if all(round(x,7)>=y for x,y in zip(xk,yk)):
                            n2=1
                            break
                    if n1==0 and n2==0:
                        Uindex.append(i)
                x_test=x_test1[Uindex]
                y_test=y_test1[Uindex]
                if len(x_test)==0:
                    ag_error2=0     
                    for m in range(j,13):
                        ag_error[2*m]=ag_error[2*m]+ag_error2
                        ag_error[2*m+1]=ag_error[2*m+1]+vag
                        break
                else:
        
                    if sum(y_ag)<5:
                        y_ag_pre=[0]*len(y_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    elif sum(y_ag)>len(y_ag)-5:
                        y_ag_pre=[1]*len(y_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    else:
                        def train_SVC(params):
                            clf=SVC(kernel='rbf',**params)
                            metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                            return -metric
                        space4svc = {
                            'C': hp.uniform('C', 0, 1000),
                            'gamma': hp.uniform('gamma', 0, 20),
                        }
                       
                        ag_error2=1000
                        best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                        svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                        svc_ag.fit(x_ag,y_ag)
                        y_ag_pre=svc_ag.predict(x_test)
                        ywrong=x_test[y_ag_pre != y_test]
                        ag_error2=len(ywrong)
                    ag_error[2*j]=ag_error[2*j]+ag_error2
                    ag_error[2*j+1]=ag_error[2*j+1]+vag
        ag_error1=[x/100 for x in ag_error]
    result.loc[k-1]=ag_error1
    #result.to_csv('...//ALEresult%d.csv'%k)

'''
##########Mappings from original space to [0,1]^2##############
#One needs to run this part first before generate grid-based designs
#Outputs AGdata%d.csv and AIdata%d.csv for 44 cases
###############################################################
'''
#Mappings for grid-based designs
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
for k in range(1,45):
    data=odata[odata['caseID']==k]
    data.index=range(0,len(data))
    for i in range(0,len(data)):
        if data['acc'][i]==-10.3:
            data['acc'][i]=0
        elif data['acc'][i]==-3.3:
            data['acc'][i]=1
        else:
            data['acc'][i]=(10.8+data['acc'][i])/8
        if data['eoff'][i]==6.6:
            data['eoff'][i]=1
        elif data['eoff'][i]==0:
            data['eoff'][i]=0
        elif data['eoff'][i]==0.1:
            data['eoff'][i]=1/128
        elif data['eoff'][i]==6.5:
            data['eoff'][i]=127/128
        else:
            data['eoff'][i]=(10*(data['eoff'][i]-0.1)/64)
    
    #data.to_csv('...//AGdata%d.csv'%k)



#Mappings for inner-grid-based designs
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash','crash1']]
for k in range(1,45):
    data=odata[odata['caseID']==k]
    data.index=range(0,len(data))
    for i in range(0,len(data)):
        if data['acc'][i]==-10.3:
            data['acc'][i]=1/16
        elif data['acc'][i]==-3.3:
            data['acc'][i]=15/16
        else:
            data['acc'][i]=(10.8+data['acc'][i])/8
        if data['eoff'][i]==6.6:
            data['eoff'][i]=255/256
        elif data['eoff'][i]==0:
            data['eoff'][i]=1/256
        elif data['eoff'][i]==0.1:
            data['eoff'][i]=1/128
        elif data['eoff'][i]==6.5:
            data['eoff'][i]=127/128
        else:
            data['eoff'][i]=(10*(data['eoff'][i]-0.1)/64)
    
    #data.to_csv('...\\AIdata%d.csv'%k)
'''
'''
#AG
#Needs AGdata%d.csv
result=pd.DataFrame(columns=['10svc','10v','15svc','15v','20svc','20v','25svc','25v','30svc','30v','35svc','35v','40svc','40v','45svc','45v','50svc','50v','55svc','55v'])
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,8)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),8)

for k in range(1,45):
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=pd.read_csv("...//AGdata%d.csv"%k,index_col=0)
    data=data[['eoff','acc','crash']]
    data['eoff']=round(data['eoff'],8)
    data['acc']=round(data['acc'],8)
    data1=odata[odata['caseID']==k]
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000001:
                if abs(data['acc'][i]-x2)<0.000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f
    
    n=[10,15,20,25,30,35,40,45,50,55]
    ag_error=[]
    for j in range(0,10):
        x_test1=np.array(data1[['eoff','acc']])
        y_test1=np.array(data1['crash'])
        ag1,agmaxg=AG(2,n[j],1)
        ag1=ag1[ag1[:,1]==3]
        ag=ag1[ag1[:,0]!=0]
        for i in range(0,len(ag)):
            if ag[i,4]==0:
                ag[i,4]=-10.3
            elif ag[i,4]==1:
                ag[i,4]=-3.3
            else:
                ag[i,4]=round(8*ag[i,4]-10.8,8)
                
            if ag[i,3]==1:
                ag[i,3]=6.6
            elif ag[i,3]==0:
                ag[i,3]=0
            elif ag[i,3]==round(1/128,8):
                ag[i,3]=0.1
            elif ag[i,3]==round(127/128,8):
                ag[i,3]=6.5
            else:
                ag[i,3]=round(ag[i,3]/10*64+0.1,8)

        for i in range(0,len(ag)):
            ag[i,3]=round(ag[i,3]/6.6,8)
            ag[i,4]=round((ag[i,4]+10.3)/(-3.3+10.3),8)
        x_ag=ag[:,3:5]
        y_ag=ag[:,0]
        y_ag=(y_ag+1)/2
        Alist=ag[ag[:,0]==-1][:,3:5]
        Blist=ag[ag[:,0]==1][:,3:5]
        Uindex=[]
        for i in range(0,len(x_test1)):
            xk=x_test1[i]
            n1=0
            n2=0
            for yk in Alist:
                if all(round(x,8)<=y for x, y in zip(xk,yk)):
                    n1=1
                    break
            for yk in Blist:
                if all(round(x,8)>=y for x,y in zip(xk,yk)):
                    n2=1
                    break
            if n1==0 and n2==0:
                Uindex.append(i)
        x_test=x_test1[Uindex]
        y_test=y_test1[Uindex]
        if len(x_test)==0:
            ag_error2=0
            vag=volume_grid(ag)
            ag_error.extend([0,vag]*(10-j))
            break
        else:

            if sum(y_ag)<5:
                y_ag_pre=[0]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            elif sum(y_ag)>len(y_ag)-5:
                y_ag_pre=[1]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            else:
                def train_SVC(params):
                    clf=SVC(kernel='rbf',**params)
                    metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                    return -metric
                space4svc = {
                    'C': hp.uniform('C', 0, 1000),
                    'gamma': hp.uniform('gamma', 0, 20),
                }
               
                ag_error2=1000
                for t in range(0,1):
                    best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                    svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                    svc_ag.fit(x_ag,y_ag)
                    y_ag_pre=svc_ag.predict(x_test)
                    ywrong=x_test[y_ag_pre != y_test]
                    ag_error1=len(ywrong)
                    if ag_error1<ag_error2:
                        ag_error2=ag_error1
                    if ag_error2<=1:
                        break
            vag=volume_grid(ag)
            ag_error.append(ag_error2)
            ag_error.append(vag)

    result.loc[k-1]=ag_error
   # result.to_csv('...//AGresult%d.csv'%k)
   
'''
'''
#GG
result=pd.DataFrame(columns=['10svc','10v','20svc','20v','30svc','30v','40svc','40v','50svc','50v','60svc','60v','70svc','70v','80svc','80v','90svc','90v','100svc','100v'])
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,8)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),8)

for k in range(1,45):
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=pd.read_csv("...//AGdata%d.csv"%k,index_col=0)
    data=data[['eoff','acc','crash']]
    data['eoff']=round(data['eoff'],8)
    data['acc']=round(data['acc'],8)
    data1=odata[odata['caseID']==k]
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000001:
                if abs(data['acc'][i]-x2)<0.000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f
    n=[10,20,30,40,50,60,70,80,90,100]
    ag_error=[]
    for j in range(0,10):
        x_test1=np.array(data1[['eoff','acc']])
        y_test1=np.array(data1['crash'])
        ag1,agmaxg=GG(2,n[j],1)
        ag1=ag1[ag1[:,1]==3]
        ag=ag1[ag1[:,0]!=0]
        for i in range(0,len(ag)):
            if ag[i,4]==0:
                ag[i,4]=-10.3
            elif ag[i,4]==1:
                ag[i,4]=-3.3
            else:
                ag[i,4]=round(8*ag[i,4]-10.8,8)
                
            if ag[i,3]==1:
                ag[i,3]=6.6
            elif ag[i,3]==0:
                ag[i,3]=0
            elif ag[i,3]==round(1/128,8):
                ag[i,3]=0.1
            elif ag[i,3]==round(127/128,8):
                ag[i,3]=6.5
            else:
                ag[i,3]=round(ag[i,3]/10*64+0.1,8)

        for i in range(0,len(ag)):
            ag[i,3]=round(ag[i,3]/6.6,8)
            ag[i,4]=round((ag[i,4]+10.3)/(-3.3+10.3),8)
        x_ag=ag[:,3:5]
        y_ag=ag[:,0]
        y_ag=(y_ag+1)/2
        Alist=ag[ag[:,0]==-1][:,3:5]
        Blist=ag[ag[:,0]==1][:,3:5]
        Uindex=[]
        for i in range(0,len(x_test1)):
            xk=x_test1[i]
            n1=0
            n2=0
            for yk in Alist:
                if all(round(x,8)<=y for x, y in zip(xk,yk)):
                    n1=1
                    break
            for yk in Blist:
                if all(round(x,8)>=y for x,y in zip(xk,yk)):
                    n2=1
                    break
            if n1==0 and n2==0:
                Uindex.append(i)
        x_test=x_test1[Uindex]
        y_test=y_test1[Uindex]
        if len(x_test)==0:
            ag_error2=0
            vag=volume_grid(ag)
            ag_error.extend([0,vag]*(10-j))
            break
        else:

            if sum(y_ag)<5:
                y_ag_pre=[0]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            elif sum(y_ag)>len(y_ag)-5:
                y_ag_pre=[1]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            else:
                def train_SVC(params):
                    clf=SVC(kernel='rbf',**params)
                    metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                    return -metric
                space4svc = {
                    'C': hp.uniform('C', 0, 1000),
                    'gamma': hp.uniform('gamma', 0, 20),
                }
               
                ag_error2=1000
                for t in range(0,1):
                    best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                    svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                    svc_ag.fit(x_ag,y_ag)
                    y_ag_pre=svc_ag.predict(x_test)
                    ywrong=x_test[y_ag_pre != y_test]
                    ag_error1=len(ywrong)
                    if ag_error1<ag_error2:
                        ag_error2=ag_error1
                    if ag_error2<=1:
                        break
            vag=volume_grid(ag)
            ag_error.append(ag_error2)
            ag_error.append(vag)
    result.loc[k-1]=ag_error
    #result.to_csv('...//GGresult%d.csv'%k)

'''
'''
#SG
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,8)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),8)
result=pd.DataFrame(columns=['4svc','4v','9svc','9v','25svc','25v','81svc','81v','255svc','255v','495svc','495v','975svc','975v','1005svc','1005v'])

for k in range(1,45):
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=pd.read_csv("...//AGdata%d.csv"%k,index_col=0)
    data=data[['eoff','acc','crash']]
    data['eoff']=round(data['eoff'],8)
    data['acc']=round(data['acc'],8)
    data1=odata[odata['caseID']==k]
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000001:
                if abs(data['acc'][i]-x2)<0.000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f

    n=[4,9,25,81,255,495,975,1005]
    ag_error=[]
    for j in range(0,8):

        ag1,maxg=SG(2,n[j],1)
        ag=ag1[ag1[:,0]!=0]
        x_test1=np.array(data1[['eoff','acc']])
        y_test1=np.array(data1['crash'])
        for i in range(0,len(ag)):
            if ag[i,4]==0:
                ag[i,4]=-10.3
            elif ag[i,4]==1:
                ag[i,4]=-3.3
            else:
                ag[i,4]=round(8*ag[i,4]-10.8,8)
                
            if ag[i,3]==1:
                ag[i,3]=6.6
            elif ag[i,3]==0:
                ag[i,3]=0
            elif ag[i,3]==round(1/128,8):
                ag[i,3]=0.1
            elif ag[i,3]==round(127/128,8):
                ag[i,3]=6.5
            else:
                ag[i,3]=round(ag[i,3]/10*64+0.1,8)

        for i in range(0,len(ag)):
            ag[i,3]=round(ag[i,3]/6.6,8)
            ag[i,4]=round((ag[i,4]+10.3)/(-3.3+10.3),8)
        x_ag=ag[:,3:5]
        y_ag=ag[:,0]
        y_ag=(y_ag+1)/2
        Alist=ag[ag[:,0]==-1][:,3:5]
        Blist=ag[ag[:,0]==1][:,3:5]
        Uindex=[]
        for i in range(0,len(x_test1)):
            xk=x_test1[i]
            n1=0
            n2=0
            for yk in Alist:
                if all(round(x,8)<=y for x, y in zip(xk,yk)):
                    n1=1
                    break
            for yk in Blist:
                if all(round(x,8)>=y for x,y in zip(xk,yk)):
                    n2=1
                    break
            if n1==0 and n2==0:
                Uindex.append(i)
        x_test=x_test1[Uindex]
        y_test=y_test1[Uindex]
        if len(x_test)==0:
            ag_error2=0
            vag=volume_grid(ag)
            ag_error.extend([0,vag]*(8-j))
            break

        else:

            if sum(y_ag)<5:
                y_ag_pre=[0]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            elif sum(y_ag)>len(y_ag)-5:
                y_ag_pre=[1]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            else:
                def train_SVC(params):
                    clf=SVC(kernel='rbf',**params)
                    metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                    return -metric
                space4svc = {
                    'C': hp.uniform('C', 0, 1000),
                    'gamma': hp.uniform('gamma', 0, 20),
                }
               
                ag_error2=1000
                for t in range(0,1):
                    best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                    svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                    svc_ag.fit(x_ag,y_ag)
                    y_ag_pre=svc_ag.predict(x_test)
                    ywrong=x_test[y_ag_pre != y_test]
                    ag_error1=len(ywrong)
                    if ag_error1<ag_error2:
                        ag_error2=ag_error1
                    if ag_error2<=1:
                        break
  
            vag=volume_grid(ag)
            ag_error.append(ag_error2)
            ag_error.append(vag)
    result.loc[k-1]=ag_error
    #result.to_csv('...//SGresult%d.csv'%k)
'''
'''
##SI
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,8)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),8)
result=pd.DataFrame(columns=['1svc','1v','9svc','9v','49svc','49v','225svc','225v','465svc','465v','945svc','945v','975svc','975v','1005svc','1005v'])
for k in range(1,45):
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=pd.read_csv("...//AIdata%d.csv"%k,index_col=0)
    data=data[['eoff','acc','crash']]
    data['eoff']=round(data['eoff'],8)
    data['acc']=round(data['acc'],8)
    data1=odata[odata['caseID']==k]
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000000001:
                if abs(data['acc'][i]-x2)<0.000000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f
    n=[1,9,49,225,465,945,975,1005]
    
    ag_error=[]
    for j in range(0,8):
        x_test1=np.array(data1[['eoff','acc']])
        y_test1=np.array(data1['crash'])
        ag1,maxg=SI(2,n[j],1)
        ag1=ag1[ag1[:,1]==3]
        ag=ag1[ag1[:,0]!=0]
        
        for i in range(0,len(ag)):
            if ag[i,4]==1/16:
                ag[i,4]=-10.3
            elif ag[i,4]==15/16:
                ag[i,4]=-3.3
            else:
                ag[i,4]=round(8*ag[i,4]-10.8,8)
                
            if ag[i,3]==round(255/256,8):
                ag[i,3]=6.6
            elif ag[i,3]==round(1/256,8):
                ag[i,3]=0
            elif ag[i,3]==round(1/128,8):
                ag[i,3]=0.1
            elif ag[i,3]==round(127/128,8):
                ag[i,3]=6.5
            else:
                ag[i,3]=round(ag[i,3]/10*64+0.1,8)
                
        for i in range(0,len(ag)):
            ag[i,3]=round(ag[i,3]/6.6,8)
            ag[i,4]=round((ag[i,4]+10.3)/(-3.3+10.3),8)
        x_ag=ag[:,3:5]
        y_ag=ag[:,0]
        y_ag=(y_ag+1)/2
        Alist=ag[ag[:,0]==-1][:,3:5]
        Blist=ag[ag[:,0]==1][:,3:5]
        Uindex=[]
        for i in range(0,len(x_test1)):
            xk=x_test1[i]
            n1=0
            n2=0
            for yk in Alist:
                if all(round(x,9)<=y for x, y in zip(xk,yk)):
                    n1=1
                    break
            for yk in Blist:
                if all(round(x,9)>=y for x,y in zip(xk,yk)):
                    n2=1
                    break
            if n1==0 and n2==0:
                Uindex.append(i)
        x_test=x_test1[Uindex]
        y_test=y_test1[Uindex]
        if len(x_test)==0:
            ag_error2=0
            vag=volume_grid(ag)
            data2 = np.random.uniform(low=[0] * 2, high=[1] * 2, size=(100000, 2))
            q=0
            for x in data2:
                resp=FindResponse(x,ag,2)
                if resp==0:
                    q=q+1

            ag_error.extend([0,vag]*(8-j))
            break

        else:

            if sum(y_ag)<5:
                y_ag_pre=[0]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            elif sum(y_ag)>len(y_ag)-5:
                y_ag_pre=[1]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            else:
                def train_SVC(params):
                    clf=SVC(kernel='rbf',**params)
                    metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                    return -metric
                space4svc = {
                    'C': hp.uniform('C', 0, 1000),
                    'gamma': hp.uniform('gamma', 0, 20),
                }
               
                ag_error2=1000
                for t in range(0,1):
                    best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                    svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                    svc_ag.fit(x_ag,y_ag)
                    y_ag_pre=svc_ag.predict(x_test)
                    ywrong=x_test[y_ag_pre != y_test]
                    ag_error1=len(ywrong)
                    if ag_error1<ag_error2:
                        ag_error2=ag_error1
                    if ag_error2<=1:
                        break
            vag=volume_grid(ag)
            ag_error.append(ag_error2)
            ag_error.append(vag)

    result.loc[k-1]=ag_error
    #result.to_csv('...//SIresult%d.csv'%k)
'''
'''
#AI
odata=pd.read_csv('crashData.csv',index_col=0)
odata=odata[['caseID','eoff','acc','crash']]
odata['eoff']=round(odata['eoff']/6.6,8)
odata['acc']=round((odata['acc']+10.3)/(-3.3+10.3),8)
result=pd.DataFrame(columns=['10svc','10v','15svc','15v','20svc','20v','25svc','25v','30svc','30v','35svc','35v','40svc','40v','45svc','45v','50svc','50v','55svc','55v','60svc','60v'])
for k in range(1,45):
    if k==23 or k==25 or k==40 or k==41:
        continue
    data=pd.read_csv("...//AIdata%d.csv"%k,index_col=0)
    data=data[['eoff','acc','crash']]
    data['eoff']=round(data['eoff'],8)
    data['acc']=round(data['acc'],8)
    data1=odata[odata['caseID']==k]
    def f_state(x,p,q):
        x1=x[0]
        x2=x[1]
        f=0
        result=data['crash']
        for i in range(0,len(result)):  
            if abs(data['eoff'][i]-x1)<=0.000000001:
                if abs(data['acc'][i]-x2)<0.000000001:
                    f=data['crash'][i]
                    if f==0:
                        f=-1
        return f
    n=[10,15,20,25,30,35,40,45,50,55,60]
    ag_error=[]
    for j in range(0,11):
        x_test1=np.array(data1[['eoff','acc']])
        y_test1=np.array(data1['crash'])
        ag1,agmaxg=AI(2,n[j],1)
        ag1=ag1[ag1[:,1]==3]
        ag=ag1[ag1[:,0]!=0]
        for i in range(0,len(ag)):
            if ag[i,4]==1/16:
                ag[i,4]=-10.3
            elif ag[i,4]==15/16:
                ag[i,4]=-3.3
            else:
                ag[i,4]=round(8*ag[i,4]-10.8,8)
                
            if ag[i,3]==round(255/256,8):
                ag[i,3]=6.6
            elif ag[i,3]==round(1/256,8):
                ag[i,3]=0
            elif ag[i,3]==round(1/128,8):
                ag[i,3]=0.1
            elif ag[i,3]==round(127/128,8):
                ag[i,3]=6.5
            else:
                ag[i,3]=round(ag[i,3]/10*64+0.1,8)
                
        for i in range(0,len(ag)):
            ag[i,3]=round(ag[i,3]/6.6,8)
            ag[i,4]=round((ag[i,4]+10.3)/(-3.3+10.3),8)
           
        x_ag=ag[:,3:5]
        y_ag=ag[:,0]
        y_ag=(y_ag+1)/2
        Alist=ag[ag[:,0]==-1][:,3:5]
        Blist=ag[ag[:,0]==1][:,3:5]
        Uindex=[]
        for i in range(0,len(x_test1)):
            xk=x_test1[i]
            n1=0
            n2=0
            for yk in Alist:
                if all(round(x,9)<=y for x, y in zip(xk,yk)):
                    n1=1
                    break
            for yk in Blist:
                if all(round(x,9)>=y for x,y in zip(xk,yk)):
                    n2=1
                    break
            if n1==0 and n2==0:
                Uindex.append(i)
        x_test=x_test1[Uindex]
        y_test=y_test1[Uindex]
        if len(x_test)==0:
            ag_error2=0
            vag=volume_grid(ag)
            ag_error.extend([0,vag]*(11-j))
            break

        else:

            if sum(y_ag)<5:
                y_ag_pre=[0]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            elif sum(y_ag)>len(y_ag)-5:
                y_ag_pre=[1]*len(y_test)
                ywrong=x_test[y_ag_pre != y_test]
                ag_error2=len(ywrong)
            else:
                def train_SVC(params):
                    clf=SVC(kernel='rbf',**params)
                    metric = cross_val_score(clf,x_ag,y_ag,cv=5).mean()
                    return -metric
                space4svc = {
                    'C': hp.uniform('C', 0, 1000),
                    'gamma': hp.uniform('gamma', 0, 20),
                }
               
                ag_error2=1000
                for t in range(0,1):
                    best_svc=fmin(train_SVC,space4svc,algo=tpe.suggest,max_evals=100)
                    svc_ag=SVC(kernel='rbf',**best_svc,probability=True)
                    svc_ag.fit(x_ag,y_ag)
                    y_ag_pre=svc_ag.predict(x_test)
                    ywrong=x_test[y_ag_pre != y_test]
                    ag_error1=len(ywrong)
                    if ag_error1<ag_error2:
                        ag_error2=ag_error1
                    if ag_error2<=1:
                        break
           
            vag=volume_grid(ag)
            ag_error.append(ag_error2)
            ag_error.append(vag)
    result.loc[k-1]=ag_error
    #result.to_csv('...//AIresult%d.csv'%k)
'''

