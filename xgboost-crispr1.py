# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:38:59 2019

@author: xuhi
"""
#%%
import xgboost as xgb
import matlab.engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,auc,roc_auc_score,roc_curve,precision_recall_curve
def xgboost_predict(train_data,train_label,test_data,test_label):
    xgb_model=xgb.XGBClassifier().fit(train_data,train_label)
    predictions=xgb_model.predict(test_data)
    predict_score=xgb_model.predict_proba(test_data)
    fpr,tpr,thresholds=roc_curve(test_label,predict_score[:,1])

    precision,recall,prthresholds=precision_recall_curve(test_label,predict_score[:,1])
    roc_auc=auc(fpr,tpr)
    prc_auc=auc(recall,precision)
    return xgb_model,predictions,predict_score,fpr,tpr,precision,recall,roc_auc,prc_auc


#%% three encoding methods

train_data=pd.read_csv('S5_Table.csv')
train_label=np.array(train_data[['observed']]>0).astype(int).ravel()
sequence=train_data[['sgRNA sequence','DNA site sequence']]
seq=sequence['sgRNA sequence']+sequence['DNA site sequence']
matlab_env=matlab.engine.start_matlab()
train_sequence1=np.array(matlab_env.sequenceEncode1(seq.tolist()))
train_sequence2=np.array(matlab_env.sequenceEncode2(seq.tolist()))
train_sequence3=np.array(matlab_env.sequenceEncode3(seq.tolist()))

kfold=KFold(n_splits=5,shuffle=True)  
mean_fpr=np.linspace(0,1,100)
mean_recall=np.linspace(0,1,100)
tprs1=[]
tprs2=[]
tprs3=[]
precisions1=[]
precisions2=[]
precisions3=[]
roc_aucs1=[]
roc_aucs2=[]
roc_aucs3=[]
prc_aucs1=[] 
prc_aucs2=[]
prc_aucs3=[]
for train_index,test_index in kfold.split(train_sequence1):
    xgb_model,predictions,predict_prob,fpr1,tpr1,precision1,recall1,roc_auc1,prc_auc1=xgboost_predict(train_sequence1[train_index,:],train_label[train_index],train_sequence1[test_index,:],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr2,tpr2,precision2,recall2,roc_auc2,prc_auc2=xgboost_predict(train_sequence2[train_index,:],train_label[train_index],train_sequence2[test_index,:],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr3,tpr3,precision3,recall3,roc_auc3,prc_auc3=xgboost_predict(train_sequence3[train_index,:],train_label[train_index],train_sequence3[test_index,:],train_label[test_index])
    tprs1.append(np.interp(mean_fpr,fpr1,tpr1))
    tprs1[-1][0]=0
    tprs2.append(np.interp(mean_fpr,fpr2,tpr2))
    tprs2[-1][0]=0
    tprs3.append(np.interp(mean_fpr,fpr3,tpr3))
    tprs3[-1][0]=0
    precisions1.append(np.interp(mean_recall,precision1,recall1))
    precisions1[-1][0]=1
    precisions2.append(np.interp(mean_recall,precision2,recall2))
    precisions2[-1][0]=1
    precisions3.append(np.interp(mean_recall,precision3,recall3))
    precisions3[-1][0]=1
    
    roc_aucs1.append(roc_auc1)
    roc_aucs2.append(roc_auc2)
    roc_aucs3.append(roc_auc3)
    prc_aucs1.append(roc_auc1)
    prc_aucs2.append(roc_auc2)
    prc_aucs3.append(roc_auc3)
#绘制ROC曲线
mean_tpr1=np.mean(tprs1,axis=0)
mean_tpr1[-1]=1
mean_tpr2=np.mean(tprs2,axis=0)
mean_tpr2[-1]=1
mean_tpr3=np.mean(tprs3,axis=0)
mean_tpr3[-1]=1
mean_auc1=auc(mean_fpr,mean_tpr1)
std_auc1=np.std(roc_aucs1)
mean_auc2=auc(mean_fpr,mean_tpr2)
std_auc2=np.std(roc_aucs2)
mean_auc3=auc(mean_fpr,mean_tpr3)
std_auc3=np.std(roc_aucs3)
plt.figure(1,figsize=[12,6])
plt.subplot(1,2,1)

#plt.figure(1,figsize=[6,6])
plt.plot(mean_fpr,mean_tpr1,label='1st encode method (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc1,std_auc1),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr2,label='2st encode method (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc2,std_auc2),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr3,label='3st encode method (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc3,std_auc3),lw=2,alpha=.8)
plt.legend(fontsize='medium')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
#out='encoderoc.png'
#plt.savefig(out,bbox_inches='tight',dpi=1000)
#plt.show()

#绘制PRC曲线
mean_precision1=np.mean(precisions1,axis=0)
mean_precision1[-1]=0
mean_precision2=np.mean(precisions2,axis=0)
mean_precision2[-1]=0
mean_precision3=np.mean(precisions3,axis=0)
mean_precision3[-1]=0
mean_prc1=auc(mean_recall,mean_precision1)
std_prc1=np.std(prc_aucs1)
mean_prc2=auc(mean_recall,mean_precision2)
std_prc2=np.std(prc_aucs2)
mean_prc3=auc(mean_recall,mean_precision3)
std_prc3=np.std(prc_aucs3)

plt.subplot(1,2,2)
#plt.figure(2,figsize=[6,6])

plt.plot(mean_recall,mean_precision1,label='1st encode method (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc1,std_prc1),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision2,label='2st encode method (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc2,std_prc2),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision3,label='3st encode method (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc3,std_prc3),lw=2,alpha=.8)
plt.legend(fontsize='medium')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC curve')

out='encodeprc.png'
plt.savefig(out,bbox_inches='tight',dpi=1000)
plt.show()

#%% 100 times cross validation
rocauc1=[]
rocauc2=[]
rocauc3=[]
prcauc1=[]
prcauc2=[]
prcauc3=[]
for i in range(0,100):   
    roc_aucs=[]
    prc_aucs=[]
    for j,(train_index,test_index) in enumerate(kfold.split(train_sequence1)):
        xgb_model,predictions,predict_prob,fpr,tpr,precision,recall,roc_auc,prc_auc=xgboost_predict(train_sequence1[train_index,:],train_label[train_index],train_sequence1[test_index,:],train_label[test_index])
        roc_aucs.append(roc_auc)
        prc_aucs.append(prc_auc)
    rocauc1.append(np.mean(roc_aucs))            
    prcauc1.append(np.mean(prc_aucs))             

print(np.mean(rocauc1),np.mean(prcauc1))      #0.8992   0.8884
for i in range(0,100):
    roc_aucs=[]
    prc_aucs=[]
    for j,(train_index,test_index) in enumerate(kfold.split(train_sequence2)):
        xgb_model,predictions,predict_prob,fpr,tpr,precision,recall,roc_auc,prc_auc=xgboost_predict(train_sequence2[train_index],train_label[train_index],train_sequence2[test_index],train_label[test_index])
        roc_aucs.append(roc_auc)
        prc_aucs.append(prc_auc)
    rocauc2.append(np.mean(roc_aucs))
    prcauc2.append(np.mean(prc_aucs))
print(np.mean(rocauc2),np.mean(prcauc2))     #0.9024   0.8867
for i in range(0,100):   
    roc_aucs=[]
    prc_aucs=[]
    for i,(train_index,test_index) in enumerate(kfold.split(train_sequence3)):
        xgb_model,predictions,predict_prob,fpr,tpr,precision,recall,roc_auc,prc_auc=xgboost_predict(train_sequence3[train_index],train_label[train_index],train_sequence3[test_index],train_label[test_index])
        roc_aucs.append(roc_auc)
        prc_aucs.append(prc_auc)
    rocauc3.append(np.mean(roc_aucs))
    prcauc3.append(np.mean(prc_aucs))
print(np.mean(rocauc3),np.mean(prcauc3))     #结果为0.9290    0.9244


#%%  use different feature to test model
train_data=pd.read_csv('S5_Table.csv')
train_score=np.array(train_data.iloc[:,-4:])
train_label=np.array(train_data[['observed']]>0).astype(int).ravel()
kfold=KFold(n_splits=5,shuffle=True)
mean_fpr=np.linspace(0,1,100)
mean_recall=np.linspace(0,1,100)
sequence=train_data[['sgRNA sequence','DNA site sequence']]
seq=sequence['sgRNA sequence']+sequence['DNA site sequence']
matlab_env=matlab.engine.start_matlab()
train_sequence1=np.array(matlab_env.sequenceEncode1(seq.tolist()))
train_sequence2=np.array(matlab_env.sequenceEncode2(seq.tolist()))
train_sequence3=np.array(matlab_env.sequenceEncode3(seq.tolist()))
train_data1=np.hstack((train_sequence1,train_score))
train_data2=np.hstack((train_sequence2,train_score))
train_data3=np.hstack((train_sequence3,train_score))
roc_aucs1=[]
roc_aucs2=[]
roc_aucs3=[]
roc_aucs4=[]
roc_aucs5=[]
roc_aucs6=[]
roc_aucs7=[]
roc_aucs8=[]
roc_aucs9=[]
roc_aucs10=[]
roc_aucs11=[]
roc_aucs12=[]
prc_aucs1=[]
prc_aucs2=[]
prc_aucs3=[]
prc_aucs4=[]
prc_aucs5=[]
prc_aucs6=[]
prc_aucs7=[]
prc_aucs8=[]
prc_aucs9=[]
prc_aucs10=[]
prc_aucs11=[]
prc_aucs12=[]
tprs1=[]
tprs2=[]
tprs3=[]
tprs4=[]
tprs5=[]
tprs6=[]
tprs7=[]
tprs8=[]
tprs9=[]
tprs10=[]
tprs11=[]
tprs12=[]
precisions1=[]
precisions2=[]
precisions3=[]
precisions4=[]
precisions5=[]
precisions6=[]
precisions7=[]
precisions8=[]
precisions9=[]
precisions10=[]
precisions11=[]
precisions12=[]
cctop_tprs=[]
optcd_tprs=[]
cfd_tprs=[]
cristaweb_tprs=[]
cctop_precisions=[]
optcd_precisions=[]
cfd_precisions=[]
cristaweb_precisions=[]
cctop_aucs=[]
optcd_aucs=[]
cfd_aucs=[]
cristaweb_aucs=[]
cctop_prcs=[]
optcd_prcs=[]
cfd_prcs=[]
cristaweb_prcs=[]
for train_index,test_index in kfold.split(train_score):
    #单个分数
    cctop_fpr,cctop_tpr,_=roc_curve(train_label[test_index],train_score[test_index,0])
    cctop_auc=auc(cctop_fpr,cctop_tpr)
    optcd_fpr,optcd_tpr,_=roc_curve(train_label[test_index],train_score[test_index,1])
    optcd_auc=auc(optcd_fpr,optcd_tpr)
    cfd_fpr,cfd_tpr,_=roc_curve(train_label[test_index],train_score[test_index,2])
    cfd_auc=auc(cfd_fpr,cfd_tpr)
    cristaweb_fpr,cristaweb_tpr,_=roc_curve(train_label[test_index],train_score[test_index,3])
    cristaweb_auc=auc(cristaweb_fpr,cristaweb_tpr)
    #prc
    cctop_precision,cctop_recall,_=precision_recall_curve(train_label[test_index],train_score[test_index,0])
    cctop_prc=auc(cctop_recall,cctop_precision)
    optcd_precision,optcd_recall,_=precision_recall_curve(train_label[test_index],train_score[test_index,1])
    optcd_prc=auc(optcd_recall,optcd_precision)
    cfd_precision,cfd_recall,_=precision_recall_curve(train_label[test_index],train_score[test_index,2])
    cfd_prc=auc(cfd_recall,cfd_precision)
    cristaweb_precision,cristaweb_recall,_=precision_recall_curve(train_label[test_index],train_score[test_index,3])
    cristaweb_prc=auc(cristaweb_recall,cristaweb_precision)
  
    #存储
    cctop_tprs.append(np.interp(mean_fpr,cctop_fpr,cctop_tpr))
    cctop_tprs[-1][0]=0
    optcd_tprs.append(np.interp(mean_fpr,optcd_fpr,optcd_tpr))
    optcd_tprs[-1][0]=0
  
    cfd_tprs.append(np.interp(mean_fpr,cfd_fpr,cfd_tpr))
    cfd_tprs[-1][0]=0
    cristaweb_tprs.append(np.interp(mean_fpr,cristaweb_fpr,cristaweb_tpr))
    cristaweb_tprs[-1][0]=0
    cctop_precisions.append(np.interp(mean_recall,cctop_precision,cctop_recall))
    cctop_precisions[-1][0]=1
    optcd_precisions.append(np.interp(mean_recall,optcd_precision,optcd_recall))
    optcd_precisions[-1][0]=1
    cfd_precisions.append(np.interp(mean_recall,cfd_precision,cfd_recall))
    cfd_precisions[-1][0]=1
    cristaweb_precisions.append(np.interp(mean_recall,cristaweb_precision,cristaweb_recall))
    cristaweb_precisions[-1][0]=1
    cctop_aucs.append(cctop_auc)
    optcd_aucs.append(optcd_auc)
    cfd_aucs.append(cfd_auc)
    cristaweb_aucs.append(cristaweb_auc)
    cctop_prcs.append(cctop_prc)
    optcd_prcs.append(optcd_prc)
    cfd_prcs.append(cfd_prc)
    cristaweb_prcs.append(cristaweb_prc)
    
#集成两个分数    
    xgb_model,predictions,predict_prob,fpr1,tpr1,precision1,recall1,roc_auc1,prc_auc1=xgboost_predict(train_score[train_index][:,(0,1)],train_label[train_index],train_score[test_index][:,(0,1)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr2,tpr2,precision2,recall2,roc_auc2,prc_auc2=xgboost_predict(train_score[train_index][:,(0,2)],train_label[train_index],train_score[test_index][:,(0,2)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr3,tpr3,precision3,recall3,roc_auc3,prc_auc3=xgboost_predict(train_score[train_index][:,(0,3)],train_label[train_index],train_score[test_index][:,(0,3)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr4,tpr4,precision4,recall4,roc_auc4,prc_auc4=xgboost_predict(train_score[train_index][:,(1,2)],train_label[train_index],train_score[test_index][:,(1,2)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr5,tpr5,precision5,recall5,roc_auc5,prc_auc5=xgboost_predict(train_score[train_index][:,(1,3)],train_label[train_index],train_score[test_index][:,(1,3)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr6,tpr6,precision6,recall6,roc_auc6,prc_auc6=xgboost_predict(train_score[train_index][:,(2,3)],train_label[train_index],train_score[test_index][:,(2,3)],train_label[test_index])
    tprs1.append(np.interp(mean_fpr,fpr1,tpr1))
    tprs1[-1][0]=0
    tprs2.append(np.interp(mean_fpr,fpr2,tpr2))
    tprs2[-1][0]=0
    tprs3.append(np.interp(mean_fpr,fpr3,tpr3))
    tprs3[-1][0]=0
    tprs4.append(np.interp(mean_fpr,fpr4,tpr4))
    tprs4[-1][0]=0
    tprs5.append(np.interp(mean_fpr,fpr5,tpr5))
    tprs5[-1][0]=0
    tprs6.append(np.interp(mean_fpr,fpr6,tpr6))
    tprs6[-1][0]=0        
    precisions1.append(np.interp(mean_recall,precision1,recall1))
    precisions1[-1][0]=1
    precisions2.append(np.interp(mean_recall,precision2,recall2))
    precisions2[-1][0]=1
    precisions3.append(np.interp(mean_recall,precision3,recall3))
    precisions3[-1][0]=1
    precisions4.append(np.interp(mean_recall,precision4,recall4))
    precisions4[-1][0]=1
    precisions5.append(np.interp(mean_recall,precision5,recall5))
    precisions5[-1][0]=1
    precisions6.append(np.interp(mean_recall,precision6,recall6))
    precisions6[-1][0]=1
    roc_aucs1.append(roc_auc1)
    roc_aucs2.append(roc_auc2)
    roc_aucs3.append(roc_auc3)
    roc_aucs4.append(roc_auc4)
    roc_aucs5.append(roc_auc5)
    roc_aucs6.append(roc_auc6)
    prc_aucs1.append(prc_auc1)
    prc_aucs2.append(prc_auc2)
    prc_aucs3.append(prc_auc3)
    prc_aucs4.append(prc_auc4)
    prc_aucs5.append(prc_auc5)
    prc_aucs6.append(prc_auc6)
    
#集成三个和四个分数
    xgb_model,predictions,predict_prob,fpr7,tpr7,precision7,recall7,roc_auc7,prc_auc7=xgboost_predict(train_score[train_index][:,(0,1,2)],train_label[train_index],train_score[test_index][:,(0,1,2)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr8,tpr8,precision8,recall8,roc_auc8,prc_auc8=xgboost_predict(train_score[train_index][:,(0,1,3)],train_label[train_index],train_score[test_index][:,(0,1,3)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr9,tpr9,precision9,recall9,roc_auc9,prc_auc9=xgboost_predict(train_score[train_index][:,(0,2,3)],train_label[train_index],train_score[test_index][:,(0,2,3)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr10,tpr10,precision10,recall10,roc_auc10,prc_auc10=xgboost_predict(train_score[train_index][:,(1,2,3)],train_label[train_index],train_score[test_index][:,(1,2,3)],train_label[test_index])
    xgb_model,predictions,predict_prob,fpr11,tpr11,precision11,recall11,roc_auc11,prc_auc11=xgboost_predict(train_score[train_index],train_label[train_index],train_score[test_index],train_label[test_index])
    tprs7.append(np.interp(mean_fpr,fpr7,tpr7))
    tprs7[-1][0]=0
    tprs8.append(np.interp(mean_fpr,fpr8,tpr8))
    tprs8[-1][0]=0
    tprs9.append(np.interp(mean_fpr,fpr9,tpr9))
    tprs9[-1][0]=0
    tprs10.append(np.interp(mean_fpr,fpr10,tpr10))
    tprs10[-1][0]=0
    tprs11.append(np.interp(mean_fpr,fpr11,tpr11))
    tprs11[-1][0]=0
       
    precisions7.append(np.interp(mean_recall,precision7,recall7))
    precisions7[-1][0]=1
    precisions8.append(np.interp(mean_recall,precision8,recall8))
    precisions8[-1][0]=1
    precisions9.append(np.interp(mean_recall,precision9,recall9))
    precisions9[-1][0]=1
    precisions10.append(np.interp(mean_recall,precision10,recall10))
    precisions10[-1][0]=1
    precisions11.append(np.interp(mean_recall,precision11,recall11))
    precisions11[-1][0]=1

    roc_aucs7.append(roc_auc7)
    roc_aucs8.append(roc_auc8)
    roc_aucs9.append(roc_auc9)
    roc_aucs10.append(roc_auc10)
    roc_aucs11.append(roc_auc11)

    prc_aucs7.append(prc_auc7)
    prc_aucs8.append(prc_auc8)
    prc_aucs9.append(prc_auc9)
    prc_aucs10.append(prc_auc10)
    prc_aucs11.append(prc_auc11)
    
    xgb_model,predictions,predict_prob,fpr12,tpr12,precision12,recall12,roc_auc12,prc_auc12=xgboost_predict(train_data3[train_index],train_label[train_index],train_data3[test_index],train_label[test_index])
    
    tprs12.append(np.interp(mean_fpr,fpr12,tpr12))
    tprs12[-1][0]=0
    precisions12.append(np.interp(mean_recall,precision12,recall12))
    precisions12[-1][0]=1
    roc_aucs12.append(roc_auc12)
    prc_aucs12.append(prc_auc12)

       

#单个分数绘制 ROC曲线
mean_cctop_tpr=np.mean(cctop_tprs,axis=0)
mean_cctop_tpr[-1]=1
mean_optcd_tpr=np.mean(optcd_tprs,axis=0)
mean_optcd_tpr[-1]=1
mean_cfd_tpr=np.mean(cfd_tprs,axis=0)
mean_cfd_tpr[-1]=1
mean_cristaweb_tpr=np.mean(cristaweb_tprs,axis=0)
mean_cristaweb_tpr[-1]=1
cctop_mean_auc=auc(mean_fpr,mean_cctop_tpr)
cctop_std_auc=np.std(cctop_aucs)
optcd_mean_auc=auc(mean_fpr,mean_optcd_tpr)
optcd_std_auc=np.std(optcd_aucs)
cfd_mean_auc=auc(mean_fpr,mean_cfd_tpr)
cfd_std_auc=np.std(cfd_aucs)
cristaweb_mean_auc=auc(mean_fpr,mean_cristaweb_tpr)
cristaweb_std_auc=np.std(cristaweb_aucs)

plt.figure(7,figsize=[12,6])
plt.subplot(1,2,1)
plt.plot(mean_fpr,mean_cctop_tpr,label='CCTop (ROC_AUC=%0.4f $\pm$ %0.2f)' %(cctop_mean_auc,cctop_std_auc),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_optcd_tpr,label='OptCD (ROC_AUC=%0.4f $\pm$ %0.2f)' %(optcd_mean_auc,optcd_std_auc),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_cfd_tpr,label='CFD (ROC_AUC=%0.4f $\pm$ %0.2f)' %(cfd_mean_auc,cfd_std_auc),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_cristaweb_tpr,label='CRISTAWEB (ROC_AUC=%0.4f $\pm$ %0.2f)' %(cristaweb_mean_auc,cristaweb_std_auc),lw=2,alpha=.8)
plt.legend(fontsize='medium',loc=4)
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
#out='onescoreroc.png'
#plt.savefig(out,bbox_inches='tight',dpi=600)
#单个分数绘制PRC曲线
mean_cctop_precision=np.mean(cctop_precisions,axis=0)
mean_cctop_precision[-1]=0
mean_optcd_precision=np.mean(optcd_precisions,axis=0)
mean_optcd_precision[-1]=0
mean_cfd_precision=np.mean(cfd_precisions,axis=0)
mean_cfd_precision[-1]=0
mean_cristaweb_precision=np.mean(cristaweb_precisions,axis=0)
mean_cristaweb_precision[-1]=0
cctop_mean_prc=auc(mean_recall,mean_cctop_precision)
cctop_std_prc=np.std(cctop_prcs)
optcd_mean_prc=auc(mean_recall,mean_optcd_precision)
optcd_std_prc=np.std(optcd_prcs)
cfd_mean_prc=auc(mean_recall,mean_cfd_precision)
cfd_std_prc=np.std(cfd_prcs)
cristaweb_mean_prc=auc(mean_recall,mean_cristaweb_precision)
cristaweb_std_prc=np.std(cristaweb_prcs)


#plt.figure(4,figsize=[6,6])
plt.subplot(1,2,2)
plt.plot(mean_recall,mean_cctop_precision,label='CCTop (ROC_AUC=%0.4f $\pm$ %0.2f)' %(cctop_mean_prc,cctop_std_prc),lw=2,alpha=.8)
plt.plot(mean_recall,mean_optcd_precision,label='OptCD (ROC_AUC=%0.4f $\pm$ %0.2f)' %(optcd_mean_prc,optcd_std_prc),lw=2,alpha=.8)
plt.plot(mean_recall,mean_cfd_precision,label='CFD (ROC_AUC=%0.4f $\pm$ %0.2f)' %(cfd_mean_prc,cfd_std_prc),lw=2,alpha=.8)
plt.plot(mean_recall,mean_cristaweb_precision,label='CRISTAWEB(ROC_AUC=%0.4f $\pm$ %0.2f)' %(cristaweb_mean_prc,cristaweb_std_prc),lw=2,alpha=.8)
plt.legend(fontsize='medium',loc=0)
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC curve')  
out='onescoreprc.png'

plt.savefig(out,bbox_inches='tight',dpi=1000)
plt.show()


#两个分数组合   
mean_tpr1=np.mean(tprs1,axis=0)
mean_tpr1[-1]=1
mean_tpr2=np.mean(tprs2,axis=0)
mean_tpr2[-1]=1
mean_tpr3=np.mean(tprs3,axis=0)
mean_tpr3[-1]=1
mean_tpr4=np.mean(tprs4,axis=0)
mean_tpr4[-1]=1
mean_tpr5=np.mean(tprs5,axis=0)
mean_tpr5[-1]=1
mean_tpr6=np.mean(tprs6,axis=0)
mean_tpr6[-1]=1
mean_auc1=auc(mean_fpr,mean_tpr1)
std_auc1=np.std(roc_aucs1)
mean_auc2=auc(mean_fpr,mean_tpr2)
std_auc2=np.std(roc_aucs2)
mean_auc3=auc(mean_fpr,mean_tpr3)
std_auc3=np.std(roc_aucs3)
mean_auc4=auc(mean_fpr,mean_tpr4)
std_auc4=np.std(roc_aucs4)
mean_auc5=auc(mean_fpr,mean_tpr5)
std_auc5=np.std(roc_aucs5)
mean_auc6=auc(mean_fpr,mean_tpr6)
std_auc6=np.std(roc_aucs6)
#roc曲线 
plt.figure(3,figsize=[12,6])
plt.subplot(1,2,1)
plt.plot(mean_fpr,mean_tpr1,label='CCTop+OptCD (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc1,std_auc1),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr2,label='CCTop+CFD (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc2,std_auc2),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr3,label='CCTop+CRISTAWEB (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc3,std_auc3),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr4,label='OptCD+CFD (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc4,std_auc4),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr5,label='OptCD+CRISTAWEB (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc5,std_auc5),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr6,label='CFD+CRISTAWEB (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc6,std_auc6),lw=2,alpha=.8)
plt.legend(fontsize='small',loc=4)
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
#out='twoscoreroc.png'
#plt.savefig(out,bbox_inches='tight',dpi=600)
#out='encoderoc.png'
#plt.savefig(out,bbox_inches='tight')

#prc曲线
mean_precision1=np.mean(precisions1,axis=0)
mean_precision1[-1]=0
mean_precision2=np.mean(precisions2,axis=0)
mean_precision2[-1]=0
mean_precision3=np.mean(precisions3,axis=0)
mean_precision3[-1]=0
mean_precision4=np.mean(precisions4,axis=0)
mean_precision4[-1]=0
mean_precision5=np.mean(precisions5,axis=0)
mean_precision5[-1]=0
mean_precision6=np.mean(precisions6,axis=0)
mean_precision6[-1]=0
mean_prc1=auc(mean_recall,mean_precision1)
std_prc1=np.std(prc_aucs1)
mean_prc2=auc(mean_recall,mean_precision2)
std_prc2=np.std(prc_aucs2)
mean_prc3=auc(mean_recall,mean_precision3)
std_prc3=np.std(prc_aucs3)   
mean_prc4=auc(mean_recall,mean_precision4)
std_prc4=np.std(prc_aucs4)
mean_prc5=auc(mean_recall,mean_precision5)
std_prc5=np.std(prc_aucs5)
mean_prc6=auc(mean_recall,mean_precision6)
std_prc6=np.std(prc_aucs6)
#plt.figure(6,figsize=[6,6])
plt.subplot(1,2,2)
plt.plot(mean_recall,mean_precision1,label='CCTop+OptCD (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc1,std_prc1),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision2,label='CCTop+CFD (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc2,std_prc2),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision3,label='CCTop+CRISTAWEB (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc3,std_prc3),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision4,label='OptCD+CFD (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc4,std_prc4),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision5,label='OptCD+CRISTAWEB (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc5,std_prc5),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision6,label='CFD+CRISTAWEB (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc6,std_prc6),lw=2,alpha=.8)
plt.legend(fontsize='small',loc=3)
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC curve')
out='twoscoreprc.png'
plt.savefig(out,bbox_inches='tight',dpi=1000)
#out='encoderoc.png'
#plt.savefig(out,bbox_inches='tight')
plt.show()

#三个分数以及四个分数组合的roc以及prc
mean_tpr7=np.mean(tprs7,axis=0)
mean_tpr7[-1]=1
mean_tpr8=np.mean(tprs8,axis=0)
mean_tpr8[-1]=1
mean_tpr9=np.mean(tprs9,axis=0)
mean_tpr9[-1]=1
mean_tpr10=np.mean(tprs10,axis=0)
mean_tpr10[-1]=1
mean_tpr11=np.mean(tprs11,axis=0)
mean_tpr11[-1]=1
mean_tpr12=np.mean(tprs12,axis=0)
mean_tpr12[-1]=1
mean_auc7=auc(mean_fpr,mean_tpr7)
std_auc7=np.std(roc_aucs7)
mean_auc8=auc(mean_fpr,mean_tpr8)
std_auc8=np.std(roc_aucs8)
mean_auc9=auc(mean_fpr,mean_tpr9)
std_auc9=np.std(roc_aucs9)
mean_auc10=auc(mean_fpr,mean_tpr10)
std_auc10=np.std(roc_aucs10)
mean_auc11=auc(mean_fpr,mean_tpr11)
std_auc11=np.std(roc_aucs11)
mean_auc12=auc(mean_fpr,mean_tpr12)
std_auc12=np.std(roc_aucs12)
#roc曲线 
plt.figure(4,figsize=[12,6])
plt.subplot(1,2,1)
plt.plot(mean_fpr,mean_tpr7,label='CCTop+OptCD+CFD (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc7,std_auc7),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr8,label='CCTop+OptCD+CRISTAWEB (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc8,std_auc8),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr9,label='CCTop+CFD+CRISTAWEB (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc9,std_auc9),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr10,label='OptCD+CFD+CRISTAWEB (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc10,std_auc10),lw=2,alpha=.8)

plt.legend(loc=4,fontsize='small')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
#out='threescoreroc.png'
#plt.savefig(out,bbox_inches='tight',dpi=600)
#out='encoderoc.png'
#plt.savefig(out,bbox_inches='tight')


mean_precision7=np.mean(precisions7,axis=0)
mean_precision7[-1]=0
mean_precision8=np.mean(precisions8,axis=0)
mean_precision8[-1]=0
mean_precision9=np.mean(precisions9,axis=0)
mean_precision9[-1]=0
mean_precision10=np.mean(precisions10,axis=0)
mean_precision10[-1]=0
mean_precision11=np.mean(precisions11,axis=0)
mean_precision11[-1]=0
mean_precision12=np.mean(precisions12,axis=0)
mean_precision12[-1]=0

mean_prc7=auc(mean_recall,mean_precision7)
std_prc7=np.std(prc_aucs7)
mean_prc8=auc(mean_recall,mean_precision8)
std_prc8=np.std(prc_aucs8)
mean_prc9=auc(mean_recall,mean_precision9)
std_prc9=np.std(prc_aucs9)   
mean_prc10=auc(mean_recall,mean_precision10)
std_prc10=np.std(prc_aucs10)
mean_prc11=auc(mean_recall,mean_precision11)
std_prc11=np.std(prc_aucs11)
mean_prc12=auc(mean_recall,mean_precision12)
std_prc12=np.std(prc_aucs12)

#plt.figure(8,figsize=[6,6])
plt.subplot(1,2,2)
plt.plot(mean_recall,mean_precision7,label='CCTop+OptCD+CFD (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc7,std_prc7),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision8,label='CCTop+OptCD+CRISTAWEB (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc8,std_prc8),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision9,label='CCTop+CFD+CRISTAWEB (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc9,std_prc9),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision10,label='OptCD+CFD+CRISTAWEB (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc10,std_prc10),lw=2,alpha=.8)

plt.legend(fontsize='small',loc=3)
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC curve')
out='threescoreprc.png'

plt.savefig(out,bbox_inches='tight',dpi=1000)
#out='encoderoc.png'
#plt.savefig(out,bbox_inches='tight')
plt.show()

plt.figure(5,figsize=[12,6])
plt.subplot(1,2,1)
plt.plot(mean_fpr,mean_tpr11,label='score (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc11,std_auc11),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_tpr12,label='sequence+score (ROC_AUC=%0.4f $\pm$ %0.2f)' % (mean_auc12,std_auc12),lw=2,alpha=.8)
plt.legend(loc=4,fontsize='medium')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
#out='score+sequenceroc.png'
#plt.savefig(out,bbox_inches='tight',dpi=600)
#out='encoderoc.png'
#plt.savefig(out,bbox_inches='tight')
#plt.show()


#plt.figure(10,figsize=[6,6])
plt.subplot(1,2,2)
plt.plot(mean_recall,mean_precision11,label='score (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc11,std_prc11),lw=2,alpha=.8)
plt.plot(mean_recall,mean_precision12,label='sequence+score (PRC_AUC=%0.4f $\pm$ %0.2f)' % (mean_prc12,std_prc12),lw=2,alpha=.8)
plt.legend(fontsize='medium',loc=3)
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC curve')
out='score+sequenceprc.png'
plt.savefig(out,bbox_inches='tight',dpi=1000)
#out='encoderoc.png'
#plt.savefig(out,bbox_inches='tight')
plt.show()
#%% comparation with other classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
names=["AdaBoost","SVM","DecisionTree","RF","MLP","XGB"]
clf=[AdaBoostClassifier(),SVC(),DecisionTreeClassifier(),RandomForestClassifier(),MLPClassifier(),XGBClassifier()]
mean_fpr=np.linspace(0,1,100)
mean_recall=np.linspace(0,1,100)
kfold=KFold(n_splits=5,shuffle=True)
for name,clf in zip(names,clf):
    tprs=[]
    aucs=[]
    precisions=[]
    prcs=[] 
    for train_index,test_index in kfold.split(train_label):
        data=train_data3
        model=clf.fit(data[train_index],train_label[train_index])
        if hasattr(model,'decision_function'):
            predict_score=model.decision_function(data[test_index])
        else:
            predict_score=model.predict_proba(data[test_index])[:,1]
        fpr,tpr,_=roc_curve(train_label[test_index],predict_score)
        precision,recall,_=precision_recall_curve(train_label[test_index],predict_score)
        
        tprs.append(np.interp(mean_fpr,fpr,tpr))
        precisions.append(np.interp(mean_recall,precision,recall))
        tprs[-1][0]=0
        precisions[-1][0]=1
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        prc_auc=auc(recall,precision)
        prcs.append(prc_auc)
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1
    mean_auc=auc(mean_fpr,mean_tpr)
    std_auc=np.std(aucs)
    mean_precision=np.mean(precisions,axis=0)
    mean_precision[-1]=0
    mean_prc=auc(mean_recall,mean_precision)
    std_prc=np.std(prcs)
    plt.figure(5,figsize=[12,6])
    plt.subplot(1,2,1)
    plt.plot(mean_fpr, mean_tpr,label=r'%s (ROC_AUC = %0.4f $\pm$ %0.2f)' % (name,mean_auc, std_auc),lw=2, alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(fontsize='medium')
    #out="classificationroc.png"
    #plt.savefig(out,bbox_inches='tight',dpi=600)

    #plt.figure(12,figsize=[6,6])
    plt.subplot(1,2,2)
    plt.plot(mean_recall, mean_precision,label=r'%s (PRC_AUC = %0.4f $\pm$ %0.2f)' % (name,mean_prc, std_prc),lw=2, alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRC')
    plt.legend(fontsize='medium')
    out="classificationprc.png"
    plt.savefig(out,bbox_inches='tight',dpi=1000)
plt.show()



    
#%% validation
import matlab.engine

train_data=pd.read_csv('S5_Table.csv')
train_score=np.array(train_data.iloc[:,-4:])
train_label=np.array(train_data[['observed']]>0).astype(int).ravel()
sequence=train_data[['sgRNA sequence','DNA site sequence']]
seq=sequence['sgRNA sequence']+sequence['DNA site sequence']
matlab_env=matlab.engine.start_matlab()
train_sequence=np.array(matlab_env.sequenceEncode3(seq.tolist()))
train_data=np.hstack((train_sequence,train_score))
validation_data=pd.read_csv('S9_Table.csv')
validation_score=np.array(validation_data.iloc[:,-4:])
validation_label=np.array(validation_data[['observed']]>0).astype(int).ravel()
validation_sequence=validation_data[['sgRNA sequence','DNA site sequence']]
validation_seq=validation_sequence['sgRNA sequence']+validation_sequence['DNA site sequence']
validation_sequenceinf=np.array(matlab_env.sequenceEncode3(validation_seq.tolist()))
validation_data=np.hstack((validation_sequenceinf,validation_score))
model=xgb.XGBClassifier(learning_rate=0.07,max_depth=5,n_estimators=97,reg_lambda=0.08,min_child_weight=1,n_jobs=4).fit(train_data,train_label)
score=model.predict_proba(validation_data)[:,1]
fpr,tpr,_=roc_curve(validation_label,score)
cctop_fpr,cctop_tpr,_=roc_curve(validation_label,validation_score[:,0])
optcd_fpr,optcd_tpr,_=roc_curve(validation_label,validation_score[:,1])
cfd_fpr,cfd_tpr,_=roc_curve(validation_label,validation_score[:,2])
cristaweb_fpr,cristaweb_tpr,_=roc_curve(validation_label,validation_score[:,3])


precision,recall,_=precision_recall_curve(validation_label,score)
cctop_precision,cctop_recall,_=precision_recall_curve(validation_label,validation_score[:,0])
optcd_precision,optcd_recall,_=precision_recall_curve(validation_label,validation_score[:,1])
cfd_precision,cfd_recall,_=precision_recall_curve(validation_label,validation_score[:,2])
cristaweb_precision,cristaweb_recall,_=precision_recall_curve(validation_label,validation_score[:,3])
roc_auc=auc(fpr,tpr)
prc_auc=auc(recall,precision)
cctop_roc_auc=auc(cctop_fpr,cctop_tpr)
optcd_roc_auc=auc(optcd_fpr,optcd_tpr)
cfd_roc_auc=auc(cfd_fpr,cfd_tpr)
cristaweb_roc_auc=auc(cristaweb_fpr,cristaweb_tpr)
cctop_prc_auc=auc(cctop_recall,cctop_precision)
optcd_prc_auc=auc(optcd_recall,optcd_precision)
cfd_prc_auc=auc(cfd_recall,cfd_precision)
cristaweb_prc_auc=auc(cristaweb_recall,cristaweb_precision)

mean_fpr=np.linspace(0,1,100)
plt.figure(6,figsize=[12,6])
plt.subplot(1,2,1)
plt.plot(fpr,tpr,label='XGBCRISPR (ROC_AUC = %0.4f)' % (roc_auc),lw=2, alpha=.8)
plt.plot(cctop_fpr,cctop_tpr,label='CCTop (ROC_AUC = %0.4f)' % (cctop_roc_auc),lw=2, alpha=.8)
plt.plot(optcd_fpr,optcd_tpr,label='OptCD (ROC_AUC = %0.4f)' % (optcd_roc_auc),lw=2, alpha=.8)
plt.plot(cfd_fpr,cfd_tpr,label='CFD (ROC_AUC = %0.4f)' % (cfd_roc_auc),lw=2, alpha=.8)
plt.plot(cristaweb_fpr,cristaweb_tpr,label='CRISTAWEB (ROC_AUC = %0.4f)' % (cristaweb_roc_auc),lw=2, alpha=.8)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(fontsize='medium')
#out="validation_roc.png"
#plt.savefig(out,bbox_inches='tight',dpi=600)
#plt.show()
#plt.figure(2,figsize=[6,6])
plt.subplot(1,2,2)
plt.plot(recall, precision,label='XGBCRISPR (PRC_AUC = %0.4f )' % (prc_auc),lw=2, alpha=.8)
plt.plot(cctop_recall, cctop_precision,label='CCTop (PRC_AUC = %0.4f )' % (cctop_prc_auc),lw=2, alpha=.8)
plt.plot(optcd_recall, optcd_precision,label='OptCD (PRC_AUC = %0.4f )' % (optcd_prc_auc),lw=2, alpha=.8)
plt.plot(cfd_recall, cfd_precision,label='CFD (PRC_AUC = %0.4f )' % (cfd_prc_auc),lw=2, alpha=.8)
plt.plot(cristaweb_recall, cristaweb_precision,label='CRISTAWEB (PRC_AUC = %0.4f )' % (cristaweb_prc_auc),lw=2, alpha=.8)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC')
plt.legend(fontsize='medium')
out="validation_prc.png"
plt.savefig(out,bbox_inches='tight',dpi=1000)

plt.show()