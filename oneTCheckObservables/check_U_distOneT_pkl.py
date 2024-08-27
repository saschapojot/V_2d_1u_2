import pickle
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
import statsmodels.api as sm
import sys
import re
import warnings
from scipy.stats import ks_2samp
import glob
from pathlib import Path
import os
import json



#This script checks if U, x,y values reach equilibrium and writes summary file of dist
#This file checks pkl files


argErrCode=2
sameErrCode=3
missingErrCode=4
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit(argErrCode)



jsonFromSummaryLast=json.loads(sys.argv[1])
jsonDataFromConf=json.loads(sys.argv[2])


TDirRoot=jsonFromSummaryLast["TDirRoot"]
U_dist_dataDir=jsonFromSummaryLast["U_dist_dataDir"]
effective_data_num_required=int(jsonDataFromConf["effective_data_num_required"])
N=int(jsonDataFromConf["unitCellNum"])

summary_U_distFile=TDirRoot+"/summary_U_dist.txt"
# print(summary_U_distFile)
lastFileNum=8
def sort_data_files_by_sweepEnd(oneDir):
    dataFilesAll=[]
    sweepEndAll=[]
    # print("entering sort")
    for oneDataFile in glob.glob(oneDir+"/*.pkl"):
        # print(oneDataFile)
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"sweepEnd(\d+)",oneDataFile)
        if matchEnd:
            indTmp=int(matchEnd.group(1))
            sweepEndAll.append(indTmp)
    endInds=np.argsort(sweepEndAll)
    sortedDataFiles=[dataFilesAll[i] for i in endInds]
    return sortedDataFiles

# def sort_data_files_by_num(oneDir):
#     dataFilesAll=[]
#     sweepEndAll=[]
#     # print("entering sort")
#     for oneDataFile in glob.glob(oneDir+"/*.pkl"):
#         # print(oneDataFile)
#         dataFilesAll.append(oneDataFile)
#         matchEnd=re.search(r"File(\d+)",oneDataFile)
#         if matchEnd:
#             indTmp=int(matchEnd.group(1))
#             sweepEndAll.append(indTmp)
#     endInds=np.argsort(sweepEndAll)
#     sortedDataFiles=[dataFilesAll[i] for i in endInds]
#     return sortedDataFiles

def parseSummaryU_Dist():
    startingFileInd=-1
    startingVecPosition=-1

    summaryFileExists=os.path.isfile(summary_U_distFile)
    if summaryFileExists==False:
        return startingFileInd,startingVecPosition

    with open(summary_U_distFile,"r") as fptr:
        lines=fptr.readlines()
    for oneLine in lines:
        #match startingFileInd
        matchStartingFileInd=re.search(r"startingFileInd=(\d+)",oneLine)
        if matchStartingFileInd:
            startingFileInd=int(matchStartingFileInd.group(1))

        #match startingVecPosition
        matchStartingVecPosition=re.search(r"startingVecPosition=(\d+)",oneLine)
        if matchStartingVecPosition:
            startingVecPosition=int(matchStartingVecPosition.group(1))

    return startingFileInd, startingVecPosition




def auto_corrForOneColumn(colVec):
    """

    :param colVec: a vector of data
    :return:
    """
    same=False
    eps=5e-2
    NLags=int(len(colVec)*1/4)
    # print("NLags="+str(NLags))
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
    try:
        acfOfVec=sm.tsa.acf(colVec,nlags=NLags)
    except Warning as w:
        same=True
    acfOfVecAbs=np.abs(acfOfVec)
    minAutc=np.min(acfOfVecAbs)

    lagVal=-1
    if minAutc<=eps:
        lagVal=np.where(acfOfVecAbs<=eps)[0][0]
    # np.savetxt("autc.txt",acfOfVecAbs[lagVal:],delimiter=',')
    return same,lagVal



def ksTestOneColumn(colVec,lag):
    """

    :param colVec: a vector of data
    :param lag: auto-correlation length
    :return:
    """
    colVecSelected=colVec[::lag]

    lengthTmp=len(colVecSelected)
    if lengthTmp%2==1:
        lengthTmp-=1
    lenPart=int(lengthTmp/2)

    colVecToCompute=colVecSelected[-lengthTmp:]

    #ks test
    selectedVecPart0=colVecToCompute[:lenPart]
    selectedVecPart1=colVecToCompute[lenPart:]
    result=ks_2samp(selectedVecPart0,selectedVecPart1)
    return result.pvalue,result.statistic, lenPart*2

def combineData(dataDir,startingRowFraction):
    """

    :param dataDir:
    :return:
    """
    sorted_dataFilesToRead=sort_data_files_by_sweepEnd(dataDir)
    startingFileInd,startingVecPosition=parseSummaryU_Dist()
    if startingFileInd<0:
        #we guess that the equilibrium starts at this file
        startingFileInd=len(sorted_dataFilesToRead)-lastFileNum

    startingFileName=sorted_dataFilesToRead[startingFileInd]
    with open(startingFileName,"rb") as fptr:
        inArrStart=pickle.load(fptr)
    in_nRowStart=len(inArrStart)

    if startingVecPosition<0:
        #we guess equilibrium starts at this position
        startingVecPosition=int(in_nRowStart*startingRowFraction)

    arr=inArrStart[startingVecPosition:]
    #read the rest of the pkl files
    for pkl_file in sorted_dataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            inArr=pickle.load(fptr)
        arr=np.append(arr,inArr)

    return arr,startingFileInd,startingVecPosition

def checkUDataFilesForOneT(UData_dir, startingRowFraction):
    """

    :param UData_dir:
    :param startingRowFraction:
    :return:
    """
    # U_sortedDataFilesToRead=sort_data_files_by_sweepEnd(UData_dir)
    # if len(U_sortedDataFilesToRead)==0:
    #     print("no data for U.")
    #     exit(0)
    #
    # startingFileInd,startingVecPosition=parseSummaryU_Dist()
    # if startingFileInd<0:
    #     #we guess that the equilibrium starts at this file
    #     startingFileInd=len(U_sortedDataFilesToRead)-lastFileNum
    # startingFileName=U_sortedDataFilesToRead[startingFileInd]
    #
    # with open(startingFileName,"rb") as fptr:
    #     inArrStart=pickle.load(fptr)
    #
    # in_nRowStart=len(inArrStart)
    # if startingVecPosition<0:
    #     #we guess equilibrium starts at this position
    #     startingVecPosition=int(in_nRowStart*startingRowFraction)
    # arr=inArrStart[startingVecPosition:]
    #
    #
    # #read the rest of the pkl files
    # for pkl_file in U_sortedDataFilesToRead[(startingFileInd+1):]:
    #     # print("reading: "+str(pkl_file))
    #     with open(pkl_file,"rb") as fptr:
    #         inArr=pickle.load(fptr)
    #         # print("len(inArr)="+str(len(inArr)))
    #     arr=np.append(arr,inArr)

    # print(arr[:100]/N)
    arr,startingFileInd,startingVecPosition=combineData(UData_dir,startingRowFraction)
    avg_Uarr=arr

    sameUTmp,lagUTmp=auto_corrForOneColumn(avg_Uarr)

    #if one lag==-1, then the auto-correlation is too large

    if sameUTmp==True or lagUTmp==-1:
        return [sameUTmp,lagUTmp,-1,-1,-1,-1,-1]


    pUTmp,statUTmp,lengthUTmp=ksTestOneColumn(avg_Uarr,lagUTmp)
    numDataPoints=lengthUTmp

    return [sameUTmp,lagUTmp,pUTmp,statUTmp,numDataPoints,startingFileInd,startingVecPosition]

def row2dist(arr_row):
    """

    :param arr_row: converted value of [x00,y00,x01,y01,x10,y10,x11,y11]
    :return:
    """
    x00,y00,x01,y01,x10,y10,x11,y11=arr_row

    A00=np.array([x00,y00])

    A01=np.array([x01,y01])

    A10=np.array([x10,y10])

    A11=np.array([x11,y11])

    d0=np.linalg.norm(A00-A01,ord=2)

    d1=np.linalg.norm(A01-A11,ord=2)

    d2=np.linalg.norm(A11-A10,ord=2)

    d3=np.linalg.norm(A10-A00,ord=2)

    d4=np.linalg.norm(A00-A11,ord=2)

    d5=np.linalg.norm(A10-A01,ord=2)

    return np.array([d0,d1,d2,d3,d4,d5])


def check_square(U_dist_dataDir,startingRowFraction):
    """

    :param converted_data_dir:
    :param startingRowFraction:
    :return:
    """
    # varNames=["x00","y00","x01","y01","x10","y10","x11","y11"]
    # dirName=[U_dist_dataDir+"/"+name for name in varNames]

    arr_x00,_,_=combineData(U_dist_dataDir+"/x00/",startingRowFraction)
    arr_x01,_,_=combineData(U_dist_dataDir+"/x01/",startingRowFraction)
    arr_x10,_,_=combineData(U_dist_dataDir+"/x10/",startingRowFraction)
    arr_x11,_,_=combineData(U_dist_dataDir+"/x11/",startingRowFraction)

    arr_y00,_,_=combineData(U_dist_dataDir+"/y00/",startingRowFraction)
    arr_y01,_,_=combineData(U_dist_dataDir+"/y01/",startingRowFraction)
    arr_y10,_,_=combineData(U_dist_dataDir+"/y10/",startingRowFraction)
    arr_y11,_,_=combineData(U_dist_dataDir+"/y11/",startingRowFraction)

    data_arr=np.array([
        arr_x00,arr_y00,arr_x01,arr_y01,
        arr_x10,arr_y10,arr_x11,arr_y11
    ]).T

    dist_arr=np.apply_along_axis(row2dist,axis=1,arr=data_arr)
    # print(dist_arr[:2,:])
    dist_pVec=[]
    dist_statsVec=[]
    dist_lengthsVec=[]
    dist_lagVec=[]

    _,nCol=dist_arr.shape
    print("nCol="+str(nCol))
    for j in range(0,nCol):
        sameTmp,lagTmp=auto_corrForOneColumn(dist_arr[:,j])
        if sameTmp==True or lagTmp==-1:
            print("column "+str(j)+" not equilibrium")
            return [],[],[],[]
        dist_lagVec.append(lagTmp)
        pTmp,statTmp,lengthTmp=ksTestOneColumn(dist_arr[:,j],lagTmp)
        dist_pVec.append(pTmp)
        dist_statsVec.append(statTmp)
        dist_lengthsVec.append(lengthTmp)
    return dist_pVec,dist_statsVec,dist_lengthsVec,dist_lagVec



startingRowFraction=1/2
UDataDir=U_dist_dataDir+"/U/"
# print(UDataDir)
sameVec=[]
lagVec=[]
pVec=[]
statVec=[]
numDataVec=[]

# print("before ")
print("checking U")
sameUTmp,lagUTmp,pUTmp,statUTmp,numDataPointsU,startingFileInd,startingVecPosition=checkUDataFilesForOneT(UDataDir,startingRowFraction)


print("lagU="+str(lagUTmp))
sameVec.append(sameUTmp)
lagVec.append(lagUTmp)
pVec.append(pUTmp)
statVec.append(statUTmp)
numDataVec.append(numDataPointsU)




dist_pVec,dist_statsVec,dist_lengthsVec,dist_lagVec=check_square(U_dist_dataDir,startingRowFraction)
if len(dist_pVec)==0:
    exit(12)

lagVec.extend(dist_lagVec)

pVec.extend(dist_pVec)

statVec.extend(dist_statsVec)

numDataVec.extend(dist_lengthsVec)

print("statVec="+str(statVec))
print("pVec="+str(pVec))

print("lagVec="+str(lagVec))

def check_equilibrium(pVec,statVec):
    """

    :param pVec:
    :param statVec:
    :return:
    """
    eqVec=[]
    for j in range(0,len(pVec)):
        pTmp=pVec[j]
        statTmp=statVec[j]
        if statTmp<=0.1 or pTmp>=0.01:
            eqTmp=True
        else:
            eqTmp=False
        eqVec.append(eqTmp)

    return eqVec

eqVec=check_equilibrium(pVec,statVec)

numDataPoints=np.min(numDataVec)
lagMax=np.max(lagVec)
if np.all(eqVec) and numDataPoints>=200:
    if numDataPoints>=effective_data_num_required:
        newDataPointNum=0
    else:
        newDataPointNum=effective_data_num_required-numDataPoints

    msg="equilibrium\n" \
        +"lag="+str(lagMax)+"\n" \
        +"numDataPoints="+str(numDataPoints)+"\n" \
        +"startingFileInd="+str(startingFileInd)+"\n" \
        +"startingVecPosition="+str(startingVecPosition)+"\n" \
        +"newDataPointNum="+str(newDataPointNum)+"\n"

    with open(summary_U_distFile,"w+") as fptr:
        fptr.writelines(msg)
    exit(0)

#continue
continueMsg="continue\n"
if numDataPoints<200:
    #not enough data number

    continueMsg+="numDataPoints="+str(numDataPoints)+" too low\n"
    continueMsg+="lag="+str(lagMax)+"\n"
with open(summary_U_distFile,"w+") as fptr:
    fptr.writelines(continueMsg)
exit(0)