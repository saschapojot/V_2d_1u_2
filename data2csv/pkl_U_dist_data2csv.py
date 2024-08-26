import numpy as np
from datetime import datetime
import sys
import re
import glob
import os
import json
from pathlib import Path
import pandas as pd
import pickle
#this script extracts effective data from pkl files

if (len(sys.argv)!=2):
    print("wrong number of arguments")
    exit()


rowNum=0
unitCellNum=int(sys.argv[1])
rowDirRoot="../dataAll/dataAllUnitCell"+str(unitCellNum)+"/row"+str(rowNum)+"/"
obs_U_dist="U_dist"


#search directory
TVals=[]
TFileNames=[]
TStrings=[]
for TFile in glob.glob(rowDirRoot+"/T*"):
    # print(TFile)
    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",TFile)
    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))
        TStrings.append("T"+matchT.group(1))


#sort T values
sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]
sortedTStrings=[TStrings[ind] for ind in sortedInds]


def parseSummary(oneTFolder,obs_name):

    startingFileInd=-1
    startingVecPosition=-1
    lag=-1
    smrFile=oneTFolder+"/summary_"+obs_name+".txt"
    summaryFileExists=os.path.isfile(smrFile)
    if summaryFileExists==False:
        return startingFileInd,startingVecPosition,-1

    with open(smrFile,"r") as fptr:
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

        #match lag
        matchLag=re.search(r"lag=(\d+)",oneLine)
        if matchLag:
            lag=int(matchLag.group(1))
    return startingFileInd, startingVecPosition,lag



def sort_data_files_by_swEnd(oneTFolder,obs_name,varName):
    """

    :param oneTFolder: Txxx
    :param obs_name: data files sorted by sweepEnd
    :return:
    """

    dataFolderName=oneTFolder+"/"+obs_name+"_dataFiles/"+varName+"/"
    dataFilesAll=[]
    sweepEndAll=[]

    for oneDataFile in glob.glob(dataFolderName+"/*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"sweepEnd(\d+)",oneDataFile)
        if matchEnd:
            sweepEndAll.append(int(matchEnd.group(1)))


    endInds=np.argsort(sweepEndAll)
    # sweepStartSorted=[sweepStartAll[i] for i in startInds]
    sortedDataFiles=[dataFilesAll[i] for i in endInds]

    return sortedDataFiles



def combineData(dataDir,obs_name,varName,startingFileInd,startingVecPosition):
    """

    :param dataDir:
    :return:
    """
    sorted_dataFilesToRead=sort_data_files_by_swEnd(dataDir,obs_name,varName)



    startingFileName=sorted_dataFilesToRead[startingFileInd]
    with open(startingFileName,"rb") as fptr:
        inArrStart=pickle.load(fptr)




    arr=inArrStart[startingVecPosition:]
    #read the rest of the pkl files
    for pkl_file in sorted_dataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            inArr=pickle.load(fptr)
        arr=np.append(arr,inArr)

    return arr

def U_dist_data2csvForOneT(oneTFolder,oneTStr,startingFileInd,startingVecPosition,lag):
    TRoot=oneTFolder
    sortedUDataFilesToRead=sort_data_files_by_swEnd(TRoot,obs_U_dist,"U")
    # print(sortedUDataFilesToRead)
    # startingUFileName=sortedUDataFilesToRead[startingFileInd]
    #
    # with open(startingUFileName,"rb") as fptr:
    #     inUStart=pickle.load(fptr)
    #
    # UVec=inUStart[startingVecPosition:]
    # for pkl_file in sortedUDataFilesToRead[(startingFileInd+1):]:
    #     with open(pkl_file,"rb") as fptr:
    #         # print(pkl_file)
    #         in_UArr=pickle.load(fptr)
    #         UVec=np.append(UVec,in_UArr)
    UVec=combineData(TRoot,obs_U_dist,"U",startingFileInd,startingVecPosition)

    UVecSelected=UVec[::lag]



    arr_x00=combineData(TRoot,obs_U_dist,"x00",startingFileInd,startingVecPosition)
    arr_x00_selected=arr_x00[::lag]

    arr_x01=combineData(TRoot,obs_U_dist,"x01",startingFileInd,startingVecPosition)
    arr_x01_selected=arr_x01[::lag]

    arr_x10=combineData(TRoot,obs_U_dist,"x10",startingFileInd,startingVecPosition)
    arr_x10_selected=arr_x10[::lag]

    arr_x11=combineData(TRoot,obs_U_dist,"x11",startingFileInd,startingVecPosition)
    arr_x11_selected=arr_x11[::lag]

    arr_y00=combineData(TRoot,obs_U_dist,"y00",startingFileInd,startingVecPosition)
    arr_y00_selected=arr_y00[::lag]

    arr_y01=combineData(TRoot,obs_U_dist,"y01",startingFileInd,startingVecPosition)
    arr_y01_selected=arr_y01[::lag]


    arr_y10=combineData(TRoot,obs_U_dist,"y10",startingFileInd,startingVecPosition)
    arr_y10_selected=arr_y10[::lag]

    arr_y11=combineData(TRoot,obs_U_dist,"y11",startingFileInd,startingVecPosition)
    arr_y11_selected=arr_y11[::lag]

    dataArraySelected=np.array([UVecSelected,
                                arr_x00_selected,arr_y00_selected,
                                arr_x01_selected,arr_y01_selected,
                                arr_x10_selected,arr_y10_selected,
                                arr_x11_selected,arr_y11_selected,]).T

    colNamesAll=["U","x00","y00","x01","y01","x10","y10","x11","y11"]



    outCsvDataRoot=rowDirRoot+"/csvOutAll/"
    outCsvFolder=outCsvDataRoot+"/"+oneTStr+"/"+obs_U_dist+"/"
    Path(outCsvFolder).mkdir(parents=True, exist_ok=True)
    outCsvFile=outCsvFolder+"/"+obs_U_dist+"Data.csv"
    dfToSave=pd.DataFrame(dataArraySelected,columns=colNamesAll)
    dfToSave.to_csv(outCsvFile,index=False)


for k in range(0,len(sortedTFiles)):
    tStart=datetime.now()
    oneTFolder=sortedTFiles[k]
    oneTStr=sortedTStrings[k]

    startingfileIndTmp,startingVecIndTmp,lagTmp=parseSummary(oneTFolder,obs_U_dist)
    if startingfileIndTmp<0:
        print("summary file does not exist for "+oneTStr+" "+obs_U_dist)
        continue

    U_dist_data2csvForOneT(oneTFolder,oneTStr,startingfileIndTmp,startingVecIndTmp,lagTmp)
    tEnd=datetime.now()
    print("processed T="+str(sortedTVals[k])+": ",tEnd-tStart)


