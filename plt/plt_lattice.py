import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
from decimal import Decimal

#This script loads csv data and plot lattice

if (len(sys.argv)!=2):
    print("wrong number of arguments")
    exit()
rowNum=0#int(sys.argv[1])
unitCellNum=int(sys.argv[1])

csvDataFolderRoot="../dataAll/dataAllUnitCell"+str(unitCellNum)+"/row"+str(rowNum)+"/csvOutAll/"
inCsvFile="../V_inv_12_6Params.csv"

TVals=[]
TFileNames=[]
def format_using_decimal(value):
    # Convert the float to a Decimal using string conversion to avoid precision issues
    decimal_value = Decimal(str(value))
    # Remove trailing zeros and ensure fixed-point notation
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

for TFile in glob.glob(csvDataFolderRoot+"/T*"):

    matchT=re.search(r"T(\d+(\.\d+)?)",TFile)
    # if float(matchT.group(1))<1:
    #     continue

    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))



sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]

def rotationMatrix(x00,y00,x01,y01):
    """

    :param x00:
    :param y00:
    :param x01:
    :param y01:
    :return:
    """

    r=np.sqrt((x01-x00)**2+(y01-y00)**2)

    sin_theta=(y01-y00)/r
    cos_theta=(x01-x00)/r

    R=np.array([
        [cos_theta,sin_theta],
        [-sin_theta,cos_theta]
    ],dtype=float)

    return R

def convert_row(arr_row):
    """

    :param arr_row: [x00, y00, x01, y01, x10, y10, x11, y11]
    :return:
    """
    x00, y00, x01, y01, x10, y10, x11, y11=arr_row

    R=rotationMatrix(x00,y00,x01,y01)

    A01Tilde=np.array([x00,y00])+R@np.array([x01-x00,y01-y00])



    A10Tilde=np.array([x00,y00])+R@np.array([x10-x00,y10-y00])

    A11Tilde=np.array([x00,y00])+R@np.array([x11-x00,y11-y00])

    arr_row_new=np.array([x00,y00,A01Tilde[0],A01Tilde[1],A10Tilde[0],A10Tilde[1],A11Tilde[0],A11Tilde[1]])

    return arr_row_new
def pltU_dist(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return: U plots, U mean, U var, dist plots, dist mean, dist var
    """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    TVal=float(matchT.group(1))
    TStr=format_using_decimal(TVal)
    U_distPath=oneTFile+"/U_dist/U_distData.csv"
    df=pd.read_csv(U_distPath)

    # filtered_df = df[(df['U'] > -1000) & (df['U'] < 100)]

    UVec=np.array(df.iloc[:,0])


    print("T="+str(TVal)+", data num="+str(len(UVec)))

    #U part
    meanU=np.mean(UVec)
    meanU2=meanU**2

    varU=np.var(UVec,ddof=1)
    sigmaU=np.sqrt(varU)
    UConfHalfLength=np.sqrt(varU/len(UVec))
    nbins=100
    fig=plt.figure()
    axU=fig.add_subplot()
    (n0,_,_)=axU.hist(UVec,bins=nbins)

    meanUStr=str(np.round(meanU,4))
    print("T="+str(TVal)+", E(U)="+meanUStr)
    sigmaUStr=str(np.round(sigmaU,4))

    axU.set_title("T="+str(TVal))
    axU.set_xlabel("$U$")
    axU.set_ylabel("#")
    xPosUText=(np.max(UVec)-np.min(UVec))*1/2+np.min(UVec)
    yPosUText=np.max(n0)*2/3
    axU.text(xPosUText,yPosUText,"mean="+meanUStr+"\nsd="+sigmaUStr)
    plt.axvline(x=meanU,color="red",label="mean")
    axU.text(meanU*1.1,0.5*np.max(n0),str(meanU)+"$\\pm$"+str(sigmaU),color="red")
    axU.hlines(y=0,xmin=meanU-sigmaU,xmax=meanU+sigmaU,color="green",linewidth=15)

    plt.legend(loc="best")

    EHistOut="T"+str(TVal)+"UHist.png"
    plt.savefig(oneTFile+"/"+EHistOut)

    plt.close()

    ### test normal distribution for mean U
    #block mean
    USelectedAll=UVec

    def meanPerBlock(length):
        blockNum=int(np.floor(len(USelectedAll)/length))
        UMeanBlock=[]
        for blkNum in range(0,blockNum):
            blkU=USelectedAll[blkNum*length:(blkNum+1)*length]
            UMeanBlock.append(np.mean(blkU))
        return UMeanBlock

    fig=plt.figure(figsize=(20,20))
    fig.tight_layout(pad=5.0)
    lengthVals=[2,5,7,10]
    for i in range(0,len(lengthVals)):
        l=lengthVals[i]
        UMeanBlk=meanPerBlock(l)
        ax=fig.add_subplot(2,2,i+1)
        (n,_,_)=ax.hist(UMeanBlk,bins=100,color="aqua")
        xPosTextBlk=(np.max(UMeanBlk)-np.min(UMeanBlk))*1/7+np.min(UMeanBlk)
        yPosTextBlk=np.max(n)*3/4
        meanTmp=np.mean(UMeanBlk)
        meanTmp=np.round(meanTmp,3)
        sdTmp=np.sqrt(np.var(UMeanBlk))
        sdTmp=np.round(sdTmp,3)
        ax.set_title("Bin Length="+str(l))
        ax.text(xPosTextBlk,yPosTextBlk,"mean="+str(meanTmp)+", sd="+str(sdTmp))
    fig.suptitle("T="+str(TVal))
    plt.savefig(oneTFile+"/T"+str(TVal)+"UBlk.png")
    plt.close()

    lattice_arr=np.array(df.iloc[:,1:])
    converted_lattice_arr=np.apply_along_axis(convert_row,axis=1,arr=lattice_arr)

    x00Array=converted_lattice_arr[:,0]
    x01Array=converted_lattice_arr[:,2]
    x10Array=converted_lattice_arr[:,4]
    x11Array=converted_lattice_arr[:,6]

    y00Array=converted_lattice_arr[:,1]
    y01Array=converted_lattice_arr[:,3]
    y10Array=converted_lattice_arr[:,5]
    y11Array=converted_lattice_arr[:,7]

    lattice_x00Array=x00Array-x00Array
    lattice_x01Array=x01Array-x00Array
    lattice_x10Array=x10Array-x00Array
    lattice_x11Array=x11Array-x00Array

    lattice_y00Array=y00Array-y00Array
    lattice_y01Array=y01Array-y00Array
    lattice_y10Array=y10Array-y00Array
    lattice_y11Array=y11Array-y00Array


    x00_avg=np.mean(lattice_x00Array)
    x01_avg=np.mean(lattice_x01Array)
    x10_avg=np.mean(lattice_x10Array)
    x11_avg=np.mean(lattice_x11Array)

    y00_avg=np.mean(lattice_y00Array)
    y01_avg=np.mean(lattice_y01Array)
    y10_avg=np.mean(lattice_y10Array)
    y11_avg=np.mean(lattice_y11Array)

    A00_avg=np.array([x00_avg,y00_avg])
    A01_avg=np.array([x01_avg,y01_avg])
    A10_avg=np.array([x10_avg,y10_avg])
    A11_avg=np.array([x11_avg,y11_avg])

    # print("A00_avg="+str(A00_avg))
    # print("A01_avg="+str(A01_avg))
    # print("A10_avg="+str(A10_avg))
    # print("A11_avg="+str(A11_avg))
    dist_A00_A01=np.linalg.norm(A00_avg-A01_avg,ord=2)
    dist_A01_A11=np.linalg.norm(A01_avg-A11_avg,ord=2)

    dist_A11_A10=np.linalg.norm(A11_avg-A10_avg,ord=2)
    dist_A10_A00=np.linalg.norm(A10_avg-A00_avg,ord=2)

    dist_A00_A11=np.linalg.norm(A00_avg-A11_avg,ord=2)
    dist_A10_A01=np.linalg.norm(A10_avg-A01_avg,ord=2)

    print("dist A00 A01="+str(dist_A00_A01))
    print("dist A01 A11="+str(dist_A01_A11))
    print("dist A11 A10="+str(dist_A11_A10))
    print("dist A10 A00="+str(dist_A10_A00))
    print("dist A00 A11="+str(dist_A00_A11))
    print("dist A10 A01="+str(dist_A10_A01))

    plt.figure()
    plt.scatter(x00_avg,y00_avg,color="black",label="A00")
    plt.text(x00_avg *1.4, y00_avg +y11_avg*0.1, "A00", fontsize=9, color="black")

    plt.scatter(x01_avg,y01_avg,color="black",label="A01")
    plt.text(x01_avg *0.9, y00_avg +y11_avg*0.1, "A01", fontsize=9, color="black")

    plt.scatter(x10_avg,y10_avg,color="black",label="A10")
    plt.text(x10_avg *0.9, y10_avg*0.9, "A10", fontsize=9, color="black")


    plt.scatter(x11_avg,y11_avg,color="black",label="A11")
    plt.text(x11_avg *0.9, y11_avg*0.9, "A11", fontsize=9, color="black")
    plt.title("T="+TStr)
    # plt.legend(loc="best")
    plt.savefig(oneTFile+"/T"+str(TVal)+"lattice.png")
    plt.close()


    return [meanU,varU,UConfHalfLength]


UMeanValsAll=[]
UVarValsAll=[]
UConfHalfLengthAll=[]
tStatsStart=datetime.now()
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    meanU,varU,UConfHalfLength=pltU_dist(oneTFile)
    UMeanValsAll.append(meanU)
    UVarValsAll.append(varU)
    UConfHalfLengthAll.append(UConfHalfLength)


UMeanValsAll=np.array(UMeanValsAll)
UVarValsAll=np.array(UVarValsAll)
UConfHalfLengthAll=np.array(UConfHalfLengthAll)

sortedTVals=np.array(sortedTVals)

TInds=np.where(sortedTVals<100)
TToPlt=sortedTVals[TInds]

######################################################
#plt E(U)
fig, ax = plt.subplots()
ax.errorbar(TToPlt,UMeanValsAll[TInds],yerr=UConfHalfLengthAll,fmt='o',color="black", ecolor='r', capsize=5,label='mc')
# EVVals=[EV(T) for T in interpolatedTVals]
# ax.plot(interpolatedTVals,EVVals,color="green",label="theory")
ax.set_xlabel('$T$')
ax.set_ylabel("E(U)")
ax.set_title("E(U)")
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/EU.png")
plt.close()
#######################################################