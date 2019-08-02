# -*- coding: utf-8 -*-
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from openpyxl import Workbook
import copy
#设置流的区域分布
def setFlowP():
    Ptemp = [[0 for m in range(K[0])] for j in range(K[0])]
    for i in range(K[0]):
        tmp = np.random.rand(K[0])
        sumTmp = tmp.sum()
        for j in range(K[0]):
            Ptemp[i][j] = tmp[j]/sumTmp
    return Ptemp

#计算平均到达率
def updateContollerPara(variables):

    for m in range(NumOfLayers):
        g_1m = g11
        for n in range(0, m):
            g_1m = g_1m * G[n]
        #print('g_1m',g_1m)
        for k in range(K[m]): #for controller c_mk
            nlArrRate = 0
            nlArrRateTmp = 0
            lArrRateTmp = 0
            for i in range(K[0]):
                for j in range(K[0]):
                    #non-local request arrive rate
                    #print('test2',g_1m[i][k],i,k,m)
                    if g_1m[i,k] != 0:
                        if dm[i][j] >= (m+1):
                             #print(i,j,m)
                             # try:
                             #    nlArrRateTmp += variables[i][j][m]*ProbOfFlowDist[i][j]*iniRate[i]
                             # except TypeError:
                             #    print(variables[i][j][m],ProbOfFlowDist[i][j],iniRate[i])
                             # else:
                             #    print('normal')
                             nlArrRateTmp += variables[i][j][m] * ProbOfFlowDist[i][j] * iniRate[i]



                    #print(nlArrRateTmp)

                    #local request arrive rate
                    tmp = 0
                    for p in range(dm[i][j]):
                        tmp += variables[i][j][p]
                    tmp = 1 - tmp
                    lArrRateTmp += tmp*ProbOfFlowDist[i][j]*iniRate[i]
            Controllers[m][k]['nlArrRate'] = nlArrRateTmp
            Controllers[m][k]['lArrRate'] = lArrRateTmp
            Controllers[m][k]['nonlocalTime'] = 1/(Controllers[m][k]['nlSerRate']-Controllers[m][k]['nlArrRate'])
    #print(Controllers)
            #Controllers[m][k]['l']
#根据某一流分布以及决策地点分布，计算平均时延
#input: flowdistribution
def caculateAverLatency(variablesCode,MaxValue):
    T = [[0 for i in range(K[0])] for j in range(K[0])] #各个流决策时延集合
    variables = decode(variablesCode,dm) #决策点变量
    updateContollerPara(variables) #根据决策点更新控制器状态

    for i in range(K[0]):
        for j in range(K[0]):
            t_u = 0
            t_d = 0
            #calculte t_u = the time of upwarding

            for m in range(dm[i][j]):
                if variables[i][j][m] == 1:
                    tmp1 = 0 #
                    g_1m = g11
                    for n in range(0, m):
                        g_1m = g_1m * G[n]
                    for x in range(K[m]):
                        if g_1m[i,x] == 1:
                            tmp1 = Controllers[m][x]['nonlocalTime']
                            if tmp1 <= 0:
                                return MaxValue-0.1

                    tmp2 = 0 #the popagation time between layers
                    for x in range(m+1):
                        tmp2 += LatencyBeLayers[m]
                    t_u = tmp1 + tmp2
                    #print("")
            #calculate t_d

            T[i][j] = t_u + t_d

    averT = 0
    for i in range(K[0]):
        for j in range(K[0]):
            averT += T[i][j]
    averT = averT/(K[0]*K[0])
    return averT

#遗传算法

def encode(variables):
    variablesCode = []
    for i in range(K[0]):
        for j in range(K[0]):
            variablesCode.extend(variables[i][j])
    return variablesCode


def decode(variablesCode,dm):
    variables =  [[list() for col in range(K[0])] for row in range(K[0])]
    cursor = 0
    for i in range(K[0]):
        for j in range(K[0]):
            variables[i][j].extend(variablesCode[cursor:(cursor+dm[i][j])])
            cursor += dm[i][j]
    return variables

def GetInitialPopulat(N,variables, dm):
    P0 = []
    for k in range(N):
        variablesTmp = copy.deepcopy(variables)
        #print('beforeset',variablesTmp)
        #print('beforeset', variables)
        for i in range(K[0]):
            for j in range(K[0]):
                tmp = random.randint(0,dm[i][j]-1)
                variablesTmp[i][j][tmp] = 1
        #print('before decode variablestmp',variablesTmp)
        # print('afterdecode', decode(encode(variablesTmp), dm))
        variablesTmp = encode(variablesTmp)
        P0.append(variablesTmp)

    # for i in range(20):
    #     print(decode(P0[i],dm))
    return P0

def GetAverLatencyList(P):
    averLatency = []
    for variables in P:
        updateContollerPara(variables)
        averLatency.append(caculateAverLatency(variables))
    return averLatency

def sortPk(Pk,Fitness):
    for i in range(len(Fitness)):
        for j in range(i):
            if Fitness[i] >Fitness[j]:
                Fitness.insert(j,Fitness.pop(i))
                Pk.insert(j, Pk.pop(i))
    return [Pk,Fitness]

# def insert_sort(ilist):
#     for i in range(len(ilist)):
#         for j in range(i):
#             if ilist[i] > ilist[j]:
#                 ilist.insert(j, ilist.pop(i))
#                 break
#     return ilist
#
# ilist = insert_sort([4, 5, 6, 7, 3, 2, 10, 9, 8])
# print(ilist)
# test=1
def Iteration(Pk,Fitness,N,LenthChromo,pc,pm):

    Mk1 = []         #选择后的种群
    Pk2 = []
    Mk2 = []
    Fitness2 = []
    randomList = []
    FitnessNormal = []
    FitnessAccum = []
    sortedPk,sortedFitness = sortPk(Pk,Fitness)
    for i in range(N):
        if i < 0.1*N:
            Mk1.append(sortedPk[i])
        else:
            Pk2.append(sortedPk[i])
            Fitness2.append(sortedFitness[i])
    fitness_sum = sum(Fitness2)
    #selection process
    N2 = len(Fitness2)
    for k in range(N2): #概率归一化
        FitnessNormal.append(Fitness2[k]/fitness_sum)

    for k in range(N2):
        FitnessAccum.append(sum(FitnessNormal[:k+1]))
        randomList.append(random.random())

    randomList.sort()
    cursorFitAccum = 0
    cursorMk = 0
    #print(FitnessNormal)
    #print(FitnessAccum)
    #print(randomList)
    while cursorMk < N2:
        #print(cursorMk,cursorFitAccum)
        if randomList[cursorMk] < FitnessAccum[cursorFitAccum]:
            Mk2.append(Pk2[cursorFitAccum])
            cursorMk += 1
        else:
            cursorFitAccum += 1
    #crossover
    for k in range(N2-1):
        if random.random() < pc :
            cpoint = random.randint(0, LenthChromo)
            tmp1 = []
            tmp2 = []
            tmp1.extend(Mk2[k][0:cpoint])
            tmp1.extend(Mk2[k+1][cpoint:LenthChromo])
            tmp2.extend(Mk2[k+1][0:cpoint])
            tmp2.extend(Mk2[k][cpoint:LenthChromo])
            Mk2[k] = tmp1[:]
            Mk2[k+1] = tmp2[:]

    #mutation
    for k in range(N2):
        if(random.random()<pm):
            #print('before:', Mk[k])
            mktmp = decode(Mk2[k], dm)
            mx = random.randint(0,K[0]-1)
            my = random.randint(0,K[0]-1)
            #mpoint = random.randint(0,lenth()-1)

            #print(mktmp)
            lenth = len(mktmp[mx][my])
            mpointt = random.randint(0,lenth-1)
            if mktmp[mx][my][mpointt] == 0:
                mktmp[mx][my][mpointt] = 1
                for i in range(lenth):
                    if i != mpointt:
                        if mktmp[mx][my][i] ==1:
                            mktmp[mx][my][i] = 0
            else:
                if lenth !=1:
                    mktmp[mx][my][mpointt] = 0
                    randomtmp = random.randint(0,lenth-2)
                    if randomtmp != mpointt:
                        mktmp[mx][my][randomtmp] = 1
                    else:
                        mktmp[mx][my][randomtmp+1] = 1

            #print(mktmp)
            Mk2[k] = copy.deepcopy(encode(mktmp))
            #print('after:',Mk[k])
            # try:
            #     if Mk[k][mpoint] == 1:
            #         Mk[k][mpoint] = 0
            #     else:
            #         Mk[k][mpoint] = 1
            # except IndexError:
            #     print(k,mpoint)
            # else:
            #     print("normal")


    Mk1.extend(Mk2)
    # print('Mk1',Mk1)
    # print('Mk2',Mk2)
    #print('Pknext:',Pknext)

    return Mk1


def calculateFitness(P,N,MaxValue):
    Fitness = []
    for k in range(N):
        # judge = False
        # while judge == False:
        averlatency = caculateAverLatency(P[k],MaxValue)
        # if judge == False:
        #     pknew = GetInitialPopulat(1,variables, dm)[0]
        #     #print('here2',pknew)
        #     P[k] = pknew[:]
        #     #print('here',P[k])
        if averlatency<=0:
            fitness = 0.1
            # print('here',fitness)
        else:
            fitness = MaxValue - averlatency
            # print('here', fitness,averlatency,MaxValue-fitness)
        #else:
        #fitness = MaxValue - averlatency
        Fitness.append(fitness)
    return Fitness

def getBest(P,Fitness,NumberPop):
    bestfitness = Fitness[0]
    bestDistribution = P[0].copy()
    #print(len(Fitness),NumberPop)
    for k in range(NumberPop):
        #print(bestfitness)
        if Fitness[k] > bestfitness:
            bestfitness = Fitness[k]
            bestDistribution = P[k].copy()


    return [bestDistribution,bestfitness]

def plotIterCurv(MaxIteration,result):
    X = [i for i in range(MaxIteration)]
    Y = [result[i][0] for i in range(MaxIteration)]
    plt.plot(X,Y)
    plt.show()

def GA(variables,dm):
#initialization
    NumberPop = 80
    MAXITERATION =  80
    pc = 0.75
    pm = 0.4
    #MaxValue = 10000
    result = []
    Pk = GetInitialPopulat(NumberPop,variables,dm)
    # for i in range(20):
    #     print(decode(Pk[i], dm))
    LenthChromo = len(encode(COMPARISON))
    bestfitness = MaxValue-0.1

    Fitness = []
    encodeCOMPARISON = encode(COMPARISON)
    bestDistribution = encodeCOMPARISON
    retrytimes1 = 0
    retrytimes2 = 0

###############################version1
    # for k in range(MAXITERATION):
    #     # 如果没有招到最优解
    #     Fitness = calculateFitness(Pk, NumberPop, MaxValue)
    #     bestDistribution, bestfitness = getBest(Pk, Fitness, NumberPop)
    #     print(bestfitness)
    #     while retrytimes1 < 10 and (bestDistribution == encodeCOMPARISON or bestfitness == 0.1):
    #         print('here')
    #         Pk = GetInitialPopulat(NumberPop, variables, dm)
    #         insertp = random.randint(0, NumberPop - 1)
    #         Pk[insertp] = encode(COMPARISON)
    #         Fitness = calculateFitness(Pk, NumberPop, MaxValue)
    #         bestDistribution, bestfitness = getBest(Pk, Fitness, NumberPop)
    #         retrytimes1 += 1
    #         print(retrytimes1)
    #     retrytimes1 = 0
    #
    #     x = MaxValue - bestfitness
    #     y = decode(bestDistribution, dm)
    #     result.append([x, y])
    #     Pk = Iteration(Pk, Fitness, NumberPop, LenthChromo, pc, pm)


#############################version2


    # while True:
    #     Pk = GetInitialPopulat(NumberPop, variables, dm)
    #     insertp = random.randint(0,NumberPop-1)
    #     Pk[insertp] = encode(COMPARISON)
    #     for k in range(MAXITERATION):
    #         # 如果没有招到最优解
    #         while retrytimes1 < 3 and (bestDistribution == encodeCOMPARISON or bestfitness == MaxValue - 0.1):
    #             Pk = GetInitialPopulat(NumberPop, variables, dm)
    #             insertp = random.randint(0, NumberPop - 1)
    #             Pk[insertp] = encode(COMPARISON)
    #             Fitness = calculateFitness(Pk, NumberPop, MaxValue)
    #             bestDistribution, bestfitness = getBest(Pk, Fitness, NumberPop)
    #             retrytimes1 += 1
    #             print('retrytimes1',retrytimes1)
    #
    #         x = MaxValue - bestfitness
    #         y = decode(bestDistribution, dm)
    #         result.append([x, y])
    #         Pk = Iteration(Pk, Fitness, NumberPop, LenthChromo, pc, pm)
    #
    #
    #     if retrytimes2 > 10 or (result[-1][1] != COMPARISON and result[-1][0] != MaxValue - 0.1):
    #         break
    #     retrytimes2 += 1
    #     print("retrytimes2",retrytimes2)

##################################

    while True:
        Pk = GetInitialPopulat(NumberPop, variables, dm)
        insertp = random.randint(0,NumberPop-2)
        Pk[insertp] = encode(COMPARISON)
        Pk[insertp+1] = encode(COMPARISON2)
        for k in range(MAXITERATION):
            # 如果没有招到最优解
            Fitness = calculateFitness(Pk, NumberPop, MaxValue)
            bestDistribution, bestfitness = getBest(Pk, Fitness, NumberPop)
            x = MaxValue - bestfitness
            y = decode(bestDistribution, dm)
            result.append([x, y])
            Pk = Iteration(Pk, Fitness, NumberPop, LenthChromo, pc, pm)


        if retrytimes2 > 5 or (result[-1][1] != COMPARISON and result[-1][0] != MaxValue - 0.1):
            break
        retrytimes2 += 1
        print("retrytimes2",retrytimes2)

    return result[-1][0]


MaxValue = 1000
#*******控制层相关初始化*****
NumOfLayers = 4

#多层结构的映射关系 需要变化
K = [12, 4, 2, 1]
g11 = np.identity(K[0])
g12 = np.matrix(
    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
     [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
g23 = np.matrix([[1, 0], [1, 0], [0, 1], [0, 1]])
g34 = np.matrix([[1], [1]])
G = [g12, g23, g34]
# g13 = g12 * g23
# g14 = g13 * g34
# print(g13)
LatencyBeLayers = [0.3,0.3,0.3,0.3]


#每个控制器的控制域
alph = 0.6
domainSizeOflayer1 = np.matrix([1000 for i in range(K[0])])

tmp = G[0]
H = []
for i in range(NumOfLayers):
    if i == 0:
        H.append(domainSizeOflayer1)
    else:
        tmp1 = alph**i*domainSizeOflayer1*tmp
        H.append(tmp1)
        if i < NumOfLayers-1:
            tmp = tmp*G[i]
#控制的数据结构初始化
CAPACITY = 2**31
Controllers = [list() for i in range(NumOfLayers)]
for m in range(NumOfLayers): #initialization
    for k in range(K[m]):
        lSerRate = CAPACITY/(H[m][0,k]**2)
        nlSerRate = CAPACITY/((H[m].sum(axis=1)[0,0])**2)
        Controllers[m].append({'localTime':0,'nonlocalTime':0,'domainSize':H[m][0,k], 'Capacity':CAPACITY,'lArrRate':0,'lSerRate':lSerRate,'nlArrRate':0,'nlSerRate':nlSerRate})
        #print('This is controller',(m+1),(k+1), Controllers[m][k])

#*****流相关初始化*******

iniRate = [1 for m in range(K[0])]
# = [10,10,10,10,10,10,10,10,10,10,10,10]
ProbOfFlowDist = [[0 for m in range(K[0])] for j in range(K[0])]
#ProbOfFlowDist = setFlowP()

for i in range(K[0]):
    tmp = 1/K[0]
    for j in range(K[0]):
        ProbOfFlowDist[i][j] = tmp
#计算每条流在该架构下的最大决策距离
dm = [[0 for i in range(K[0])] for i in range(K[0])]
for i in range(K[0]):
    for j in range(K[0]):
        sumTmp = 0
        for n in range(1, NumOfLayers+1):
            if n == 1:
                tmp = np.identity(K[0])
                sumTmp += (int)(tmp[i].dot(tmp[j].T))
            else:
                g_1n = 1
                for m in range(0,n-1):
                    g_1n = g_1n*G[m]
                a = (int)(g_1n[i].dot(g_1n[j].T))
                sumTmp += (int)(g_1n[i].dot(g_1n[j].T))
        dm[i][j] = NumOfLayers - sumTmp + 1

#初始化决策变量
variables = [[list() for col in range(K[0])] for row in range(K[0])]
COMPARISON = [[list() for col in range(K[0])] for row in range(K[0])]
COMPARISON2 = [[list() for col in range(K[0])] for row in range(K[0])]

for i in range(K[0]):
    for j in range(K[0]):
        for k in range(dm[i][j]):
            if k == dm[i][j]-1:
                COMPARISON[i][j].append(1)
            else:
                COMPARISON[i][j].append(0)
for i in range(K[0]):
    for j in range(K[0]):
        for k in range(dm[i][j]):
            if k == 0:
                COMPARISON2[i][j].append(1)
            else:
                COMPARISON2[i][j].append(0)

for i in range(K[0]):
    for j in range(K[0]):
        for k in range(dm[i][j]):
                variables[i][j].append(0)


# #*****优化算法*****
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set(title='example',ylabel='Average Latency',xlabel='request rate per domains')
# ax.plot(X,Y)
# ax.plot(X,Y2)
#
# plt.show()
# a =GA(variables,dm)
# b =caculateAverLatency(encode(COMPARISON))
# print("optimal",a)
# print("comparison",b)
# X=[1,2,3,4,5,5,6,7]
# Y=[1,2,3,4,5,6,7,4]
# Y2=[1,2,3,4,5,5,6,7]
# test algorithm
#GA(variables,dm)



NUMBERS = 60
ResultHHA= []
ResultHA= []
ResultFA= []
Resultx = []
x = 1
X = []

numbers = 500

COMPARISONList = GetInitialPopulat(numbers,variables,dm)
ResultList = [[] for i in range(numbers+2)]

for i in range(NUMBERS):
    for k in range(K[0]):
        iniRate[k]+=x
    X.append(iniRate[0])
    ResultList[0].append(caculateAverLatency(encode(COMPARISON),MaxValue))
    ResultList[1].append(caculateAverLatency(encode(COMPARISON2),MaxValue))
    for k in range(numbers):
        ResultList[k+2].append(caculateAverLatency(COMPARISONList[k],MaxValue))



wb = Workbook()
ws=wb.active
ws.append(X)
for i in ResultList:
    ws.append(i)
    print('write into excel',i)
wb.save(r'G:\example7.xlsx')

for i in ResultList:
    plt.plot(X,i)

plt.show()




# for i in range(NUMBERS):
#
#     for k in range(K[0]):
#         iniRate[k]+=x
#     X.append(iniRate[0])
#     print(iniRate)
#     #ProbOfFlowDist = setFlowP()
#     ResultHA.append(caculateAverLatency(encode(COMPARISON),MaxValue))#对比
# #    ResultFlat.append(caculateAverLatency(encode(COMPARISON2)))
#     print(ResultHA)
#     ResultFA.append(caculateAverLatency(encode(COMPARISON2),MaxValue))
#     print(ResultFA)
#     ResultHHA.append(GA(variables,dm))#最优值，调用GA函数
#     print(ResultHHA)
#
# # print(ResultHA)
# # print(ResultHHA)
# # X = [i for i in range(NUMBERS)]
# # Y = ResultHHA
# print(X)
# Y2= ResultHA
# Y3 = ResultFA
# Y4 = Resultx
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.set(ylabel='Average Latency',xlabel='request rate per domains')
# # ax.plot(X,Y)
# # ax.plot(X,Y2)
# #
#
# # wb = Workbook()
# # ws=wb.active
# # ws.append(X)
# # # ws.append(Y)
# # ws.append(Y2)
# # ws.append(Y3)
# # wb.save(r'G:\example6.xlsx')
#
# # plt.plot(X,Y)
# plt.plot(X,Y2)
# plt.plot(X,Y3)
# plt.plot(X,Y4)
# plt.xlabel("request rate per domains")
# plt.ylabel("Average Latency")
# plt.show()