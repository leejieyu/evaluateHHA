# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from openpyxl import Workbook
import copy


# 设置流的区域分布
def setFlowP():
    Ptemp = [[0 for m in range(K[0])] for j in range(K[0])]
    for i in range(K[0]):
        tmp = np.random.rand(K[0])
        sumTmp = tmp.sum()
        for j in range(K[0]):
            Ptemp[i][j] = tmp[j] / sumTmp
    return Ptemp


# 计算平均到达率
def updateContollerPara(variables):
    for m in range(NumOfLayers):
        g_1m = g11
        for n in range(0, m):
            g_1m = g_1m * G[n]
        # print('g_1m',g_1m)
        for k in range(K[m]):  # for controller c_mk
            nlArrRate = 0
            nlArrRateTmp = 0
            lArrRateTmp = 0
            for i in range(K[0]):
                for j in range(K[0]):
                    # non-local request arrive rate
                    # print('test2',g_1m[i][k],i,k,m)
                    if g_1m[i, k] != 0:
                        if dm[i][j] > (m+1):
                        # if dm[i][j] >= (m + 1):
                            # print(i,j,m)
                            # try:
                            #    nlArrRateTmp += variables[i][j][m]*ProbOfFlowDist[i][j]*iniRate[i]
                            # except TypeError:
                            #    print(variables[i][j][m],ProbOfFlowDist[i][j],iniRate[i])
                            # else:
                            #    print('normal')
                            nlArrRateTmp += variables[i][j][m] * ProbOfFlowDist[i][j] * RatePerDomain[i]

                    # print(nlArrRateTmp)

                    # local request arrive rate
                    # tmp = 0
                    # for p in range(dm[i][j]):# 这个求和结果必然=1啊。。。。
                    #     tmp += variables[i][j][p]
                    # tmp = 1 - tmp
                    # lArrRateTmp += tmp * ProbOfFlowDist[i][j] * iniRate[i]
                        if dm[i][j] == (m+1):
                            lArrRateTmp += variables[i][j][m] * ProbOfFlowDist[i][j] * RatePerDomain[i]

            Controllers[m][k]['nlArrRate'] = nlArrRateTmp
            Controllers[m][k]['lArrRate'] = lArrRateTmp
            Controllers[m][k]['nonlocalTime'] = 1 / (Controllers[m][k]['nlSerRate'] - Controllers[m][k]['nlArrRate'])
            Controllers[m][k]['localTime'] = 1 / (Controllers[m][k]['lSerRate'] - Controllers[m][k]['lArrRate'])
    # print(Controllers)
    # Controllers[m][k]['l']


# 根据某一流分布以及决策地点分布，计算平均时延
# input: flowdistribution
def caculateAverLatency(variablesCode, MaxValue):
    T = [[0 for i in range(K[0])] for j in range(K[0])]  # 各个流决策时延集合
    variables = decode(variablesCode, dm)  # 决策点变量
    updateContollerPara(variables)  # 根据决策点更新控制器状态

    for i in range(K[0]):
        for j in range(K[0]):
            t_u = 0
            t_d = 0
            # calculte t_u = the time of upwarding

            for m in range(dm[i][j]):
                if variables[i][j][m] == 1:
                    tmp1 = 0  #
                    g_1m = g11
                    for n in range(0, m):
                        g_1m = g_1m * G[n]
                    for x in range(K[m]):
                        if g_1m[i, x] == 1:
                            if m == dm[i][j]-1:
                                tmp1 = Controllers[m][x]['localTime']
                            else:
                                tmp1 = Controllers[m][x]['nonlocalTime']
                            if tmp1 <= 0:
                                print('the break down controller:',m,",",x,",",Controllers[m][x]['nlArrRate'],",",Controllers[m][x]['lArrRate'])
                                return MaxValue - 0.1

                    tmp2 = 0  # the popagation time between layers
                    for x in range(m + 1):
                        tmp2 += LatencyBeLayers[m]
                    t_u = tmp1 + tmp2
                    # print("")
            # calculate t_d

            T[i][j] = t_u + t_d

    averT = 0
    for i in range(K[0]):
        for j in range(K[0]):
            averT += T[i][j]
    averT = averT / (K[0] * K[0])
    return averT


# 遗传算法

def encode(variables):
    variablesCode = []
    for i in range(K[0]):
        for j in range(K[0]):
            variablesCode.extend(variables[i][j])
    return variablesCode


def decode(variablesCode, dm):
    variables = [[list() for col in range(K[0])] for row in range(K[0])]
    cursor = 0
    for i in range(K[0]):
        for j in range(K[0]):
            variables[i][j].extend(variablesCode[cursor:(cursor + dm[i][j])])
            cursor += dm[i][j]
    return variables


def GetInitialPopulat(N, variables, dm):
    P0 = []
    for k in range(N):
        variablesTmp = copy.deepcopy(variables)
        # print('beforeset',variablesTmp)
        # print('beforeset', variables)
        for i in range(K[0]):
            for j in range(K[0]):
                tmp = random.randint(0, dm[i][j] - 1)
                variablesTmp[i][j][tmp] = 1
        # print('before decode variablestmp',variablesTmp)
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


def sortPk(Pk, Fitness):
    for i in range(len(Fitness)):
        for j in range(i):
            if Fitness[i] > Fitness[j]:
                Fitness.insert(j, Fitness.pop(i))
                Pk.insert(j, Pk.pop(i))
    return [Pk, Fitness]


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
def Iteration(Pk, Fitness, N, LenthChromo, pc, pm):
    Mk1 = []  # 选择后的种群
    Pk2 = []
    Mk2 = []
    Fitness2 = []
    randomList = []
    FitnessNormal = []
    FitnessAccum = []
    sortedPk, sortedFitness = sortPk(Pk, Fitness)
    for i in range(N):
        if i < 0.1 * N:
            Mk1.append(sortedPk[i])
        else:
            Pk2.append(sortedPk[i])
            Fitness2.append(sortedFitness[i])
    fitness_sum = sum(Fitness2)
    # selection process
    N2 = len(Fitness2)
    for k in range(N2):  # 概率归一化
        FitnessNormal.append(Fitness2[k] / fitness_sum)

    for k in range(N2):
        FitnessAccum.append(sum(FitnessNormal[:k + 1]))
        randomList.append(random.random())

    randomList.sort()
    cursorFitAccum = 0
    cursorMk = 0
    # print(FitnessNormal)
    # print(FitnessAccum)
    # print(randomList)
    while cursorMk < N2:
        # print(cursorMk,cursorFitAccum)
        if randomList[cursorMk] < FitnessAccum[cursorFitAccum]:
            Mk2.append(Pk2[cursorFitAccum])
            cursorMk += 1
        else:
            cursorFitAccum += 1
    # crossover
    for k in range(N2 - 1):
        if random.random() < pc:

            while True:
                tmp1 = []
                tmp2 = []
                cpoint = random.randint(0, LenthChromo)
                # cpoint = 10
                # print(cpoint)
                tmp1.extend(Mk2[k][0:cpoint])
                tmp1.extend(Mk2[k + 1][cpoint:LenthChromo])
                tmp2.extend(Mk2[k + 1][0:cpoint])
                tmp2.extend(Mk2[k][cpoint:LenthChromo])
                # print("交换前1:",k,",",decode(Mk2[k],dm))
                # print("交换前2:",k+1,",",decode(Mk2[k + 1],dm))
                # print("交换后1:",decode(tmp1,dm))
                # print("交换后2:",decode(tmp2,dm))
                if check(decode(tmp1,dm)) == True:
                    # print("test1")
                    break
                # print("test2")

            Mk2[k] = tmp1[:]
            Mk2[k + 1] = tmp2[:]
            if (check(decode(Mk2[k],dm)) == False) or (check(decode(Mk2[k+1],dm))==False):
                print(cpoint)
                print(decode(Mk2[k],dm))
                print(decode(Mk2[k+1], dm))
                print("!!!!!!ERROR in crossover after!!!!!!!!!!!!!!!")
                os._exit(0)

    # mutation
    for k in range(N2):
        if (random.random() < pm):
            # print('before:', Mk[k])
            mktmp = decode(Mk2[k], dm)
            mx = random.randint(0, K[0] - 1)
            my = random.randint(0, K[0] - 1)
            # mpoint = random.randint(0,lenth()-1)

            # print(mktmp)
            lenth = len(mktmp[mx][my])
            mpointt = random.randint(0, lenth - 1)
            if mktmp[mx][my][mpointt] == 0:
                mktmp[mx][my][mpointt] = 1
                for i in range(lenth):
                    if i != mpointt:
                        if mktmp[mx][my][i] == 1:
                            mktmp[mx][my][i] = 0
            else:
                if lenth != 1:
                    mktmp[mx][my][mpointt] = 0
                    randomtmp = random.randint(0, lenth - 2)
                    if randomtmp != mpointt:
                        mktmp[mx][my][randomtmp] = 1
                    else:
                        mktmp[mx][my][randomtmp + 1] = 1

            # print(mktmp)
            Mk2[k] = copy.deepcopy(encode(mktmp))
            if check(decode(Mk2[k],dm)) == False:
                print("!!!!!!ERROR in mutation!!!!!!!!!!!!!!!")
                os._exit(0)
            # print('after:',Mk[k])
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
    # print('Pknext:',Pknext)

    return Mk1

def check(construct):
    for i in range(K[0]):
        for j in range(K[0]):
            # print(construct[i][j])
            if sum(construct[i][j]) != 1:
                # print(construct[i][j])
                return False
    return True


def calculateFitness(P, N, MaxValue):
    Fitness = []
    for k in range(N):
        # judge = False
        # while judge == False:
        averlatency = caculateAverLatency(P[k], MaxValue)
        # if judge == False:
        #     pknew = GetInitialPopulat(1,variables, dm)[0]
        #     #print('here2',pknew)
        #     P[k] = pknew[:]
        #     #print('here',P[k])
        if averlatency <= 0:
            fitness = 0.1
            # print('here',fitness)
        else:
            fitness = MaxValue - averlatency
            # print('here', fitness,averlatency,MaxValue-fitness)
        # else:
        # fitness = MaxValue - averlatency
        Fitness.append(fitness)
    return Fitness


def getBest(P, Fitness, NumberPop):
    bestfitness = Fitness[0]
    bestDistribution = P[0].copy()
    # print(len(Fitness),NumberPop)
    for k in range(NumberPop):
        # print(bestfitness)
        if Fitness[k] > bestfitness:
            bestfitness = Fitness[k]
            bestDistribution = P[k].copy()

    return [bestDistribution, bestfitness]


def plotIterCurv(MaxIteration, result):
    X = [i for i in range(MaxIteration)]
    Y = [result[i][0] for i in range(MaxIteration)]
    plt.plot(X, Y)
    plt.show()


def GA(variables, dm):
    # initialization
    NumberPop = 100
    MAXITERATION = 100
    pc = 0.75
    pm = 0.4
    # MaxValue = 10000
    result = []
    Pk = GetInitialPopulat(NumberPop, variables, dm)
    # for i in range(20):
    #     print(decode(Pk[i], dm))
    LenthChromo = len(encode(COMPARISON))
    bestfitness = MaxValue - 0.1

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
        insertp = random.randint(0, NumberPop - 3)
        Pk[insertp] = encode(COMPARISON)
        Pk[insertp + 1] = encode(COMPARISON2)
        if retrytimes2 > 0:
            Pk[insertp + 2] = encode(result[-1][1])
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
        print("retrytimes2", retrytimes2)

    return result[-1]


MaxValue = 1000
# *******控制层相关初始化*****
NumOfLayers = 4  #number of layers
RatioOfLayers = [2,2,2] # the ratio of controllers number between layers
NumOfL1 = 8 # the contorllers number of L1
LatencyBeLayers = [0.3 for i in range(NumOfLayers)]

#set the number of controllers in each layers
K = []
tmp = NumOfL1
for i in range(NumOfLayers):
    if i == 0:
        K.append(NumOfL1)
    else:
        tmp = tmp//RatioOfLayers[i-1]
        K.append(tmp)

# K = [12, 4, 2, 1]
print(K)
# 多层结构的映射关系 需要变化
g11 = np.identity(K[0])
G=[]
for i in range(NumOfLayers-1):
    rows = K[i]
    columns = K[i+1]
    structArray = np.zeros((rows,columns),dtype=np.int)
    # print(structArray)
    ratio = rows // columns
    for j in range(rows):
        column = j // ratio
        structArray[j][column] = 1
    print(structArray)
    G.append(np.matrix(structArray))
print(G)

# K = [12, 4, 2, 1]
# print(K)
# g12 = np.matrix(
#     [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
#      [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
# g23 = np.matrix([[1, 0], [1, 0], [0, 1], [0, 1]])
# g34 = np.matrix([[1], [1]])
# G = [g12, g23, g34]
# g13 = g12 * g23
# g14 = g13 * g34
# print(g13)


# 每个控制器的控制域
TotalHost = 12000
alph = 0.7
sizeOflayer1 = TotalHost/NumOfL1
domainSizeOflayer1 = np.matrix([sizeOflayer1 for i in range(K[0])])

tmp = G[0]
H = []
for i in range(NumOfLayers):
    if i == 0:
        H.append(domainSizeOflayer1)
    else:
        tmp1 = alph ** i * domainSizeOflayer1 * tmp
        H.append(tmp1)
        if i < NumOfLayers - 1:
            tmp = tmp * G[i]
print(H)
# 控制的数据结构初始化
CAPACITY = 2 ** 31
Controllers = [list() for i in range(NumOfLayers)]
for m in range(NumOfLayers):  # initialization
    for k in range(K[m]):
        belta =1
        belta = (H[m][0,0]/H[0][0,0]) ** 1.5
        # belta =(H[m].sum(axis=1)[0, 0]/H[0].sum(axis=1)[0, 0]) ** 2
        lSerRate = belta * CAPACITY / (H[m][0, k] ** 2)
        nlSerRate = belta * CAPACITY / ((H[m].sum(axis=1)[0, 0]) ** 2)
        Controllers[m].append(
            {'localTime': 0, 'nonlocalTime': 0, 'domainSize': H[m][0, k], 'Capacity': CAPACITY, 'lArrRate': 0,
             'lSerRate': lSerRate, 'nlArrRate': 0, 'nlSerRate': nlSerRate})
        # print('This is controller',(m+1),(k+1), Controllers[m][k])


# 计算每条流在该架构下的最大决策距离
dm = [[0 for i in range(K[0])] for i in range(K[0])]
for i in range(K[0]):
    for j in range(K[0]):
        sumTmp = 0
        for n in range(1, NumOfLayers + 1):
            if n == 1:
                tmp = np.identity(K[0])
                sumTmp += (int)(tmp[i].dot(tmp[j].T))
            else:
                g_1n = 1
                for m in range(0, n - 1):
                    g_1n = g_1n * G[m]
                a = (int)(g_1n[i].dot(g_1n[j].T))
                sumTmp += (int)(g_1n[i].dot(g_1n[j].T))
        dm[i][j] = NumOfLayers - sumTmp + 1

# 初始化决策变量
variables = [[list() for col in range(K[0])] for row in range(K[0])]
COMPARISON = [[list() for col in range(K[0])] for row in range(K[0])]
COMPARISON2 = [[list() for col in range(K[0])] for row in range(K[0])]

for i in range(K[0]):
    for j in range(K[0]):
        for k in range(dm[i][j]):
            if k == dm[i][j] - 1:
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
# print(COMPARISON)
# print(COMPARISON2)

# test = decode(encode(COMPARISON),dm)
#
# if test == COMPARISON:
#     print("NO")
# else:
#     print("YES")
#
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
# GA(variables,dm)


NUMBERS = 200
ResultHHA = []
ResultHA = []
ResultFA = []
ResultHHAD = []

X = []

numbers = 3000

COMPARISONList = GetInitialPopulat(numbers, variables, dm)
ResultList = [[] for i in range(numbers + 2)]

# *****流相关初始化******
RatePerHost = 10**-3
xaxis = RatePerHost / (10**-3) #横坐标的开始
cluster = TotalHost / NumOfL1
Ratetmp = cluster * RatePerHost
RatePerDomain = [Ratetmp for m in range(K[0])]
# RatePerDomain[7]= 8*Ratetmp

detaRatePerHost = 1 * RatePerHost
detax = detaRatePerHost / (10**-3) #横坐标的间隔
deta = cluster * detaRatePerHost

# = [10,10,10,10,10,10,10,10,10,10,10,10]
ProbOfFlowDist = [[0 for m in range(K[0])] for j in range(K[0])]
ProbOfFlowDist = setFlowP()
for i in range(K[0]):
    tmp = 1 / K[0]
    for j in range(K[0]):

        ProbOfFlowDist[i][j] = tmp
#
# for i in range(NUMBERS):
#     for k in range(K[0]):
#         # if k == 7:
#         #     RatePerDomain[k] += 5*deta
#         # else:
#         #     RatePerDomain[k] += deta
#         RatePerDomain[k] += deta
#     xaxis += detax
#     print("the x is",xaxis)
#     X.append(xaxis)
#     print("this is HS")
#     ResultList[0].append(caculateAverLatency(encode(COMPARISON), MaxValue))
#     print("this is FS")
#     ResultList[1].append(caculateAverLatency(encode(COMPARISON2), MaxValue))
#     for k in range(numbers):
#         ResultList[k + 2].append(caculateAverLatency(COMPARISONList[k], MaxValue))
#
# for i in ResultList:
#     plt.plot(X, i)
# plt.ylim(0, 2)
# plt.show()
#
# wb = Workbook()
# ws = wb.active
# ws.append(X)
# for i in ResultList:
#     ws.append(i)
#     print('write into excel', i)
# wb.save(r'E:\ljy‘s experiment\data\layersequal3-str.xlsx')



for i in range(NUMBERS):
    # while xaxis<40:
    #     for k in range(K[0]):
    #             # if k == 7:
    #             #     RatePerDomain[k] += 5*deta
    #                 # else:
    #             #     RatePerDomain[k] += deta
    #             RatePerDomain[k] += deta
    #     xaxis += detax
    #     print("the x is",xaxis)
    #     X.append(xaxis)
    if xaxis >=0:
        for k in range(K[0]):
                # if k == 7:
                #     RatePerDomain[k] += 5*deta
                    # else:
                #     RatePerDomain[k] += deta
                RatePerDomain[k] += deta
        xaxis += detax
        print("the x is",xaxis)
        X.append(xaxis)

    #ProbOfFlowDist = setFlowP()
    ResultHA.append(caculateAverLatency(encode(COMPARISON),MaxValue))#对比
#    ResultFlat.append(caculateAverLatency(encode(COMPARISON2)))
    print(ResultHA)
    ResultFA.append(caculateAverLatency(encode(COMPARISON2),MaxValue))
    print(ResultFA)
    result = GA(variables,dm)
    ResultHHA.append(result[0])#最优值，调用GA函数
    ResultHHAD.append(result[1])
    if check(ResultHHAD[-1]) == False:
        print("!!!!!!ERROR!!!!!!!!!!!!!!!")
        os._exit()
    # if X[-1] >=45 and (ResultHHAD[-1] != COMPARISON):
    #     print(ResultHHAD[-1])
    #     detax = detaRatePerHost / (10 ** -3)  # 横坐标的间隔
    #     deta = cluster * detaRatePerHost
    #     X1 = []
    #     Y= []
    #     Y2=[]
    #     xaxis = RatePerHost / (10 ** -3)
    #     RatePerDomain = [Ratetmp for m in range(K[0])]
    #     for i in range(100):
    #         for k in range(K[0]):
    #             # if k == 7:
    #             #     RatePerDomain[k] += 5*deta
    #             # else:
    #             #     RatePerDomain[k] += deta
    #             RatePerDomain[k] += deta
    #         xaxis += detax
    #         # print("the x is", xaxis)
    #         X1.append(xaxis)
    #         Y.append(caculateAverLatency(encode(ResultHHAD[-1] ), MaxValue))
    #         Y2.append(caculateAverLatency(encode(COMPARISON), MaxValue))
    #     plt.plot(X1, Y)
    #     plt.plot(X1,Y2)
    #     plt.ylim(0, 2)
    #
    #     plt.show()
    #     os._exit()

    print(ResultHHA)

# print(ResultHA)
# print(ResultHHA)
# X = [i for i in range(NUMBERS)]
Y = ResultHHA
print(X)
Y2= ResultHA
Y3 = ResultFA
# Y4 = Resultx
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set(ylabel='Average Latency',xlabel='request rate per domains')
# ax.plot(X,Y)
# ax.plot(X,Y2)
#

wb = Workbook()
ws1=wb.active
ws2=wb.active
ws1.append(X)
ws1.append(Y)
ws1.append(Y2)
ws1.append(Y3)
# ws2.append(X)
# ws2.append(ResultHHAD)

wb.save(r'E:\ljy‘s experiment\data\layersequal4.xlsx')

# plt.plot(X,Y)
plt.plot(X,Y2)
plt.plot(X,Y3)
plt.plot(X,Y)
plt.xlabel("request rate per domains")
plt.ylabel("Average Latency")
plt.ylim(0, 2)
plt.show()


#计算方差
