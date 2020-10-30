import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

dataset = pd.read_excel('D:\WaterMelon_2.0.xlsx', encoding='gbk')  # 读取数据
Attributes = dataset.columns[:-1]  # 所有属性的名称
# print(Attributes)
dataset = np.matrix(dataset)
m, n = np.shape(dataset)
D_train = []  # 得到所有的训练样本编号和验证样本编号
D_test = []
for i in range(m):
    if dataset[i, n - 1] == 'train':
        D_train.append(i)
    else:
        D_test.append(i)
# print(D_test)
# print(D_train)
dataset = dataset[:, :-1]
m, n = np.shape(dataset)  # 得到数据集大小
for i in range(m):  # 将标签替换成 好瓜 和 坏瓜
    if dataset[i, n - 1] == '是':
        dataset[i, n - 1] = '好瓜'
    else:
        dataset[i, n - 1] = '坏瓜'
attributeList = []  # 属性列表，每一个属性的取值，列表中元素是集合
for i in range(n):
    curSet = set()  # 使用集合是利用了集合里面元素不可重复的特性，从而提取出了每个属性的取值
    for j in range(m):
        curSet.add(dataset[j, i])
    attributeList.append(curSet)
# print(attributeList)
A = list(np.ones(n))  # 表示每一个属性是否被使用，使用过了标为 -1
A[-1] = -1  # 将数据里面的标签和编号列标记为 -1
A[0] = -1


# print(A)
# print(D)

class Node(object):  # 创建一个类，用来表示节点的信息
    def __init__(self, title):
        self.title = title  # 上一级指向该节点的线上的标记文字
        self.v = 1  # 节点的信息标记
        self.children = []  # 节点的孩子列表
        self.train = []  # 节点上所含的训练样本编号，主要用于在剪枝时确定节点的标签类别
        self.deep = 0  # 节点深度
        self.ID = -1  # 节点编号


def isSameY(D):  # 判断所有样本是否属于同一类
    curY = dataset[D[0], n - 1]
    for i in range(1, len(D)):
        if dataset[D[i], n - 1] != curY:
            return False
    return True


def isBlankA(A):  # 判断 A 是否是空，是空则返回true
    for i in range(n):
        if A[i] > 0: return False
    return True


def isSameAinD(D, A):  # 判断在D中，是否所有的未使用过的样本属性均相同
    for i in range(n):
        if A[i] > 0:
            for j in range(1, len(D)):
                if not isSameValue(dataset[D[0], i], dataset[D[j], i]):
                    return False
    return True


def isSameValue(v1, v2):  # 判断v1、v2 是否相等
    return v1 == v2


def mostCommonY(D):  # 寻找D中样本数最多的类别
    res = dataset[D[0], n - 1]  # D中第一个样本标签
    maxC = 1
    count = {}
    count[res] = 1  # 该标签数量记为1
    for i in range(1, len(D)):
        curV = dataset[D[i], n - 1]  # 得到D中第i+1个样本的标签
        if curV not in count:  # 若之前不存在这个标签
            count[curV] = 1  # 则该标签数量记为1
        else:
            count[curV] += 1  # 否则 ，该标签对应的数量加一
        if count[curV] > maxC:  # maxC始终存贮最多标签对应的样本数量
            maxC = count[curV]  # res 存贮当前样本数最多的标签类型
            res = curV
    return res  # 返回的是样本数最多的标签的类型


def gini(D):  # 参数D中所存的样本的基尼值
    types = []  # 存贮类别标签
    count = {}  # 存贮每个类别对应的样本数量
    for i in range(len(D)):  # 统计D中存在的每个类型的样本数量
        curY = dataset[D[i], n - 1]
        if curY not in count:
            count[curY] = 1
            types.append(curY)
        else:
            count[curY] += 1
    ans = 1
    total = len(D)  # D中样本总数量
    for i in range(len(types)):  # 计算基尼值
        ans -= (count[types[i]] / total) ** 2
    return ans


def gini_indexD(D, p):  # 属性 p 上的基尼指数
    types = []
    count = {}
    for i in range(len(D)):  # 得到每一个属性取值上的样本编号
        a = dataset[D[i], p]
        if a not in count:
            count[a] = [D[i]]
            types.append(a)
        else:
            count[a].append(D[i])
    res = 0
    total = len(D)
    for i in range(len(types)):  # 计算出每一个属性取值分支上的基尼值，再计算出基尼指数
        res += len(count[types[i]]) / total * gini(count[types[i]])
    return res


def treeGenerate(D, A, title):
    node = Node(title)
    node.train = D
    if isSameY(D):  # D中所有样本是否属于同一类
        node.v = dataset[D[0], n - 1]
        return node

    # 是否所有属性全部使用过  或者  D中所有样本的未使用的属性均相同
    if isBlankA(A) or isSameAinD(D, A):
        node.v = mostCommonY(D)  # 此时类别标记为样本数最多的类别（暗含可以处理存在异常样本的情况）
        return node  # 否则所有样本的类别应该一致

    gini_index = float('inf')
    p = 0
    for i in range(len(A)):  # 循环遍历A,找可以获得最小基尼指数的属性
        if (A[i] > 0):
            curGini_index = gini_indexD(D, i)
            if curGini_index < gini_index:
                p = i  # 存贮属性编号
                gini_index = curGini_index

    node.v = Attributes[p] + "=?"  # 节点信息
    curSet = attributeList[p]  # 该属性的所有取值
    for i in curSet:
        Dv = []
        for j in range(len(D)):  # 获得该属性取某一个值时对应的训练集样本标号
            if dataset[D[j], p] == i:
                Dv.append(D[j])

            # 若该属性取值对应没有符合的样本，则将该分支作为叶子，类别是D中样本数最多的类别
            # 其实就是处理在没有对应的样本情况下的问题。那就取最大可能性的一类。
        if Dv == []:
            nextNode = Node(i)
            nextNode.v = mostCommonY(D)
            node.children.append(nextNode)
        else:  # 若存在对应的样本，则递归继续生成该节点下的子树
            newA = copy.deepcopy(A)  # 注意是深度复制，否则会改变A中的值
            newA[p] = -1
            node.children.append(treeGenerate(Dv, newA, i))
    return node


def postPruning(root):  # 后剪枝操作
    maxDeep = getMaxDeep(root, 0)  # 得到树的最大深度
    for de in range(maxDeep - 1, 0, -1):  # 循环依次从最低层进行遍历操作
        notLeafnode = getNotLeafnode(root, de)  # 得到指定深度上的非叶子节点列表
        notLeafnode = np.array(notLeafnode).flatten()  # 主要是进行形状变换，拉平成一维数组
        for i in range(len(notLeafnode)):  # 循环遍历每一个非叶节点
            befpruning = getRightNum(root, D_test) / len(D_test)  # 剪枝之前的精确度
            node = notLeafnode[i]  # 得到一个节点
            curv = node.v  # 当前节点的信息
            v = mostCommonY(node.train)  # 根据该节点包含的训练集样本得到剪枝后的类别
            node.v = v  # 进行剪枝，注意此时仅仅是改了信息，与子节点的连接依然存在
            aftpruning = getRightNum(root, D_test) / len(D_test)  # 剪枝后的精确度
            if aftpruning > befpruning:  # 此处用严格大于，是为了可以画出一个好看的树，实际情况下应该用大于等于，参见西瓜书P82页的解释
                node.children = []  # 彻底进行剪枝，去除和子节点的连接信息
                print("去掉划分属性 ", curv[0:2])
                print("剪之前精确度：", befpruning)
                print("剪之后精确度：", aftpruning)
            else:
                node.v = curv  # 若不需要剪枝，则直接更改节点的信息即可恢复到原树
                print("恢复划分属性 ", curv[0:2])
                print("剪之前精确度：", befpruning)
                print("剪之后精确度：", aftpruning)


def getMaxDeep(root, deep):  # 得到决策树的最大深度
    root.deep = deep
    if root.v == '好瓜' or root.v == '坏瓜':
        return deep
    curdeep = deep
    for i in root.children:
        b = getMaxDeep(i, deep + 1)
        if b > curdeep:
            curdeep = b
    return curdeep


def getNotLeafnode(root, deep):  # 迭代得到指定深度处的非叶子节点
    if root.v != '好瓜' and root.v != '坏瓜' and root.deep == deep:
        return root
    else:
        node = []  # 注意，这个语句只能放在else内，切不可放在函数开头！！！
        if root.children != []:
            for i in root.children:
                curnode = getNotLeafnode(i, deep)
                if curnode != []:
                    node.append(curnode)
    return node


def getRightNum(root, D):  # 得到在样本集合D上正确分类的样本数目
    if root.v == '好瓜':
        good = getGoodNum(D)
        return good
    if root.v == '坏瓜':
        bad = getBadNum(D)
        return bad
    children = root.children
    child = children[0]
    num = 0
    v = root.v[0:2]
    p = getIndex(Attributes, v)
    curSet = attributeList[p]
    for i in curSet:
        for k in children:
            if k.title == i:
                child = k
                break
        Dv = []
        for j in range(len(D)):
            if dataset[D[j], p] == i:
                Dv.append(D[j])
        if Dv != []:
            num += getRightNum(child, Dv)
    return num


def getGoodNum(D):  # 若标签是好瓜，得到样本中好瓜的数目
    num = 0
    for i in range(len(D)):
        if dataset[D[i], n - 1] == '好瓜':
            num += 1
    return num


def getBadNum(D):  # 同上，得到坏瓜的数目
    num = 0
    for i in range(len(D)):
        if dataset[D[i], n - 1] == '坏瓜':
            num += 1
    return num


def getIndex(LL, aa):  # 得到一个列表里面指定元素的索引
    for i in range(len(LL)):
        if LL[i] == aa:
            return i


def countLeaf(root, deep):
    root.deep = deep
    res = 0
    if root.v == '好瓜' or root.v == '坏瓜':  # 说明此时已经是叶子节点了，所以直接返回
        res += 1
        return res, deep
    curdeep = deep  # 记录当前深度
    for i in root.children:  # 得到子树中的深度和叶子节点的个数
        a, b = countLeaf(i, deep + 1)
        res += a
        if b > curdeep: curdeep = b
    return res, curdeep


def giveLeafID(root, ID):  # 给叶子节点编号
    if root.v == '好瓜' or root.v == '坏瓜':
        root.ID = ID
        ID += 1
        return ID
    for i in root.children:
        ID = giveLeafID(i, ID)
    return ID


def plotNode(nodeTxt, centerPt, parentPt, nodeType, arrow_args):  # 绘制节点
    plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                 textcoords='axes fraction', va="center", ha="center", bbox=nodeType,
                 arrowprops=arrow_args)


def dfsPlot(root, decisionNode, leafNode, arrow_args, cnt, deep):
    if root.ID == -1:  # 说明根节点不是叶子节点
        childrenPx = []
        meanPx = 0
        for i in root.children:
            cur = dfsPlot(i, decisionNode, leafNode, arrow_args, cnt, deep)
            meanPx += cur
            childrenPx.append(cur)
        meanPx = meanPx / len(root.children)
        c = 0
        for i in root.children:
            nodetype = leafNode
            if i.ID < 0: nodetype = decisionNode
            plotNode(i.v, (childrenPx[c], 0.9 - i.deep * 0.8 / deep), (meanPx, 0.9 - root.deep * 0.8 / deep), nodetype,
                     arrow_args)
            plt.text((childrenPx[c] + meanPx) / 2, (0.9 - i.deep * 0.8 / deep + 0.9 - root.deep * 0.8 / deep) / 2,
                     i.title)
            c += 1
        return meanPx
    else:
        return 0.1 + root.ID * 0.8 / (cnt - 1)


def plotTree(root):  # 绘制决策树
    cnt, deep = countLeaf(root, 0)  # 得到树的深度和叶子节点的个数
    giveLeafID(root, 0)
    decisionNode = dict(boxstyle="sawtooth", fc="0.9", color='blue')
    leafNode = dict(boxstyle="round4", fc="0.9", color='red')
    arrow_args = dict(arrowstyle="<-", color='green')
    fig = plt.figure(1, facecolor='white')
    rootX = dfsPlot(root, decisionNode, leafNode, arrow_args, cnt, deep)
    plotNode(root.v, (rootX, 0.9), (rootX, 0.9), decisionNode, arrow_args)
    plt.show()


myDecisionTreeRoot = treeGenerate(D_train, A, "root")  # 生成未剪枝决策树
plotTree(myDecisionTreeRoot)  # 未剪枝的决策树
postPruning(myDecisionTreeRoot)  # 进行后剪枝
plotTree(myDecisionTreeRoot)  # 后剪枝的决策树