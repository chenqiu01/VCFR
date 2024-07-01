import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus, simulate,RegretSolver_discounted
import time
from  vime import VIME


class VIME_DIS:
    # 初始化：传入game, Type)
    def __init__(self, game , alpha, beta, gamma, Type="default"):
        self.game = game
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Type = Type
        Solver = None
        Solver = RegretSolver_discounted  # Solver预设为RegretSolver（测试是也可替换为Plus）
        _, rews = self.game.resample()
        self.vime = VIME(rews)  # 初试化

        self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]
        # isetflag用于做信息集flag，全-1数组[[-1,-1,-1] , [-1,-,1,-1]]数组长度分别为玩家0，1的信息集大小

        self.solvers = []  # 按照映射 将nactsOnIset的数据写进solvers，根据原nactsOnIset的顺序导入（0，1玩家同操作）
        self.solvers.append(
            list(map(lambda x: Solver(game.nactsOnIset[0][x], alpha, beta, gamma), range(game.numIsets[0]))))
        self.solvers.append(
            list(map(lambda x: Solver(game.nactsOnIset[1][x], alpha, beta, gamma), range(game.numIsets[1]))))
        self.stgy = [[], []]
        for i, iset in enumerate(range(game.numIsets[0])):  # 以玩家0信息集大小为范围进行迭代，i为序号，iset为元素
            nact = game.nactsOnIset[0][iset]  # nact赋值为player0对应信息集的nact
            if game.playerOfIset[0][iset] == 0:  # 当前做决策的player=0(目前是遍历玩家0的信息集，但这里求的是在当前信息集正在做动作的玩家)
                self.stgy[0].append(np.ones(nact) / nact)  # 玩家0的stgy插入： 1/nact动作的数组
            else:  # 玩家1做决策
                self.stgy[0].append(np.ones(0))  # 插入0数组

        for i, iset in enumerate(range(game.numIsets[1])):  # 同理 迭代player=1的信息集
            nact = game.nactsOnIset[1][iset]  # nact为palyer1的信息集对应的nact
            if game.playerOfIset[1][iset] == 1:  # 当前为player=1做决策（对其来说是 己方决策）
                self.stgy[1].append(np.ones(nact) / nact)  # player1的stgy插入 1/nact
            else:
                self.stgy[1].append(np.ones(0))  # player0做决策 插入0数组

        self.outcome, self.reward = generateOutcome(game, self.stgy)  # 根据game和stgy求每个节点的outcome和reward generateOutcome
        self.nodestouched = 0  # nodestouched初始化为0 碰到的节点
        self.round = -1  # 轮次赋值为-1

        self.sumstgy = [[], []]  # 这里开始求策略和组合
        for i, iset in enumerate(range(game.numIsets[0])):  # 这里完全是复制前一段代码 可能是写错了 不影响outcome和reward
            nact = game.nactsOnIset[0][iset]
            if game.playerOfIset[0][iset] == 0:
                self.sumstgy[0].append(np.ones(nact) / nact)
            else:
                self.sumstgy[0].append(np.ones(0))
        for i, iset in enumerate(range(game.numIsets[1])):
            nact = game.nactsOnIset[1][iset]
            if game.playerOfIset[1][iset] == 1:
                self.sumstgy[1].append(np.ones(nact) / nact)
            else:
                self.sumstgy[1].append(np.ones(0))

        self.sampledtrans1 = list(map(lambda _u: np.zeros(game.nactsOnHist[_u]), range(game.numHists)))  # 采样的转移
        self.sampledrews1 = list(map(lambda _u: np.zeros(2), range(game.numHists)))  # 采样的reward
        self.weight1 = np.zeros(game.numHists)  # 权重

    def updateAll(self):  # update..迭代开始
        game = self.game  # 引入game
        self.round += 1  # 轮次++
        transpsrl, rewpsrl = game.resample()  # transpsrl rewpsrl 根据游戏进行采样
        transvalidation, rewvalidation = game.resample()  # transvalidation，rewvalidation跟上边的操作一样

        # 最后w位置的传参实际应为所有curtrans[h][a]相乘
        def avgchance(h, curtrans, currew, w, sumtrans, sumrew,
                      sumw):  # 这里的传参分别对应： h(hists),curtrans(转移-transpsrl)，currew（收益-rewpsrl），w应该是权重（初始为1），sumtrans转移和（sampledtrans1），sumrew收益和（sampledrews1），权重和（weight1）
            sumw[h] += w  # sumw[h] 对应当前h的权重和 +参数w
            term = game.isTerminal[h]  # 判断当前h是否为终止节点（term为bool）
            player = game.playerOfHist[h]  # player为当前节点h对应的玩家
            # currew[h][0] 对应 rewpsrl[h][0] rewpsrl由resample得到 且rew = np.array([1 - a[1] * 2, a[1] * 2 - 1]) 分别对应0,1

            if term == True:  # 终止节点
                sumrew[h] += (w * currew[h][0], w * currew[h][1])  # sumrew 为两个玩家的收益 拼在一起的数组
                return
            if player == 2:  # chance节点
                for a in range(game.nactsOnHist[h]):  # 对历史节点中的所有nacts进行遍历，计算他们的avgchance
                    avgchance(game.histSucc[h][a], curtrans, currew, w * curtrans[h][a], sumtrans, sumrew,
                              sumw)  # 对当前的act和hist计算avgchance所其中w变为w * curtrans[h][a]
                sumtrans[h] += w * curtrans[h]  # sumtrans +=权重*当前trans
            else:
                for a in range(game.nactsOnHist[h]):  # 非chance节点，遍历h的nacts
                    avgchance(game.histSucc[h][a], curtrans, currew, w, sumtrans, sumrew,
                              sumw)  # 迭代！权重处w数值不变（也就是迭代的传参啥也没改）


        avgchance(0, transpsrl, rewpsrl, 1.0, self.sampledtrans1, self.sampledrews1, self.weight1)  # 此处为真正调用avgchance函数
        #经过avgchance之后的 transpsrl, rewpsrl没变化，而sampledtrans1，sampledrews1变为加权求和后的

        self.vime.update_reward(rewpsrl)
        # transpsrl应该是没被avgchance改变的值，因此为resample的 直接结果
        # 但transpsrl对应CFR算法中是chance的prob。那，avg那步有啥意义？？？
        self.update(0, 0, [np.ones(1), np.ones(1)], [0], rewpsrl,
                    transpsrl)  # the CFR algorithm          #此处为对玩家0进行CFR操作
        self.update(1, 0, [np.ones(1), np.ones(1)], [0], rewpsrl,
                    transpsrl)  # the CFR algorithm          #此处为对玩家1 进行CFR

        # 将CFR算法后solver中存入的stgy到处到stgy数组（CFR常规内容）
        def updStgy(owner, iset):
            if self.isetflag[owner][iset] != self.round:
                return
            player = game.playerOfIset[owner][iset]
            if player == owner:
                self.stgy[owner][iset] = self.solvers[owner][iset].curstgy.copy()
            for nxtiset in game.isetSucc[owner][iset]:
                updStgy(owner, nxtiset)

        updStgy(0, 0)  # 对玩家0的策略更新
        updStgy(1, 0)  # 玩家1策略更新

        # 更新策略和（CFR常规内容）
        def updSumstgy(owner, iset, prob=1.0):
            player = game.playerOfIset[owner][iset]
            if player == owner:
                ###加gammma
                self.sumstgy[owner][iset] += prob * self.stgy[player][iset] * (self.round ** self.gamma)
                for aid, nxtiset in enumerate(game.isetSucc[owner][iset]):
                    if prob * self.stgy[player][iset][aid] > 1e-8:
                        updSumstgy(owner, nxtiset, prob * self.stgy[player][iset][aid])
            else:
                for aid, nxtiset in enumerate(game.isetSucc[owner][iset]):
                    updSumstgy(owner, nxtiset, prob)

        updSumstgy(0, 0)  # 玩家0策略和更新
        updSumstgy(1, 0)  # 对玩家1的策略和更新

        avgstgy = self.avgstgyprofile()  # 平均策略组合

        # 求Exploration Strategy（公式5的实现）
        # 参数 owner-玩家，iset-信息集，explore_stgy-?, oppstgy-对家策略？,ds_c1 - 概率prob下的rewvalidation, transvalidation,ds_c2 - 同
        # 实际调用的传参getExploreStgy(0, 0, explore_stgy[0], avgstgy[1], (rewvalidation, transvalidation, prob2), (avgchrew, avgchtrans, prob1))
        def getExploreStgy(owner, iset, explore_stgy, oppstgy, ds_c1, ds_c2):
            rew1, trans1, reachp1 = ds_c1  # ds_c1的传参格式rewvalidation, transvalidation, prob2（估计reachp1-到达的概率）
            rew2, trans2, reachp2 = ds_c2  # 同
            hists = game.Iset2Hists[owner][iset]  # 根据玩家和信息集得到hists
            if game.isTerminal[hists[0]] == True:  # 当前节点是终止节点 终止函数
                return
            player = game.playerOfIset[owner][iset]  # 确定当前节点正在做决策的玩家
            if player == owner:  # 当前为我方做决策
                nacts = game.nactsOnIset[owner][iset]  # 当前owner,iset对应的nacts
                outcome1 = np.zeros(nacts)  # outcome1 nacts长度的全零数组
                outcome2 = np.zeros(nacts)  # 同
                for a in range(nacts):  # 遍历nacts
                    getExploreStgy(owner, game.isetSucc[owner][iset][a], explore_stgy, oppstgy, ds_c1, ds_c2)
                    # 递归当前节点的后继，其他参数全都不变
                    for h in hists:  # 遍历hists里所有的h
                        outcome1[a] += reachp1[h] * rew1[game.histSucc[h][a]][
                            owner]  # outcome 累加 参数中的概率（这个概率不知道怎么得出的）*reeard
                        outcome2[a] += reachp2[h] * rew2[game.histSucc[h][a]][owner]  # 通过ds_c2的参数进行累加
                a_star = np.argmax(outcome1 - outcome2)  # a_star 取最大值的索引 outcome1-outcome2 差最大的那个动作
                _stgy = np.zeros(nacts)  # _stgy 与nacts长度相同的全零数组
                _stgy[a_star] = 1  # 刚刚那个差最大的对应策略置一
                explore_stgy[iset] = _stgy  # explore_stgy[iset]是一个 只有一个1剩下全零的数组
                for h in hists:
                    rew1[h] = rew1[game.histSucc[h][a_star]]  # rew1 后继节点 、 取对应差最小的那个a
                    rew2[h] = rew2[game.histSucc[h][a_star]]  # rew2
            else:  # 对家做决策
                truenacts = game.nactsOnHist[hists[0]]  # 没用到这个变量
                obsnacts = game.nactsOnIset[owner][iset]  # obsnacts 当前玩家（不是Player）在iset下的nacts
                for h in hists:  # 遍历hist
                    _stgy1 = None  # _stgy1为NONE
                    _stgy2 = None  # 同
                    if player == 2:  # chance节点
                        _stgy1 = trans1[h]  # _stgy1 赋值为trans1[h](ds_c1传参进来的)
                        _stgy2 = trans2[h]  # 同
                    else:  # 非chance节点
                        piset = game.Hist2Iset[player][h]  # piset 当前做决策玩家的iset
                        _stgy1 = oppstgy[piset]  # 对方_stgy1
                        _stgy2 = oppstgy[piset]  # 完全相同，ummmm有点迷惑，那没必要搞两个
                    nactsh = game.nactsOnHist[h]  # nactsh 当前h下的nacts??
                    for a in range(nactsh):  # 遍历nactsh
                        reachp1[game.histSucc[h][a]] = reachp1[h] * _stgy1[a]  # 同理， de_cs中的参数 概率*策略得到新的reachp1
                        reachp2[game.histSucc[h][a]] = reachp2[h] * _stgy2[a]  # 同，_stgy12完全没区别
                for a in range(obsnacts):  # 遍历obsnacts
                    getExploreStgy(owner, game.isetSucc[owner][iset][a], explore_stgy, oppstgy, ds_c1, ds_c2)
                # 递归后继节点 其他不变
                for h in hists:  # 跟刚刚那段基本一样
                    _stgy1 = None
                    _stgy2 = None
                    if player == 2:
                        _stgy1 = trans1[h]
                        _stgy2 = trans2[h]
                    else:
                        piset = game.Hist2Iset[player][h]
                        _stgy1 = oppstgy[piset]
                        _stgy2 = oppstgy[piset]
                    nactsh = game.nactsOnHist[h]
                    for a in range(nactsh):  # 从此处开始有区别
                        rew1[h] += rew1[game.histSucc[h][a]] * _stgy1[a]  # 改为计算rew1、2
                        rew2[h] += rew2[game.histSucc[h][a]] * _stgy2[a]
            if iset == 0:  # iset = 0 空博弈
                pass
            # print("check", rew1[0] - rew2[0], rew1[0], rew2[0])

        # avgchrew = np.zeros(game.numHists).tolist()
        # def getavgchance(h, avgchrew, avgchtrans):

        # getavgchance(0, avgchrew, avgchtrans)
        prob1 = np.ones(game.numHists)  # prob1全零
        prob2 = np.ones(game.numHists)  # 同：在之后的计算中prob1和2都是同样的使用方法，进行的计算也一样
        explore_stgy = [[], []]  # 表示两个玩家的explore_stgy
        for i, iset in enumerate(range(game.numIsets[0])):  # 应该是直接求平均策略的步骤
            nact = game.nactsOnIset[0][iset]
            if game.playerOfIset[0][iset] == 0:
                explore_stgy[0].append(np.ones(nact) / nact)
            else:
                explore_stgy[0].append(np.ones(0))
        for i, iset in enumerate(range(game.numIsets[1])):  # 同
            nact = game.nactsOnIset[1][iset]
            if game.playerOfIset[1][iset] == 1:
                explore_stgy[1].append(np.ones(nact) / nact)
            else:
                explore_stgy[1].append(np.ones(0))

        avgchtrans = copy.deepcopy(self.sampledtrans1)  # avgchtrans为前半部分sampledtrans1的copy，这个值根本没用过，不知道为啥要copy
        avgchrew = copy.deepcopy(self.sampledrews1)  # 同样，做rew的copy

        for h in range(game.numHists):  # 给刚刚两个值做加权
            if game.isTerminal[h]:
                avgchrew[h] /= self.weight1[h]
            if game.playerOfHist[h] == 2:
                avgchtrans[h] /= self.weight1[h]

        if self.Type == "default":  # 对于default的类型（PSRL默认）
            self.vime.update_reward(rewvalidation)
            #avg的rew是否需要VIME
            #rewvalidation是从环境里采样的 ，avgchrew是经过avgchance得到加权求和的

            # prob1和porb2是一样的初始化方式，不知道为啥要倒过来用
            getExploreStgy(0, 0, explore_stgy[0], avgstgy[1], (rewvalidation, transvalidation, prob2),
                           (avgchrew, avgchtrans, prob1))  # 玩家1调用getExploreStgy
            getExploreStgy(1, 0, explore_stgy[1], avgstgy[0], (rewvalidation, transvalidation, prob2),
                           (avgchrew, avgchtrans, prob1))  # 玩家2调用getExploreStgy
            for t in range(1):  # 两个迭代
                simulate(game, 0, [explore_stgy[0], self.stgy[1]])  # 这一段是跟环境交互
                simulate(game, 0, [self.stgy[0], explore_stgy[1]])

        if self.Type == "br_dirc":
            self.vime.update_reward(rewvalidation)
            #print(avgchrew)
            getExploreStgy(0, 0, explore_stgy[0], avgstgy[1], (rewvalidation, transvalidation, prob2),
                           (np.asarray(avgchrew) * 0., avgchtrans, prob1))
            getExploreStgy(1, 0, explore_stgy[1], avgstgy[0], (rewvalidation, transvalidation, prob2),
                           (np.asarray(avgchrew) * 0., avgchtrans, prob1))
            for t in range(1):
                simulate(game, 0, [explore_stgy[0], self.stgy[1]])
                simulate(game, 0, [self.stgy[0], explore_stgy[1]])

        if self.Type == "ordinary":
            simulate(game, 0, avgstgy)
            simulate(game, 0, avgstgy)
        if self.Type == "random":
            for i in range(2):
                for iset in range(game.numIsets[i]):
                    pl = game.playerOfIset[i][iset]
                    explore_stgy[i].append(0)
                    if pl == i:
                        nacts = game.nactsOnIset[i][iset]
                        explore_stgy[i][iset] = np.ones(nacts) / nacts
            simulate(game, 0, explore_stgy)
            simulate(game, 0, explore_stgy)

    # update 即为CFR那一套操作
    # 其中chanceprob即为之前求出的transprl,所以
    # self.update(1, 0, [np.ones(1), np.ones(1)], [0], rewpsrl, transpsrl)
    def update(self, owner, iset, probs, histories, rewards, chanceprob):
        self.isetflag[owner][iset] = self.round
        if len(histories) == 0:
            return np.zeros(0)

        self.nodestouched += len(histories)
        game = self.game
        player = game.playerOfIset[owner][iset]

        if game.isTerminal[game.Iset2Hists[owner][iset][0]]:
            ret = np.zeros(len(histories))
            for hid, h in enumerate(histories):
                ##去掉game.是原来的

                ret[hid] = rewards[h][owner]
            return ret

        if player == owner:

            nacts = game.nactsOnIset[owner][iset]

            ret = np.zeros(len(histories))
            cfv = np.zeros(game.nactsOnIset[owner][iset])
            for a in range(nacts):
                nxthistories = []
                nxtprobs = [probs[0].copy(), probs[1].copy()]
                nxtprobs[owner] *= self.stgy[owner][iset][a]
                for h in histories:
                    nxthistories.append(game.histSucc[h][a])
                tmpr = self.update(owner, game.isetSucc[owner][iset][a], nxtprobs, nxthistories, rewards, chanceprob)
                for nhid, nh in enumerate(nxthistories):
                    cfv[a] += probs[1 - owner][nhid] * tmpr[nhid]
                    ret[nhid] += tmpr[nhid] * self.stgy[player][iset][a]

            self.solvers[owner][iset].receive(cfv, weight=probs[owner][0])
            return ret

        else:
            obsnact = game.nactsOnIset[owner][iset]
            nacts = np.array(list(map(lambda _u: game.nactsOnHist[_u], histories)))
            truenact = int(nacts.max())

            if truenact == obsnact:
                ret = np.zeros(len(histories))
                for i in range(truenact):
                    nxtiset = game.isetSucc[owner][iset][i]
                    nxtprobs = [[], []]
                    nxthistories = []
                    _ps = []
                    _ids = []
                    for j, h in enumerate(histories):
                        player = game.playerOfHist[h]
                        stgy = None
                        if player == 2:
                            stgy = chanceprob[h]
                        else:
                            piset = game.Hist2Iset[player][h]
                            stgy = self.stgy[player][piset]
                        if probs[1 - owner][j] * stgy[i] < 1e-5:
                            pass
                        else:
                            nxtprobs[1 - owner].append(probs[1 - owner][j] * stgy[i])
                            nxthistories.append(game.histSucc[h][i])
                            nxtprobs[owner].append(probs[owner][j])
                            _ps.append(stgy[i])
                            _ids.append(j)
                    nxtprobs[0] = np.array(nxtprobs[0])
                    nxtprobs[1] = np.array(nxtprobs[1])
                    tmpr = self.update(owner, nxtiset, nxtprobs, nxthistories, rewards, chanceprob)
                    if len(_ids) > 0:
                        for _id, _r in enumerate(tmpr):
                            ret[_ids[_id]] += _ps[_id] * _r
                return ret

            else:
                ret = np.zeros(len(histories))
                nxtprobs = [[], []]
                nxthistories = []
                _ps = []
                _ids = []
                for aid in range(truenact):
                    for hid, h in enumerate(histories):

                        player = game.playerOfHist[h]
                        stgy = None
                        if player == 2:
                            stgy = chanceprob[h]
                        else:
                            piset = game.Hist2Iset[player][h]
                            stgy = self.stgy[player][piset]
                        if probs[1 - owner][hid] * stgy[aid] < 1e-5:  # player == 2 and
                            pass
                        else:
                            nxtprobs[1 - owner].append(probs[1 - owner][hid] * stgy[aid])
                            nxthistories.append(game.histSucc[h][aid])
                            nxtprobs[owner].append(probs[owner][hid])
                            _ps.append(stgy[aid])
                            _ids.append(hid)
                nxtiset = game.isetSucc[owner][iset][0]
                nxtprobs[0] = np.array(nxtprobs[0])
                nxtprobs[1] = np.array(nxtprobs[1])
                tmpr = self.update(owner, game.isetSucc[owner][iset][0], nxtprobs, nxthistories, rewards, chanceprob)
                for _i, _r in enumerate(tmpr):
                    ret[_ids[_i]] += _r * _ps[_i]
                return ret

    # 平均策略组合，将两个玩家的策略通过映射连在一起（返回值stgy_prof中为两个玩家策略）
    def avgstgyprofile(self):

        stgy_prof = []

        def avg(_x):
            s = np.sum(_x)
            l = _x.shape[0]
            if s < 1e-5:
                return np.ones(l) / l
            return _x / s

        stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[0])))
        stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[1])))
        return stgy_prof

    def getExploitability(self):
        stgy_prof = self.avgstgyprofile()
        return exploitability(self.game, stgy_prof)


"""
bidmaximum = 4
gamepath = "leduc_3_" + str(bidmaximum) + ".npz"
game = Game(path=gamepath)#bidmaximum=bidmaximum)
print("game", game.numHists, game.numIsets)
cfr =CFRPSRL(game)

expls = []
for i in range(100000):
	cfr.updateAll()
	ae = cfr.getExploitability()
	print(i, Type, ae)
	expls.append(ae)
	if i % 50 == 0:
		np.savez("leduc_"+str(bidmaximum) + "cfrpsrl_"+Type, expls=np.array(expls))

"""