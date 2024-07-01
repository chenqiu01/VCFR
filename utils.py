import copy
import numpy as np


class RegretSolver:
	def __init__(self, dim):
		self.round = 0
		self.sumRewVector = np.zeros(dim)
		self.sumStgyVector = np.zeros(dim)
		self.gained = 0
		self.sumWeight = 0.0
		self.dim = dim
		self.curstgy = np.ones(dim) / self.dim
		
	def take(self):
		ret = np.zeros(self.dim)
		for d in range(self.dim):
			if self.sumRewVector[d] > self.gained:
				ret[d] = self.sumRewVector[d] - self.gained
		s = sum(ret)
		if s < 1e-8:
			return np.ones(self.dim) / self.dim
		return ret / s

	def receive(self, rew, stgy=0, weight=1.0):
		if type(stgy) == list:
			stgy = np.array(stgy)
		elif type(stgy) == int:
			stgy = self.take()
		curgain = np.inner(rew, stgy)
		self.gained += curgain
		self.round += 1
		self.sumRewVector += rew
		self.sumStgyVector += self.curstgy * weight 
		self.curstgy = stgy.copy()
		self.sumWeight += weight

	def avg(self):
		if self.sumWeight < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumStgyVector / self.sumWeight

	def regret(self):
		m = -np.inf 
		for i in range(self.dim):
			m = max(m, self.sumRewVector[i])
		return m - self.gained

def exploitability(game, stgy_prof):
	def best(owner, iset, probs):
		hists = game.Iset2Hists[owner][iset]
		if game.isTerminal[hists[0]]:
			ret = np.zeros(2)
			for i, h in enumerate(hists):
				tmp = np.array(game.reward[h]) * probs[i]
				ret += tmp

			return ret
		player = game.playerOfIset[owner][iset]
		if player != owner:
			obsnacts = game.nactsOnIset[owner][iset]
			if obsnacts == 1:
				realnacts = game.nactsOnHist[hists[0]]
				nxtprobs = np.zeros(0)
				for i in range(realnacts):
					tmp = np.zeros(len(hists))
					for j, p in enumerate(probs):
						h = hists[j]
						_stgy = None
						if player == 2:
							_stgy = game.chanceprob[h]
						else:
							piset = game.Hist2Iset[player][h]
							_stgy = stgy_prof[player][piset]
						tmp[j] = probs[j] * _stgy[i]
					nxtprobs = np.concatenate((nxtprobs, tmp))
				return best(owner, game.isetSucc[owner][iset][0], nxtprobs)
			else:
				ret = np.zeros(2)
				for i in range(obsnacts):
					nxtprobs = np.zeros(0)
					tmp = np.zeros(len(hists))
					for j, p in enumerate(probs):
						h = hists[j]
						_stgy = None
						if player == 2:
							_stgy = game.chanceprob[h]
						else:
							piset = game.Hist2Iset[player][h]
							_stgy = stgy_prof[player][piset]
						tmp[j] = probs[j] * _stgy[i]
					nxtprobs = np.concatenate((nxtprobs, tmp))
					ret += best(owner, game.isetSucc[owner][iset][i], nxtprobs)
				return ret
		else:
			nacts = game.nactsOnIset[owner][iset]
			ret = -np.inf * np.ones(2)
			for i in range(nacts):
				tmp = best(owner, game.isetSucc[owner][iset][i], probs.copy())
				if tmp[owner] > ret[owner]:
					ret = tmp
			return ret



	b0 = best(0, 0, np.ones(1))
	b1 = best(1, 0, np.ones(1))
	return b0[0] + b1[1]


def generateOutcome(game, stgy_prof):
	outcome = np.zeros(game.numHists).tolist()
	rew = np.zeros(game.numHists)
	def solve(hist):
		if game.isTerminal[hist]:
			rew[hist] = game.reward[hist][0]
			outcome[hist] = []
			return rew[hist]
		outcome[hist] = []
		player = game.playerOfHist[hist]
		stgy = None
		if player < 2:
			iset = game.Hist2Iset[player][hist]
			stgy = stgy_prof[player][iset]
		else:
			stgy = game.chanceprob[hist]
		for a in range(game.nactsOnHist[hist]):
			srew = solve(game.histSucc[hist][a])
			outcome[hist].append(srew)
			rew[hist] += stgy[a] * srew

		return rew[hist]

	solve(0)
	return outcome, rew

	

class RegretSolverPlus:
	def __init__(self, dim):
		self.round = 0
		self.sumRewVector = np.zeros(dim)
		self.sumStgyVector = np.zeros(dim)
		self.sumQ = np.zeros(dim)
		self.gained = 0
		self.sumWeight = 0.0
		self.dim = dim
		self.curstgy = np.ones(dim) / self.dim
		
	def take(self):
		s = sum(self.sumQ)
		if s < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumQ / s

	def receive(self, rew, stgy=0, weight=1.0):
		if type(stgy) == list:
			stgy = np.array(stgy)
		elif type(stgy) == int:
			stgy = self.take()
		curgain = np.inner(rew, stgy)
		for i in range(self.dim):
			self.sumQ[i] += rew[i] - curgain
			self.sumQ[i] = max(self.sumQ[i], 0)
		self.gained += curgain
		self.round += 1
		self.sumRewVector += rew
		self.sumStgyVector += self.curstgy * self.round# weight 
		self.curstgy = stgy.copy()
		self.sumWeight += self.round

	def avg(self):
		if self.sumWeight < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumStgyVector / self.sumWeight

	def regret(self):
		m = -np.inf 
		for i in range(self.dim):
			m = max(m, self.sumRewVector[i])
		return m - self.gained

def simulate(game, h, stgy_prof):
	if game.isTerminal[h]:
		game.simulate(h)
		return
	player = game.playerOfHist[h]
	if player == 2:
		nh = game.simulate(h)
		simulate(game, nh, stgy_prof)
		return
	piset = game.Hist2Iset[player][h]
	stgy = stgy_prof[player][piset]
	a = np.random.choice(game.nactsOnHist[h], p=stgy)
	simulate(game, game.histSucc[h][a], stgy_prof)

class RegretSolver_discounted:

	def __init__(self, dim , alpha, beta, gamma):
		self.round = 0                          #轮次
		self.sumRewVector = np.zeros(dim)       #收益向量和 - 初始化为全零数组
		self.sumStgyVector = np.zeros(dim)      #策略向量和 - 初始化为全零数组
		self.gained = 0                         #rew*stgy的加和 具体意义没搞清
		self.sumWeight = 0.0                    #权重
		self.dim = dim                          #大概是维度
		self.curstgy = np.ones(dim) / self.dim  #当前策略 - 初始化为全1/dim数组
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma



	# take(self)  返回平均策略组合
	def take(self):                                               #ret是regret , 这步为求平均策略组合
		ret = np.zeros(self.dim)                                  # ret 维度dim的全零数组 ？没理解这个变量想干什么
		for d in range(self.dim):                                 #迭代dim次

			# for d in range(self.dim):  # 迭代dim次
			# 	if self.sumRewVector[d] > self.gained:  # 收益向量和 > gained 时 当前ret = sumRewVector - gained
			# 		ret[d] = self.sumRewVector[d] - self.gained
			#info_state.cumulative_regret[action] *= (self._iteration**self.alpha /(self._iteration**self.alpha + 1))

			if self.sumRewVector[d] - self.gained >= 0  :
				ret[d] = self.sumRewVector[d] - self.gained

				ret[d]*= (self.round ** self.alpha /(self.round ** self.alpha + 1))
			# else:
			# 	ret[d] = self.sumRewVector[d] - self.gained
			# 	ret[d] *= (self.round ** self.beta / (self.round ** self.beta + 1))




			#
			# if self.sumRewVector[d] > self.gained:                #收益向量和 > gained 时 当前ret = sumRewVector - gained
			# 	ret[d] = self.sumRewVector[d] - self.gained       #比较像求遗憾值，ret是regret....
		s = sum(ret)                                              #ret求和
		if s < 1e-8:
			return np.ones(self.dim) / self.dim                   #当s<0 返回一个和为1的等值数组
		return ret / s                                            #返回ret/s(求平均策略组合？)


	#recieve:给strategy累计权重
	#from: 当前节点reward, 当前strategy（初始为0），权重（初始1）
	#return:null
	def receive(self, rew, stgy=0, weight=1.0):
		if type(stgy) == list:               #当stgy为list 则转为np.array
			stgy = np.array(stgy)
		elif type(stgy) == int:              #当stgy为int，调用take，返回平均策略组合；
			#这一步存在的主要情况应该为第一次stgy被赋值为初始0时，调用take得到一个等值策略组那个if
			stgy = self.take()
		curgain = np.inner(rew, stgy)        #curgain rew和stgy的内积
		self.gained += curgain               #gained累加
		self.round += 1                      #round累加
		self.sumRewVector += rew             #rew累加
		# if self._linear_averaging:
		# 	info_state_node.cumulative_policy[action] += (
		# 			reach_prob * action_prob * (self._iteration ** self.gamma))
		# else:
		# 	info_state_node.cumulative_policy[action] += reach_prob * action_prob

		self.sumStgyVector += self.curstgy * weight    #stgy累加 当前策略*权重
		self.curstgy = stgy.copy()           #curstgy为stgy的复制
		self.sumWeight += weight             #weight累加

	#avg：求策略加权平均
	#from:self
	#return: 加权平均后的策略组合向量
	def avg(self):
		if self.sumWeight < 1e-8:                  #sumWeight为负（初始阶段
			return np.ones(self.dim) / self.dim    #返回等值和为一的数组
		return self.sumStgyVector / self.sumWeight #返回加权平均后的策略组合向量

	#regret: 求遗憾值 这里的sumRewVector是累加了全局还是某个节点？？存疑
	#from self
	#return regret值
	def regret(self):
		m = -np.inf                                #m赋为极小
		for i in range(self.dim):                  #取sumRewVector向量最大的那个值
			m = max(m, self.sumRewVector[i])
		return m - self.gained                     #最大rew-gained为regret值（这里的regret应该是一个数值)