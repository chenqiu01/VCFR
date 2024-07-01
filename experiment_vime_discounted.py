import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus
import time
import argparse
import ficplay
import cfrpsrl
import mccfroutcome
import fittedqfsp
# import cfrpsrl_discounted
import  vime_discount
import logging
import json
from collections import defaultdict

def write_JSON(resultpath, Type, expls, rounds):
    #path ：JSON文件地址，url,language,descip：Github相关信息
    js = defaultdict(list)

    js["resultpath"]=(resultpath)
    js["rounds"]=(rounds)
    js["expls"]=(expls)

    json_str = json.dumps(js)

    with open(resultpath+Type, 'a') as json_file:
        json_file.write(json_str)
        json_file.write("\n")
# run(game, solvername=algo, Type=Type, resultpath="leduc_"+str(cards)+"_"+str(betm)+"_"+str(_gameid) )
def run(game, path="result",  solvername = "vime_discount", Type="br_dirc", resultpath="leduc_3_4_10"):

	def solve(gamesolver, reporttime=2, timelim = 3000, minimum=0, mrounds = 20000): # gamesolver是什么？？？怎么引过来的？
		cumutime = 0
		timestamp = time.time()
		result = [] # 结果
		expls = []  # 可利用度
		rounds = [] # 轮次
		rnds = -1
		while rnds < mrounds: # 一共循环10000局
			rnds += 1
			gamesolver.updateAll()
			if rnds % 50 == 0: # 每30轮就 计算并获取 一次可利用度
				expl = gamesolver.getExploitability() # 获得可利用度
				print("solvername", solvername, Type, "game", gamepath, "expl", expl, "rounds", rnds)
				expls.append(expl)
				rounds.append(rnds)
				# print("results/" + resultpath+"_"+solvername+"_"+Type, expls = expls, rounds = rounds)
				print("results/" + resultpath+"_"+solvername+"_"+Type, expls, rounds)
				write_JSON(resultpath, Type, expls, rounds)
				print("results/leduc_"+str(cards)+"_"+str(betm)+"_"+str(_gameid)+"_"+algo+"_"+Type +".npz")
				# print("没在这里")
				#  results/leduc_3_4_10_cfrpsrl_default.npz
				np.savez( "/root/progs/submit/results/" + resultpath+"_"+solvername+"_dcfr_"+Type + ".npz", expls = expls, rounds = rounds)
				print("results/" + resultpath+"_"+solvername+"_"+Type + ".npz" + " " + "saved")
				#  games/leduc_3_4_10
		return (expls,  rounds)   # solve函数返回的是 可利用度和轮次

	solver = None
	if solvername == "vime_discount":
		solver = vime_discount.VIME_DIS(game,  3/2 , 0 ,2 ,Type=Type)
	if solvername == "cfrpsrl_discounted":
		solver = cfrpsrl_discounted.CFRPSRL_discounted(game, 1 , 1 ,1 ,Type=Type)
	if solvername == "ficplay": # 虚拟对局
		solver = ficplay.FICPLAY(game)
	if solvername == "mccfros": # 基于结果抽样的MCCFR
		solver = mccfroutcome.MCCFR_OS(game)
	if solvername == "qfsp":
		solver = fittedqfsp.FICPLAY(game)

	res = solve(solver)  # res就是 可利用度+轮次 （expls,rounds）
	return res

if __name__ == "__main__":
	# mc_rmplus, lazy_rmplus, vanilla_rmplus = run(game, Type="regretmatching+")
	# mc_rm, lazy_rm, vanilla_rm = run(game, Type="regretmatching")
	"""
	algo = "vime_cfr"
	Type = "avg"
	print("haha")
	bedm = 5
	gameid = 1

	gamepath="games/leduc_3_"+str(bidm)+"_"+str(gameid) 
	game = Game(path=gamepath+ ".npz")

	print("game info:", game.numHists, game.numIsets, algo, Type)
	res = run(game, solvername=algo, Type=Type)
	"""
	parser = argparse.ArgumentParser()  # 创建AregumentParser()对象
	algo = "vime_discount"
	Type = "ordinary"
	game = None
	betm = 4  # 最高下注
	cards = 3
	gameid = 10

	# 调用add_argument()方法添加参数
	parser.add_argument("--algo", help="specify an algorithm: cfrpsrl, mccfros or fictplay", type=str)
	parser.add_argument("--type",
						help="specify the exploration strategy: if algo == cfrpsrl, you can select default, ordinary or random",
						type=str)
	parser.add_argument("--cards", help="specify the exploration strategy", type=int)  # 指定探索策略
	parser.add_argument("--betm", help="the bet maximum", type=int)
	parser.add_argument("--gameid", help=".", type=int)

	# 使用parse_args()解析添加的参数
	args = parser.parse_args()
	algo = "vime_discount"
	Type = "ordinary"
	if args.cards:
		cards = args.cards
	if args.betm:
		betm = args.betm

	if args.gameid:
		gameid = args.gameid
	for _gameid in range(gameid, 11):
		# print("results/leduc_"+str(cards)+"_"+str(betm)+"_"+str(_gameid)+"_"+algo+"_"+Type +".npz")
		try:
			# print("已经运行了try这里")
			print(
				"results/leduc_" + str(cards) + "_" + str(betm) + "_" + str(_gameid) + "_" + algo + "_" + Type + ".npz")
			#  results/leduc_3_4_10_cfrpsrl_default.npz
			a = np.load(
				"results/leduc_" + str(cards) + "_" + str(betm) + "_" + str(_gameid) + "_" + algo + "_" + Type + ".npz")
		except:
			gamepath = "/root/progs/submit/games/leduc_" + str(cards) + "_" + str(betm) + "_" + str(_gameid)
			print(gamepath)
			print(
				"results/leduc_" + str(cards) + "_" + str(betm) + "_" + str(_gameid) + "_" + algo + "_" + Type + ".npz")
			game = Game(path=gamepath + ".npz")
			print("game info: cards: ", cards, "betmaximum: ", betm, "numHists: ", game.numHists, "algorithm info: ",
				  algo, "explore strategy: ", Type)
			# f = open('name.txt', mode='w')
			# f.write("game info: cards: ", cards, "betmaximum: ", betm, "numHists: ", game.numHists, "algorithm info: ",
			# 		algo, "explore strategy: ", Type)
			# logging.basicConfig(level=logging.INFO,
			# 					format='levelname:%(levelname)s filename: %(filename)s '
			# 						   'outputNumber: [%(lineno)d]  thread: %(threadName)s output msg:  %(message)s'
			# 						   ' - %(asctime)s', datefmt='[%d/%b/%Y %H:%M:%S]',
			# 					filename='./loggmsg.log')
			# print("111")


			run(game, solvername=algo, Type=Type,
				resultpath="leduc_" + str(cards) + "_" + str(betm) + "_" + str(_gameid))