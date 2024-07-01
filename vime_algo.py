from  vime import VIME
import numpy as np
from LeducHoldem import Game
import time
 


class vime_algo:
	def __init__(self, game, Type="default"):
		self.game = game
		_, rews = self.game.resample()
		self.vime = VIME(rews)


	def updateAll(self):
				

		

		

		_, rewvalidation = self.game.resample()
		self.vime.update_reward(rewvalidation)
		# 需要获取 策略或探索策略时，更新rews。
		# 例如 update 更新self.stgy时
		# 例如使用getExploreStgy更新explore_stgy时


		#getExploreStgy(0, 0, explore_stgy[0], avgstgy[1], (rewvalidation, transvalidation, prob2), (avgchrew, avgchtrans, prob1))
		


def main():
	gamepath="/root/progs/submit/games/leduc_3_4_0"
	print(gamepath)
	game = Game(path=gamepath+".npz")
	algo = vime_algo(game)
	for _ in range(10):
		start_time = time.time()
		algo.updateAll()
		end_time = time.time()
		print("time", end_time - start_time)

if __name__ == "__main__":
    main()