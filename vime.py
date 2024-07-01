import numpy as np
from bnn import BNN
import random
import torch
from collections import deque
import time
import torch.multiprocessing as _mp
from torch.utils import data



class VIME():
    def __init__(self,
                rews,
                eta=1e-2,
                kl_q_len=10
                ):
        self.eta = eta

        self.postions = []
        self.device = try_gpu()
        input_num = 0
               
            
        for i, r in enumerate(rews):
            if not isinstance(r, int):
                input_num += 1
                self.postions.append(i)
        
        self._inputs = torch.pca_lowrank(torch.nn.functional.one_hot(torch.tensor(range(input_num))).float())[0]#.to(self.device)
        self.inited = False
        self.dynamics = BNN(
                n_in=6,
                n_hidden=[64],
                n_out=1,
                n_batches=5,
                learning_rate=0.0001 * 50 * 2,

                )#.to(self.device)
        self.kl_q_len = kl_q_len
        self._kl_mean = deque(maxlen=self.kl_q_len)
        self._kl_std = deque(maxlen=self.kl_q_len)
        self.kl_previous = deque(maxlen=self.kl_q_len)
        
        #mp.set_start_method('spawn')
        # if try_gpu():
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cudnn.deterministic = True

        

    def update_reward(self, rewards):
        if self.inited:

            second_order_update=True
            n_itr_update=1
            use_replay_pool=True
            use_kl_ratio=True
            use_kl_ratio_q=True
            num_processes=64
            kl_batch_size = 1

            _labels = self.postions
            _rews = []
            for i in _labels:
                _rews.append(rewards[i][1])
            
            _inputs = self._inputs
            _targets = torch.tensor(_rews).reshape(len(_rews), 1).float()
            #_targets = _targets_.to(self.device)
        

            kl = torch.zeros(_targets.shape)
            kl_nums = len(kl)
            
            step = int(np.ceil(kl_nums / num_processes))
           

            # _inputs = torch.Tensor(_inputs).to(device)
            # _targets = torch.Tensor(_targets).to(device)

            # KL vector assumes same shape as reward.

            processes = []
            if num_processes == 1:
                compute_intrinsic_reward(self.dynamics, 0, _inputs, _targets, kl, step, kl_batch_size,
                                            second_order_update, n_itr_update, use_replay_pool)
            else:
                for p in range(num_processes):
                    import copy
                    dynamics = copy.deepcopy(self.dynamics)
                    mp = _mp.get_context('spawn')
                    p = mp.Process(target=compute_intrinsic_reward, args=(dynamics,p, _inputs, _targets, kl,
                                                        step, kl_batch_size, second_order_update,
                                                        n_itr_update, use_replay_pool))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()



            # Perform normalization of the intrinsic rewards.
            if use_kl_ratio:
                if use_kl_ratio_q:
                    # Update kl Q
                    self.kl_previous.append(np.median(np.hstack(kl)))
                    previous_mean_kl = np.mean(np.asarray(self.kl_previous))
                    kl = kl / previous_mean_kl

            




            
            # old_acc = 0.
                
            # _out = self.dynamics.pred_fn(_inputs)
            # old_acc += torch.mean((_out - _targets)**2)
            # old_acc /= len(_inputs)

            # print(f"Old Accuracy: {old_acc}")



            _rews = (_targets + self.eta * kl).reshape(kl.shape[0]).tolist()
            for i, pos in enumerate(_labels):
                rewards[pos][0] = -_rews[i]
                rewards[pos][1] = _rews[i]

            # train_start_time = time.time()
            if use_replay_pool:
                ### Train Dynamics use_replay_pool

                batch_size = 500 * 2
                dataset = data.TensorDataset(_inputs, _targets)
                data_iter = data.DataLoader(dataset, batch_size, shuffle=True)
                for inputs, targets in data_iter:
                    self.dynamics.train_fn(inputs, targets)

                
               

                
                # new_acc = 0.
                
                # _out = self.dynamics.pred_fn(_inputs)
                # new_acc += torch.mean((_out - _targets)**2)
                # new_acc /= len(_inputs)

                # print(f"New Accuracy: {new_acc}")
            # train_end_time = time.time()
            # print("train_time", train_end_time-train_start_time)
        self.inited = True      

        
def compute_intrinsic_reward(dynamics, p, _inputs, _targets, kl, step, kl_batch_size, second_order_update, n_itr_update, use_replay_pool):
    for k in range(p * step,
                int((p * step) + np.ceil(step / float(kl_batch_size)))):

        # Save old params for every update.
        dynamics.save_old_params()
        start = k * kl_batch_size
        end = np.minimum(
            (k + 1) * kl_batch_size, _targets.shape[0])

        if start >= end:
            return

        if second_order_update:
            # We do a line search over the best step sizes using
            # step_size * invH * grad
            #                 best_loss_value = np.inf
            for step_size in [0.01]:
                dynamics.save_old_params()
                loss_value = dynamics.train_update_fn(
                    _inputs[start:end], _targets[start:end], second_order_update, step_size)
                loss_value = loss_value.detach()
                kl_div = torch.clamp(loss_value, 0, 1000)
                # If using replay pool, undo updates.
                if use_replay_pool:
                    dynamics.reset_to_old_params()
        else:
            # Update model weights based on current minibatch.
            for _ in range(n_itr_update):
                dynamics.train_update_fn(
                    _inputs[start:end], _targets[start:end], second_order_update)
            # Calculate current minibatch KL.
            kl_div = torch.clamp(
                float(dynamics.f_kl_div_closed_form().detach()), 0, 1000)

        for k in range(start, end):
            
            kl[start] = kl_div

        # If using replay pool, undo updates.
        if use_replay_pool:
            dynamics.reset_to_old_params()  

def resample():
    rewObs = list(map(lambda _u: np.ones(2), range(10000)))
    rews = []
    for i in range(10000):
        if i%2:
            a = np.random.dirichlet(rewObs[i])
            rew = np.array([1 - a[1] * 2, a[1] * 2 - 1])
            rews.append(rew)#(np.random.dirichlet(self.rewObs[i]))
        else:
            rews.append(0)
    return rews

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def main():

    rews = resample()
    vime = VIME(rews)
    for i in range(100):
        rews = resample()
        starttime = time.time()
        
        vime.update_reward(rews)
        endtime = time.time()
        print("time", endtime - starttime)
    


if __name__ == "__main__":
    #mp.set_start_method('spawn')
    main()
