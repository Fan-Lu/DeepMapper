import copy

import numpy as np
import control as ct
import torch

import torch.nn.functional as F

class Wound(object):

    def __init__(self):
        self.state_space = np.array([1, 0, 0, 0])

        self.space_EF = np.linspace(0, 1, 20)
        self.space_Flx = np.linspace(0, 1, 10)

        self.action_space = np.array([(a, b) for a in self.space_EF for b in self.space_Flx])

        # self.action_space = np.linspace(0, 1, 20)

        kh, ki, kp = 0.5, 0.3, 0.1
        self.ts = 0.4
        self.A = np.array([[-kh,   0,   0, 0],
                           [ kh, -ki,   0, 0],
                           [ 0,   ki, -kp, 0],
                           [ 0,    0,  kp, 0]]) * self.ts + np.identity(4)
        self.B = np.array([[-0.1, 0, 0, 0],
                           [0.1, -0.01, 0, 0],
                           [0, 0.01, -0.21, 0],
                           [0, 0, 0.21, 0]]) + 1e-3

        self.B_EF = np.array([[0,    0, 0, 0],
                              [0, -0.1, 0, 0],
                              [0,  0.1, 0, 0],
                              [0,    0, 0, 0]])

        self.B_Flx = np.array([[0, 0,    0, 0],
                              [0, 0,    0, 0],
                              [0, 0, -0.1, 0],
                              [0, 0,  0.1, 0]])

        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]])
        self.R = np.eye(self.B.shape[1])
        self.K, S, E = ct.dlqr(self.A, self.B, self.Q, self.R)

    def noctrTraj(self):
        zk = self.reset()
        zbuf = [zk]

        for i in range(60):
            zkp1, reward, done, info = self.step(0)
            zbuf.append(zkp1)
        return np.array(zbuf).reshape(-1, 4)

    def funcEF(self, z, a):
        pass

    def reset(self):
        self.z = np.array([1.0, 0, 0, 0])
        return self.z

    def step(self, a, comb=1.0):
        a = self.action_space[a]
        alt_a = self.funcEF(self.z, a)
        z_ = (self.A + self.B + alt_a - 1e-3) @ self.z
        # if comb > 0:
        #     try:
        #         self.K, S, E = ct.dlqr(self.A, self.B + alt_a, self.Q, self.R)
        #     except:
        #         pass
        z_opt = (self.A + alt_a) @ self.z + self.B @ (-self.K @ self.z)
        dist = np.linalg.norm(z_ - z_opt)
        # dist = F.binary_cross_entropy(torch.softmax(torch.from_numpy(z_).view(-1, 4), dim=1),
        #                               torch.softmax(torch.from_numpy(z_opt).view(-1, 4), dim=1)).numpy()
        reward = comb * np.exp(-100.0 * dist) - (1 - comb) * int(self.z[3] <= 0.95)
        # reward = 0.9 * np.exp(-100.0 * dist) - 0.1 * int(self.z[3] <= 0.95)
        done = False
        # done = self.z[3] >= 0.95
        self.z = copy.deepcopy(z_)

        return z_, reward, done, None


class WoundEF(Wound):

    def __init__(self):
        super(WoundEF, self).__init__()
        self.action_space = copy.deepcopy(self.space_EF)

        # self.B_EF = np.array([[0,    0, 0, 0],
        #                       [0, -0.15, 0, 0],
        #                       [0,  0.15, 0, 0],
        #                       [0,    0, 0, 0]])

    def funcEF(self, z, e):
        if z[3] < 0.5:
            return -self.B_EF * np.sin(e * 2 * np.pi)
        return self.B_EF * np.sin(e * 2 * np.pi)

class WoundFlx(Wound):

    def __init__(self):
        super(WoundFlx, self).__init__()
        self.action_space = copy.deepcopy(self.space_Flx)

        self.B_Flx = np.array([[0, 0,    0, 0],
                              [0, 0,    0, 0],
                              [0, 0, -0.35, 0],
                              [0, 0,  0.35, 0]])


    def funcEF(self, z, e):
        if z[3] < 0.5:
            return -self.B_Flx * np.sin(e * 2 * np.pi)
        return self.B_Flx * np.sin(e * 2 * np.pi)

    def step(self, a, comb=1.0):
        a = self.action_space[a]
        alt_a = self.funcEF(self.z, a)
        z_ = (self.A + self.B + alt_a - 1e-3) @ self.z
        # if comb > 0:
        #     try:
        #         self.K, S, E = ct.dlqr(self.A, self.B + alt_a, self.Q, self.R)
        #     except:
        #         pass
        z_opt = self.A @ self.z + self.B @ (-self.K @ self.z)
        dist = np.linalg.norm(z_ - z_opt)
        # dist = F.binary_cross_entropy(torch.softmax(torch.from_numpy(z_).view(-1, 4), dim=1),
        #                               torch.softmax(torch.from_numpy(z_opt).view(-1, 4), dim=1)).numpy()
        reward = comb * np.exp(-100.0 * dist) - (1 - comb) * int(self.z[3] <= 0.95)
        # reward = 0.9 * np.exp(-100.0 * dist) - 0.1 * int(self.z[3] <= 0.95)
        done = False
        # done = self.z[3] >= 0.95
        self.z = copy.deepcopy(z_)

        return z_, reward, done, None
class WoundComb(Wound):

    def __init__(self):
        super(WoundComb, self).__init__()

        ef_space = np.linspace(0, 1, 10)
        flx_space = np.linspace(0, 1, 10)
        self.action_space = np.array([(a, b) for a in ef_space for b in flx_space])


    def funcEF(self, z, a):
        e, f = a
        if z[3] < 0.5:
            return -(self.B_EF * np.sin(e * 2 * np.pi) + self.B_Flx * np.sin(f * 2 * np.pi))
        return self.B_EF * np.sin(e * 2 * np.pi) + self.B_Flx * np.sin(f * 2 * np.pi)