#!-*- coding:utf-8 -*-

import numpy as np
from typing import Callable

class RectFlow(object):
    def __init__(self) -> None:
        self.name = "rectified_flow"

    def straight_process(self, x0, xT, t):
        xt = (1 - t) * x0 + t * xT

        return xt
    def forward_process(self, xt: np.ndarray, score: np.ndarray, dt: float):
        xt = xt + dt * score

        return xt
    def reverse_process(self, xt: np.ndarray, score: np.ndarray, dt: float):
        xt = xt - dt * score

        return xt
    