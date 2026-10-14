# -*- coding: utf-8 -*-


class SingleProcessDistributedMixin:
    rank = 0
    size = 1
    is_distributed = False

    def barrier(self):
        return None

    def allreduce(self, x, op="sum"):
        return x

    def broadcast(self, x, root=0):
        return x

    def gather(self, x, root=0):
        return [x]

    def allgather(self, x):
        return [x]
