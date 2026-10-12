"""Serialize backward, DP/CP reduction and later TP reduction at step end."""


def fenced_grad_sync(finish_grad_sync, rendezvous):
    def finish(*args, **kwargs):
        rendezvous()
        result = finish_grad_sync(*args, **kwargs)
        rendezvous()
        return result

    return finish
