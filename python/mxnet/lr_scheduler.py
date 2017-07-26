"""Scheduling learning rate."""
import logging
import math

class LRScheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    """
    def __init__(self, base_lr=0.01):
        self.base_lr = base_lr

    def __call__(self, num_update):
        """Return a new learning rate.

        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.

        Assume the optimizer has udpated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::

            num_update = max([k_i for all i])

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class FactorScheduler(LRScheduler):
    """Reduce the learning rate by a factor for every *n* steps.

    It returns a new learning rate by::

        base_lr * pow(factor, floor(num_update/step))

    Parameters
    ----------
    step : int
        Changes the learning rate for every n updates.
    factor : float, optional
        The factor to change the learning rate.
    stop_factor_lr : float, optional
        Stop updating the learning rate if it is less than this value.
    """
    def __init__(self, step, factor=1, stop_factor_lr=1e-8):
        super(FactorScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while num_update > self.count + self.step:
            self.count += self.step
            self.base_lr *= self.factor
            if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr
                logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                             "change in the future", num_update, self.base_lr)
            else:
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
        return self.base_lr

class MultiFactorScheduler(LRScheduler):
    """Reduce the learning rate by given a list of steps.

    Assume there exists *k* such that::

       step[k] <= num_update and num_update < step[k+1]

    Then calculate the new learning rate by::

       base_lr * pow(factor, k+1)

    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    """
    def __init__(self, step, factor=1):
        super(MultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr
class PolyScheduler(LRScheduler):
    """Reduce learning rate in a poly rate
    Assume the weight has been updated by n times, then the learning rate will
    be
    base_lr * (1 - iter / total_update) ^ power
    Parameters
    ----------
    total_update: int
        total number of weight updates
    power: float
        the rate of learning rate reduction
    """
    def __init__(self, total_update, power=0.9):
        super(PolyScheduler, self).__init__()
        assert isinstance(total_update, int)
        if power > 1.0 or power < 0.0:
                        raise ValueError("Power must be no more than 1 and larger than 0.")
        self.power = power
        self.total_update = total_update

    def __call__(self, num_update):
        """
        Call to schedule current learning rate
        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        return self.base_lr * math.pow(1 - float(num_update) / self.total_update, self.power)

class WarmUpScheduler(LRScheduler):
    """Gradually ramps up the learning rate from a small to a large value

    Learning rate will increase from *begin_lr* to *end_lr* during *begin_iter* ~ *end_iter*

    Assume there exists *k* such that::
       step[k] <= num_update and num_update < step[k+1]

    Then calculate the new learning rate by::
       cur_lr * pow(factor, k+1)

    Parameters
    ----------
    begin_lr, end_lr : float
        The beginning and ending learning rate
    begin_iter, end_iter : int
        WarmUp iterations
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate
    """
    def __init__(self, begin_lr, end_lr, begin_iter, end_iter, step, factor=1):
        super(WarmUpScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.lr_diff = (end_lr - begin_lr) / (end_iter - begin_iter)
        self.cur_lr = begin_lr
        self.end_lr = end_lr
        self.begin_iter = begin_iter
        self.end_iter = end_iter
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.begin_iter <= num_update and num_update < self.end_iter:
            self.cur_lr += self.lr_diff
        elif num_update == self.end_iter:
            self.cur_lr = self.end_lr
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.cur_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.cur_lr)
            else:
                return self.cur_lr
        return self.cur_lr