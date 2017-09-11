class Meter:
    """Computes and stores the average and total value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = list()
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val.append(val)
        self.sum += val
        self.count += n

    def avg(self):
        return self.sum / self.count