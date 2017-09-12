import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


# loss,prec1,prec3 are Meter instances
# loss,prec@k displays averaged loss/prec
# of batches within an epoch

class Plot(object):

    def __init__(self, model_name):
        plt.clf()
        self.figure, self.axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey="row",facecolor='W')

        map(lambda x: x.set_xlabel("epoch"), self.axes)
        map(lambda x: x.set_ylabel("Average Batch Loss"), self.axes[0])
        map(lambda x: x.set_ylabel("Average Batch Precision@k"), self.axes[1])
        plt.title(model_name)

        self.l1, = self.axes[0, 0].plot([], [], 'bo-', label="train-loss")
        self.l2, = self.axes[0, 1].plot([], [], 'bo-', label="val-loss")
        self.l3, = self.axes[1, 0].plot([], [], 'ro-', label="val-prec1")
        self.l4, = self.axes[1, 1].plot([], [], 'ro-', label="val-prec3")

        map(lambda x: x.set_ylim(0, 7), self.axes[0])
        map(lambda x: x.set_ylim(0, 1), self.axes[1])
        self.lines = [self.l1,self.l2,self.l3,self.l4]
        self.figure.legend(self.lines,[l.get_label() for l in self.lines])
        plt.show(block=False)

    def update_statistics(self, ith_epoch, train_loss, val_loss, prec1, prec3):
        map(lambda x: x.set_xlim(0, ith_epoch), self.axes)
        # map(lambda x: x.set_xdata(np.append(x.get_xdata(), ith_epoch)), self.lines)
        self.l1.set_xdata(np.append(self.l1.get_xdata(),ith_epoch))
        self.l2.set_xdata(np.append(self.l2.get_xdata(), ith_epoch))
        self.l3.set_xdata(np.append(self.l3.get_xdata(), ith_epoch))
        self.l4.set_xdata(np.append(self.l4.get_xdata(), ith_epoch))

        self.l1.set_ydata(np.append(self.l1.get_ydata(), train_loss))
        self.l2.set_ydata(np.append(self.l2.get_ydata(), val_loss))
        self.l3.set_ydata(np.append(self.l3.get_ydata(), prec1))
        self.l4.set_ydata(np.append(self.l4.get_ydata(), prec3))

        print(self.l1.get_ydata())
        print(self.l1.get_xdata())
        print(self.l2.get_ydata())
        print(self.l2.get_xdata())
        print(self.l3.get_ydata())
        print(self.l3.get_xdata())
        print(self.l4.get_ydata())
        print(self.l4.get_xdata())

        plt.show()
        





