import matplotlib.pyplot as plt
import numpy as np
import os

# loss,prec1,prec3 are Meter instances
# loss,prec@k displays averaged loss/prec
# of batches within an epoch

class Plot(object):

    def __init__(self, model_name):

        self.model = model_name

    def save_stats(self, epochs,train_loss,val_loss,train_prec1,train_prec3,val_prec1,val_prec3):

        self.epochs = epochs
        self.train_loss = train_loss.val
        self.val_loss = val_loss.val
        self.val_prec1 = val_prec1.val
        self.val_prec3 = val_prec3.val
        self.train_prec1 = train_prec1
        self.train_prec3 = train_prec3
        np.savez("{}.npz".format(self.model),
                 epochs=self.epochs,
                 train_loss=self.train_loss,
                 val_loss=self.val_loss,
                 val_prec1=self.val_prec1,
                 val_prec3=self.val_prec3,
                 train_prec1=self.train_prec1,
                 train_prec3=self.train_prec3)

    def plot_stats(self):

        if os.path.exists("{}.npz".format(self.model)):

            stats = np.load("{}.npz" % self.model)

            plt.clf()
            figure, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey="row", facecolor='W')

            map(lambda x: x.set_xlabel("epoch"), axes)
            map(lambda x: x.set_ylabel("Average Batch Loss"), axes[0])
            map(lambda x: x.set_ylabel("Average Batch Precision@k"), axes[1])
            plt.title(self.model_name)

            x = list(range(stats['epochs']))
            l1, = axes[0, 0].plot(x, stats['train_loss'], 'bo-', label="train-loss")
            l2, = axes[0, 1].plot(x, stats['val_loss'], 'bo-', label="val-loss")
            l3, = axes[1, 0].plot(x, stats['val_prec1'], x , stats['train_prec1'], 'ro-', label="prec1")
            l4, = axes[1, 1].plot(x, stats['val_prec3'], x , stats['train_prec3'], 'ro-', label="prec3")

            map(lambda x: x.set_ylim(0, 7), axes[0])
            map(lambda x: x.set_ylim(0, 1), axes[1])
            lines = [l1, l2, l3, l4]
            figure.legend(lines, [l.get_label() for l in lines])

            map(lambda x: x.set_xlim(0,stats['epochs']), axes)

            plt.show()
        else:
            raise ValueError('specify the model name trained before')






