# import matplotlib.pyplot as plt
import numpy as np
# import os

# loss,prec1,prec3 are Meter instances
# loss,prec@k displays averaged loss/prec
# of batches within an epoch

class Plot(object):

    def __init__(self, model_name,depth,lr,batchSize):

        self.model = model_name
        self.depth = depth
        self.lr = lr
        self.batchSize = batchSize

    def save_stats(self, epochs,train_loss,val_loss,train_prec1,train_prec3,val_prec1,val_prec3):

        self.epochs = epochs
        self.train_loss = train_loss.val
        self.val_loss = val_loss.val
        self.val_prec1 = val_prec1.val
        self.val_prec3 = val_prec3.val
        self.train_prec1 = train_prec1.val
        self.train_prec3 = train_prec3.val
        np.savez("{}_{}_{}_{}.npz".format(self.model,self.depth,self.lr,self.batchSize),
                 epochs=self.epochs,
                 train_loss=self.train_loss,
                 val_loss=self.val_loss,
                 val_prec1=self.val_prec1,
                 val_prec3=self.val_prec3,
                 train_prec1=self.train_prec1,
                 train_prec3=self.train_prec3)

    """def plot_stats(self):

        if os.path.exists("{}.npz".format(self.model)):

            stats = np.load("{}.npz".format(self.model))
            plt.clf()
            figure, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey="row", facecolor='W')

            plt.title(self.model)

            axes[0,0].set_ylabel("train loss")
            axes[0,1].set_ylabel("val loss")
            axes[1,0].set_ylabel("Precision@1")
            axes[1,1].set_ylabel("Precision@3")
            axes[0,0].set_xlabel("epoch")
            axes[0,1].set_xlabel("epoch")
            axes[1,0].set_xlabel("epoch")
            axes[1,1].set_xlabel("epoch")
            x = list(range(stats['epochs']))
            l1, = axes[0, 0].plot(x, stats['train_loss'], 'b-', label="train-loss")
            axes[0,0].legend(l1.get_label())
            l2, = axes[0, 1].plot(x, stats['val_loss'], 'b-', label="val-loss")
            axes[0,1].legend(l2.get_label())
            l3, = axes[1, 0].plot(x, stats['val_prec1'],'r-', label="val-prec1")
            l4, = axes[1, 0].plot(x, stats['train_prec1'], 'g-', label="train-prec1")
            axes[1,0].legend([l3,l4],[l.get_label() for l in [l3,l4]])
            l5, = axes[1, 1].plot(x, stats['val_prec3'],'r-', label="val-prec3")
            l6, = axes[1, 1].plot(x, stats['train_prec3'], 'g-',label="train_prec3")
            axes[1,1].legend([l5,l6],[l.get_label() for l in [l5,l6]])

            plt.tight_layout()

            plt.show()
        else:
            raise ValueError('specify the model name trained before')
        """






