import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback

class LRFinder(Callback):
    
    '''
    Callback for finding the optimal learning rate range for your model + dataset. 
    
    Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-4, 
                                 max_lr=1e-1, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            lr_finder.plot_loss()
        ```
    
    Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    References
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, optimizer_name, base_model_tag, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.optimizer_name = optimizer_name
        self.base_model_tag = base_model_tag
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        

    def clr(self):
        '''
        Calculate the learning rate.
        '''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        

    def on_train_begin(self, logs=None):
        '''
        Initialize the learning rate to the minimum value at the start of training.
        '''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        

    def on_batch_end(self, epoch, logs=None):
        '''
        Record previous batch statistics and update the learning rate.
        '''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 

    def plot_lr(self):
        '''
        Helper function to quickly inspect the learning rate schedule.
        '''
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(self.history['iterations'], self.history['lr'])
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning rate')
        ax.set_title(self.base_model_tag + " / " + self.optimizer_name)
        plt.show()
 
        
    def plot_loss(self):
        '''
        Helper function to quickly observe the learning rate experiment results.
        '''
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(self.history['lr'], self.history['loss'])
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        ax.set_title(self.base_model_tag + " / " + self.optimizer_name)
        plt.show()