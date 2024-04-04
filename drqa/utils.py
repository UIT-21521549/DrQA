"""This file use for ober see loss when trainning, and update AverageMeter with init weight is 0"""

class  AverageMeter(object):

    def __init__(self):
        self.beta = 0.99 #this is value use to parameter of upadte weight
        self.weight = 0 #init weight
        self.step = 0 #step  means how many times that the meter has been updated
        self.moment = 0


    """state_dict for print dictionary for epochs"""
    @property
    def state_dict(self):
        return vars(self)
    
    """get loss in epochs"""
    def load(self, state_dict):
        for loss, val  in state_dict.item():
            self.__setattr__(loss, val)


    """def update val of weight"""
    def update(self, val):
        self.step += 1
        #it get cal and  then add it to moment
        # for example: init weight is 0 it will get moment = (0.99*moment) + (1-0.99) * val it will get 0.01 new weight and 0.99 from old weight
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.weight = self.moment / (1 - self.beta ** self.t)
