import tensorflow as tf

'''
Optimizers and learning rate schedules.
'''

class VaswaniSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule employed in Vaswani et
    al. (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(VaswaniSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class RecurrentVaswaniSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, cycle_length=10000, warmup_steps=4000):
        super(RecurrentVaswaniSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.cycle_length = cycle_length
        self.warmup_steps = warmup_steps

    def vaswani_step(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) 
        
    def __call__(self, step):
        if step > self.cycle_length:
            return 1/(step//cycle_length+1)**(0.75) * \
                self.vaswani_step(step%self.cycle_length+self.warmup_steps)
        else:
            return self.vaswani_step(step)
        
