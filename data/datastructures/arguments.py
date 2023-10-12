class TrainArgs:

    def __int__(self,num_epochs,resume_global_step,n_gpu):
        self.num_epochs = num_epochs
        self.resume_global_step = resume_global_step
