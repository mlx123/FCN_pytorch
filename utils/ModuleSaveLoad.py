import  torch


def ModelSave(model,optim,saveRoot,epoch = 0,iteration = 0,archName = None,bestMeanIu = 0):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'epoch':epoch,
        'iteration': iteration,
        'arch': archName,
        'best_mean_iu':bestMeanIu
        }, saveRoot)



def ModelLoad(loadRoot,model,optim = None):
    if loadRoot:
        checkpoint = torch.load(loadRoot)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optim is not None:
            optim.load_state_dict(checkpoint['optim_state_dict'])

        start_epoch =     checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        best_mean_iu =    checkpoint[ 'best_mean_iu']


        return start_epoch,start_iteration,best_mean_iu
    else:
        pass