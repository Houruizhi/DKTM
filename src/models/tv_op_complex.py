import torch.nn.functional as F
def nabla_x(x):
    '''
    x: (n,c,h,w,2)
    '''
    x = F.pad(x, (0,0,1,0), mode='constant', value=0.0)
    return x[...,1:,:]-x[...,0:-1,:]
    
def nabla_y(x):
    '''
    x: (n,c,h,w,2)
    '''
    x = F.pad(x, (0,0,0,0,1,0), mode='constant', value=0.0)
    return x[...,1:,:,:]-x[...,0:-1,:,:]

def nablat_x(x):
    '''
    x: (n,c,h,w,2)
    '''
    x = F.pad(x, (0,0,0,1), mode='constant', value=0.0)
    return x[...,0:-1,:]-x[...,1:,:]

def nablat_y(x):
    '''
    x: (n,c,h,w,2)
    '''
    x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=0.0)
    return x[...,0:-1,:,:]-x[...,1:,:,:]