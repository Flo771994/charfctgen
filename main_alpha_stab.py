from datetime import datetime, date
import time
from sklearn.datasets import make_spd_matrix as spd_matrix
import numpy as np
import importlib # needed for testing only to reload scripts importlib.reload(Lossfct)
import matplotlib
# matplotlib.use("Qt5Agg") #otherwise plots not shown with error --- has to be removed for computations on server
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchinfo import summary
# import cmath
import Lossfct
import model_generator
from torch.utils.tensorboard import SummaryWriter
from collections.abc import Mapping
from scipy.special import gamma



# set the device for computation on cpu or gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Cuda available?", torch.cuda.is_available(), "device: ",device)



#set seed for reproducibility
np.random.seed(771994)
torch.manual_seed(771994)

#Set default datatype to make conversion from numpy easier
torch.set_default_dtype(torch.float32)

# Set hyperparameters
widths=[300,50] #Widhts of the hidden layers
iterations=6000
learning_rate_Adam=0.01
batchsize=6000
s_size_approx=6000 #sample size of the W vectors to approximate kernel evaluations
testsize=20000  #number of random variables used to estimate loss on test set
plotsize=100000# sample size of the W vectors to approximate kernel evaluations
s_size_int=6000
t_size_int=6000
k=2 #factor for input dimension
eps=0.1 # parameter of the PRM approximation of the alpha stable random vector


print("batchsize:",batchsize)

# Define the characteristic function that is supposed to be simulated
dim=5
rho=0.5 # correlation parameter for the normal distribution which is normalized to the l-2 sphere
alpha=0.5
shift=torch.tensor(np.ones(dim),dtype=torch.float32).to(device)
shift_cpu=shift.to("cpu").detach().numpy()
c=np.eye(dim)+rho*np.ones((dim,dim))##np.eye(dim)
m=np.zeros(dim)
mean=torch.tensor(m,dtype=torch.float32).to(device)
mean_cpu=torch.tensor(m,dtype=torch.float32).to("cpu")
cov_cpu=torch.tensor(c,dtype=torch.float32).to("cpu")
cov=torch.tensor(c,dtype=torch.float32).to(device)

print("alpha=", alpha,"cov=",cov, "mean=",mean, "shift",shift)

### sample to approximate the integral in the characteristic function of the stable distribution
gauss_sample=torch.tensor(np.random.multivariate_normal(mean=m,cov=c,size=s_size_int),dtype=torch.float32).to(device)
norm=torch.pow(torch.sum(torch.pow(gauss_sample,2),dim=1),-1/2).to(device)
gauss_sample_norm=torch.mul(gauss_sample,norm.view(-1,1))


def charfct_vec(W):
    return Lossfct.char_fct_alpha_stab(W,alpha,shift,gauss_sample_norm,device)
def charfct_vec_cpu(W):
    return Lossfct.char_fct_alpha_stab(W,alpha,shift.to("cpu"),gauss_sample_norm.to("cpu"),"cpu")


# Define the kernelmatrix corresponding to the kernel evaluations and the samples to approximate the kernel evaluations
bandwidth=[0.02,0.5,1,5,100] #must be a list
def kernelmatrix(X):
    return Lossfct.gaussian_kernelmatrix(X,device,bandwidth) #needs to be compatible with the distribution that is sampled in HERE
def kernelmatrix_cpu(X):
    return Lossfct.gaussian_kernelmatrix(X,"cpu",bandwidth) #needed to conduct evaluation on cpu
def sample_kernel(size):
    return Lossfct.kernelsample_gaussian(size,device,dim,bandwidth) # HERE
def sample_kernel_cpu(size):
    return Lossfct.kernelsample_gaussian(size,"cpu",dim,bandwidth) #needed to conduct evaluation on cpu


# Define the model
hidden_architecture=[]
for w in widths:
    hidden_architecture.append( (w,torch.nn.ReLU()) )

model=model_generator.MLP(k*dim,dim,hidden_architecture).to(device)
summary(model,input_data=torch.tensor(np.zeros(shape=(1,k*dim)),dtype=torch.float32),device=device)

#set optimizer
# optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate_SGD )
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate_Adam )
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[2000,4000],gamma=0.1)


# Create summarywriter for tensorboard
writer = SummaryWriter()

################# Training loop
now=time.time()
for epoch in np.arange(iterations):

    # Create the random input data
    U=np.matrix(np.random.normal(size=k*dim * batchsize)).reshape(batchsize, k*dim)
    U=torch.tensor(U,dtype=torch.float32).to(device)
    # Sample to approximation the dim-dimensional kernel and the alphs-stab charfct (every 20th iteration)
    if epoch%20==0:
        W = sample_kernel(s_size_approx)
        gauss_sample=torch.tensor(np.random.multivariate_normal(mean=m,cov=c,size=s_size_int),dtype=torch.float32).to(device)
        norm=torch.pow(torch.sum(torch.pow(gauss_sample,2),dim=1),-1/2).to(device)
        gauss_sample_norm=torch.mul(gauss_sample,norm.view(-1,1)).to(device)

        def charfct_vec(W):
            return Lossfct.char_fct_alpha_stab(W,alpha,shift,gauss_sample_norm,device)#Lossfct.gaussian_char_fct_vec(W,mean,cov,device)
        
    # Make predictions
    samples=model(U)

    # Empty gradient
    optimizer.zero_grad()
    # Calculate Loss
    loss = Lossfct.lossfunction_vec(samples, kernelmatrix, charfct_vec, W, device, False) #True means using exact representation of kernel where possible
    # Calculate gradient
    loss.backward()
    # Take one SGD step
    optimizer.step()
    scheduler.step() 
    #logging for tensorboard
    elapsed_time=time.time()-now
    writer.add_scalar('Loss_epoch', loss.item(), epoch) 
    writer.add_scalar('Loss_time', loss.item(), elapsed_time) 
    

    if epoch%25==0:
        print("epoch: ",epoch," loss=",loss)
        
################# end of training loop
print("Training time: ",time.time()-now)

def charfct_vec_cpu(W):
        return Lossfct.char_fct_alpha_stab(W,alpha,shift.to("cpu"),gauss_sample_norm.to("cpu"),"cpu")

# Set model to evaluation mode
model.eval()

with torch.no_grad():
    #visualize the model
    writer.add_graph(model, torch.tensor(np.random.normal(size=k*dim),dtype=torch.float32).to(device))
    #start tensorboard with tensorboard --logdir=runs

    #Send model to cpu for calculation on larger matrices
    model.cpu()
    
    # Calculate estimate of true loss by using a large sample size and including the term in the MMD estimate that does not depend on the generator
    U=np.matrix(np.random.normal(size=k*dim * testsize)).reshape(testsize, k*dim)
    U=torch.tensor(U,dtype=torch.float32).to("cpu")
    samples=model(U)
    W=sample_kernel_cpu(testsize).to("cpu") #Used to approximate d-dimensional kernel
    
    approx_loss= Lossfct.lossfunction_vec(samples,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

    print("The approximate loss of the final model is equal to ",approx_loss,". It should be close to 0. Otherwise the model has not learned the distribution correctly")


# Save the model
widthstring='_'.join([str(w) for w in widths])
filenamemodel="saved_models/alpha_stab_pos_cor_"+str(alpha)+"_"+str(dim)+"-dim_"+widthstring+"widths_"+date.today().strftime('%d-%m-%Y')+".pth"
filenamecovandmean="saved_models/params_alpha_stab_pos_cor_"+str(alpha)+"_"+str(dim)+"-dim_"+date.today().strftime('%d-%m-%Y'+".npz")
torch.save([model.kwargs,model.state_dict()], filenamemodel)
np.savez(filenamecovandmean,cov=c,mean=m,alpha=alpha,shift=shift_cpu)


############################## Comparison with simulation by cutting off sum over atoms of PRM inside an eps ball around 0
with torch.no_grad():
    # Define the sample to approximate the intergal in the loss function to evaluate the final loss
    gauss_sample=torch.tensor(np.random.multivariate_normal(mean=m,cov=c,size=t_size_int),dtype=torch.float32).to(device)
    norm=torch.pow(torch.sum(torch.pow(gauss_sample,2),dim=1),-1/2).to(device)
    gauss_sample_norm=torch.mul(gauss_sample,norm.view(-1,1))
    #update the charactersitic functions
    def charfct_vec(W):
        return Lossfct.char_fct_alpha_stab(W,alpha,shift,gauss_sample_norm,device)#Lossfct.gaussian_char_fct_vec(W,mean,cov,device)
    def charfct_vec_cpu(W):
        return Lossfct.char_fct_alpha_stab(W,alpha,shift.to("cpu"),gauss_sample_norm.to("cpu"),"cpu")#Lossfct.gaussian_char_fct_vec(W,mean_cpu,cov_cpu,"cpu") #needed to conduct evaluation on cpu

    

    c_alpha=np.sqrt(np.pi)*(2**(-alpha))*(alpha**(-1))*gamma( (2-alpha)/2 )*((gamma( (1+alpha)/2 ))**(-1)) #constant needed to normalize the PRm rep
    factor=alpha*c_alpha
    
    # sampling algorithm
    now=time.time()
    sample_prm=np.zeros((testsize,dim))
    counter=0
    # need compensator for the correct approximation via the Levy Ito decomposition
    compensator=np.zeros(dim)
    if alpha>1:
        compensator= -eps**(1-alpha)/(c_alpha*(1-alpha))*torch.sum(gauss_sample_norm,dim=0).to("cpu").detach().numpy()/t_size_int
    if alpha==1:
        compensator= -np.log(eps)/c_alpha*torch.sum(gauss_sample_norm,dim=0).to("cpu").detach().numpy()/t_size_int
    for i in np.arange(testsize):
        sum=0
        delta=np.random.exponential(size=1)
        while (factor*delta)**(-1/alpha)>eps:
            g_sample_prm=np.random.multivariate_normal(mean=m,cov=c,size=1)
            n_prm=np.sum(g_sample_prm**2)**(-1/2)
            g_sample_n_prm= g_sample_prm*n_prm
            obs=g_sample_n_prm*(factor*delta)**(-1/alpha)
            sum=sum+obs
            delta=delta+np.random.exponential(size=1)
            counter=counter+1
        sample_prm[i,]=sum+shift_cpu-compensator
        if np.sum(np.isnan(compensator))>0:
                print("nan detected")
        if np.sum(np.isnan(sum))>0:
                print("nan detected")
    training_time=time.time()-now

    sample_prm_torch=torch.tensor(sample_prm,dtype=torch.float32).to("cpu")
    approx_loss_prm= Lossfct.lossfunction_vec(sample_prm_torch,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

    print("The approximate loss of the PRM sample is equal to ",approx_loss_prm,". It should be close to 0. Otherwise the model has not learned the distribution correctly")

    dist_to_true_sam=0
    for b in bandwidth:
        dist_to_true_sam=Lossfct.MMD_equal_case(samples,device,b)- Lossfct.MMD_mixed_case(samples,sample_prm_torch,device,b) +Lossfct.MMD_equal_case(sample_prm_torch,device,b)
    dist_to_true_sam=dist_to_true_sam/len(bandwidth)
    
    print("The estimated MMD distance of a sample from the model tothe sample from the PRM approximation is ",dist_to_true_sam,". It should be close to 0. Otherwise the either of the two approximations does not work")

    print("Training time: ",training_time)
    print("average time per sample", training_time/testsize)
    print("average number of atoms per sample", counter/testsize)
    print("expected number of atoms per sample", (1/eps)**alpha/factor)

# Load model
# kwargs, state_dict =torch.load(filenamemodel)
# model = model_generator.MLP(**kwargs)
# model.load_state_dict(state_dict)
# model.eval()
# print(model)