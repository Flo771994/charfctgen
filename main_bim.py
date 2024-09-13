from datetime import datetime, date
import time
from sklearn.datasets import make_spd_matrix as spd_matrix
import numpy as np
import importlib # needed for testing only to reload scripts importlib.reload(Lossfct)
import matplotlib
# matplotlib.use("Qt5Agg") #otherwise plots not shown with error --- has to be removed for computations on server
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchinfo import summary
# import cmath
import Lossfct
import model_generator
from torch.utils.tensorboard import SummaryWriter
from collections.abc import Mapping


# set the device for computation on cpu or gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Cuda available?", torch.cuda.is_available(), "device: ",device)


# set seed for reproducibility
np.random.seed(771994)
torch.manual_seed(771994)

# set default datatype to make conversion from numpy easier
torch.set_default_dtype(torch.float32)

######################### set hyperparameters
widths=[300,50] #Widhts of the hidden layers
iterations=6000
learning_rate_Adam=0.01
batchsize=6000
s_size_approx=6000 #sample size of the W vectors to approximate kernel evaluations
testsize=20000  #number of random variables used to estimate loss on test set
plotsize=100000
k=2 #factor for input dimension

print("batchsize:",batchsize)
#########################


########################## Define the characteristic function that is supposed to be simulated
dim=5
n_mix=10
cov_list=[]
cov_list_torch=[]
cov_list_cpu=[]
mean_list=[]
mean_list_torch=[]
mean_list_cpu=[]

# randomly create parameters for gaussian mixture components
for j in np.arange(n_mix):
    c=spd_matrix(dim)
    cov_cpu=torch.tensor(c,dtype=torch.float32).to("cpu")
    cov=torch.tensor(c,dtype=torch.float32).to(device)
    cov_list.append(c)
    cov_list_torch.append(cov)
    cov_list_cpu.append(cov_cpu)

    m=np.random.uniform(-10,10,dim)
    mean=torch.tensor(m,dtype=torch.float32).to(device)
    mean_cpu=torch.tensor(m,dtype=torch.float32).to("cpu")
    mean_list.append(m)
    mean_list_torch.append(mean)
    mean_list_cpu.append(mean_cpu)

print("Gaussian mixture model","means:",mean_list," covariances:",cov_list)


def charfct_vec(W):
    l=Lossfct.gaussian_char_fct_vec(W,mean_list_torch[0],cov_list_torch[0],device)/n_mix
    for j in np.arange(n_mix-1):
        l= l+Lossfct.gaussian_char_fct_vec(W,mean_list_torch[j+1],cov_list_torch[j+1],device)/n_mix       # Lossfct.gaussian_char_fct_vec(W,mean,cov,device)/2+Lossfct.gaussian_char_fct_vec(W,mean1,cov1,device)/2#Lossfct.t_char_fct_vec(W,mean,cov,dgf,device)#Lossfct.char_fct_rational_quad_vec(W,alpha,beta,mean,device)
    return l

def charfct_vec_cpu(W):
    l=Lossfct.gaussian_char_fct_vec(W,mean_list_cpu[0],cov_list_cpu[0],"cpu")/n_mix
    for j in np.arange(n_mix-1):
        l= l+Lossfct.gaussian_char_fct_vec(W,mean_list_cpu[j+1],cov_list_cpu[j+1],"cpu")/n_mix      
    return l
    #return Lossfct.gaussian_char_fct_vec(W,mean_cpu,cov_cpu,"cpu")/2+Lossfct.gaussian_char_fct_vec(W,mean1_cpu,cov1_cpu,"cpu")/2

################################


################################# Define the kernelmatrix corresponding to the kernel evaluations and the samples to approximate the kernel evaluations
bandwidth=[0.02,0.5,1,5,100] #must be a list
def kernelmatrix(X):
    return Lossfct.gaussian_kernelmatrix(X,device,bandwidth) #needs to be compatible with the distribution that is sampled in HERE
def kernelmatrix_cpu(X):
    return Lossfct.gaussian_kernelmatrix(X,"cpu",bandwidth) #needed to conduct evaluation on cpu
def sample_kernel(size):
    return Lossfct.kernelsample_gaussian(size,device,dim,bandwidth) # HERE
def sample_kernel_cpu(size):
    return Lossfct.kernelsample_gaussian(size,"cpu",dim,bandwidth) #needed to conduct evaluation on cpu
################################

################################# Define the model
hidden_architecture=[]
for w in widths:
    hidden_architecture.append( (w,torch.nn.ReLU()) )

model=model_generator.MLP(k*dim,dim,hidden_architecture).to(device)
summary(model,input_data=torch.tensor(np.zeros(shape=(1,k*dim)),dtype=torch.float32),device=device)

# set optimizer
# optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate_SGD )
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate_Adam )
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[2000,4000],gamma=0.1)


# create summarywriter for tensorboard
writer = SummaryWriter()
###############################



################################ Training loop
now=time.time()
for epoch in np.arange(iterations):

    # Create the random input data
    U=np.matrix(np.random.normal(size=k*dim * batchsize)).reshape(batchsize,k* dim)
    U=torch.tensor(U,dtype=torch.float32).to(device)
    # Sample to approximation the dim-dimensional kernel (every 10th iteration)
    if epoch%20==0:
        W = sample_kernel(s_size_approx)
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

 
    # logging for tensorboard
    # track mmd distance to an exact sample
    with torch.no_grad():
        category=np.random.choice(n_mix,size=batchsize)
        true_sample=np.zeros((batchsize,dim))
        for j in np.arange(n_mix):
            indicator=np.equal(category,j)
            indicator=np.tile(indicator.reshape((batchsize,1)),(1,dim))
            true_sample=true_sample+np.multiply(indicator,np.random.multivariate_normal(mean=mean_list[j],cov=cov_list[j],size=batchsize))

        true_sample_torch=torch.tensor(true_sample,dtype=torch.float32).to(device)
        dist_to_true_sam=0
        for b in bandwidth:
            dist_to_true_sam=Lossfct.MMD_equal_case(samples,device,b)- Lossfct.MMD_mixed_case(samples,true_sample_torch,device,b) +Lossfct.MMD_equal_case(true_sample_torch,device,b)
        dist_to_true_sam=dist_to_true_sam/len(bandwidth)

        elapsed_time=time.time()-now
        writer.add_scalar('MMD distance to exact sample', dist_to_true_sam, epoch)  
        writer.add_scalar('MMD distance to exact sample', dist_to_true_sam, elapsed_time)   
        writer.add_scalar('Loss', loss.item(), epoch) 
        writer.add_scalar('Loss', loss.item(), elapsed_time) 

    if epoch%25==0:
        print("epoch: ",epoch," loss=",loss)
    if epoch%200==0:
        with torch.no_grad():
            # track if the mean of the sample converges to the true mean
            model.eval()
            U = np.matrix(np.random.normal(size=k*dim * plotsize)).reshape(plotsize, k*dim)
            U = torch.tensor(U, dtype=torch.float32).to("cpu").requires_grad_(False)
            model.to("cpu") # GPU cannot handle large sample sizes
            samples = model(U)
            print(np.mean(samples.cpu().detach().numpy(), axis=0)-np.mean(mean_list,axis=0))
            model.to(device)
            model.train()

print("Training time: ",time.time()-now)      
################################## end of training loop



################################## Evaluate the final model
model.eval()

with torch.no_grad():
    # visualize the model
    writer.add_graph(model, torch.tensor(np.random.normal(size=k*dim),dtype=torch.float32).to(device))
    #start tensorboard with tensorboard --logdir=runs

    # send model to cpu for calculation on larger matrices
    model.cpu()
    
    ########### Calculate estimate of the loss by using a large sample size and including the term in the MMD estimate that does not depend on the generator
    U=np.matrix(np.random.normal(size=k*dim * testsize)).reshape(testsize, k*dim)
    U=torch.tensor(U,dtype=torch.float32).to("cpu")
    samples=model(U)
    W=sample_kernel_cpu(testsize).to("cpu") #Used to approximate d-dimensional kernel
    
    approx_loss= Lossfct.lossfunction_vec(samples,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

    print("The approximate loss of the final model is equal to ",approx_loss,". It should be close to 0. Otherwise the model has not learned the distribution correctly")


    ########## benchmark the estimated loss of the final model with the estimated loss of an exact sample
    category=np.random.choice(n_mix,size=testsize)
    true_sample=np.zeros((testsize,dim))
    for j in np.arange(n_mix):
        indicator=np.equal(category,j)
        indicator=np.tile(indicator.reshape((testsize,1)),(1,dim))
        true_sample=true_sample+np.multiply(indicator,np.random.multivariate_normal(mean=mean_list[j],cov=cov_list[j],size=testsize))

    true_sample_torch=torch.tensor(true_sample,dtype=torch.float32).to("cpu")
    ex_loss= Lossfct.lossfunction_vec(true_sample_torch,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

    print("The approximate loss of the exact sample is equal to ",ex_loss,". It should be close to 0. Otherwise the model has not learned the distribution correctly")

    dist_to_true_sam=0
    for b in bandwidth:
        dist_to_true_sam=Lossfct.MMD_equal_case(samples,device,b)- Lossfct.MMD_mixed_case(samples,true_sample_torch,device,b) +Lossfct.MMD_equal_case(true_sample_torch,device,b)
    dist_to_true_sam=dist_to_true_sam/len(bandwidth)
    
    print("The estimated MMD distance of a sample from the model to an exact sample is ",dist_to_true_sam,". It should be close to 0. Otherwise the model has not learned the distribution correctly")


    ########## create some plots
    U=np.matrix(np.random.normal(size=k*dim * plotsize)).reshape(plotsize, k*dim)
    U=torch.tensor(U,dtype=torch.float32).to("cpu")
    samples=model(U)
    
    # Print estimated mean for comparison
    print(np.mean(samples.cpu().detach().numpy(), axis=0)-np.mean(mean_list,axis=0) )

    #### plots for model samples
    plot=sns.jointplot(x=samples.cpu().detach().numpy()[:,0],y=samples.cpu().detach().numpy()[:,1],kind="kde")
    plot.savefig("Plots/Gauss_mix_density_dim_2.pdf",format="pdf")
    plt.close()
    plot=plt.hist(samples.cpu().detach().numpy(),bins=30)
    plt.savefig("Plots/Gauss_mix_mean_hist_1_dim_1.pdf",format="pdf")
    plt.close()
    plot=sns.displot(samples.cpu().detach().numpy(), kind="kde")
    plot.savefig("Plots/Gauss_mix_mean_dens_1_dim_1.pdf",format="pdf")

# Save the model
widthstring='_'.join([str(w) for w in widths])
filenamemodel="saved_models/Gauss_mix_"+str(n_mix)+"-comp_"+str(dim)+"-dim_"+widthstring+"widths_"+date.today().strftime('%d-%m-%Y')+".pth"
filenamecovandmean="saved_models/covandmean_Gauss_mix_"+str(n_mix)+"-comp_"+str(dim)+"-dim_"+date.today().strftime('%d-%m-%Y'+".npz")
torch.save([model.kwargs,model.state_dict()], filenamemodel)
np.savez(filenamecovandmean,cov=cov_list,mean=mean_list,mixcom=n_mix,loss_model=approx_loss,loss_exact=ex_loss)


