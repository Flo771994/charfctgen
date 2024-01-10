from datetime import datetime, date
from sklearn.datasets import make_spd_matrix as spd_matrix
import numpy as np
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
from collections.abc import Mapping  # sometimes needed to prevent error on Windows


# set the device for computation on cpu or gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Cuda available?", torch.cuda.is_available(), "device: ",device)



#set seed for reproducibility
np.random.seed(771994)
torch.manual_seed(771994)

#Set default datatype to make conversion from numpy easier
torch.set_default_dtype(torch.float32)



############## Set hyperparameters
widths=[300,50] #Widhts of the hidden layers
iterations=500 # number of epochs
learning_rate_Adam=0.01
batchsize=600 # n= sample size from generator used to estimate loss function
s_size_approx=600 # m= sample size of the W vectors to approximate kernel evaluations
testsize=1500  #number of random variables used to estimate loss on test set
plotsize=5000  #number of random variables used to create density plots

print("batchsize:",batchsize, " testsize:",testsize )



# Define the characteristic function that is supposed to be simulated
dim=10
c=spd_matrix(dim,random_state=771994)
m=np.random.uniform(-10,10,dim)
mean=torch.tensor(m,dtype=torch.float32).to(device)
mean_cpu=torch.tensor(m,dtype=torch.float32).to("cpu")
cov_cpu=torch.tensor(c,dtype=torch.float32).to("cpu")
cov=torch.tensor(c,dtype=torch.float32).to(device)
dgf=3
alpha=2
beta=1


print("mean:",m," cov:",c,"dgf:",dgf, "alpha:",alpha,"beta:",beta)


#################################### Define the characteristic function to be learned by the generator              #########################
#################################### It must be a function that takes vectors in the same dimension as W as input   #########################
#################################### The function must be defined using torch.tensors's                             #########################
#################################### If you want to simulate your own custom charactersitic function simply replace #########################
#################################### the respective characteristic functions below                                  #########################
#################################### The code should then produce a generator from this characteristic function     #########################
def charfct_vec(W):
    return Lossfct.char_fct_rational_quad_vec(W,alpha,beta,mean,device)#Lossfct.gaussian_char_fct_vec(W,mean,cov,device)
def charfct_vec_cpu(W):
    return Lossfct.char_fct_rational_quad_vec(W,alpha,beta,mean_cpu,"cpu")#Lossfct.gaussian_char_fct_vec(W,mean_cpu,cov_cpu,"cpu") #needed to conduct evaluation on cpu
############################################################################################################################################





# Define the kernelmatrix corresponding to the kernel evaluations and the samples to approximate the kernel evaluations
bandwidth=[0.02,0.5,1,5,50] #must be a list
### If exact mathematical expression of kernel should be used when possible
def kernelmatrix(X):
    return Lossfct.gaussian_kernelmatrix(X,device,bandwidth) #needs to be compatible with the distribution that is sampled in !HERE!
def kernelmatrix_cpu(X):
    return Lossfct.gaussian_kernelmatrix(X,"cpu",bandwidth) #needed to conduct evaluation on cpu

### Distribution of W to approximate the kernel
def sample_kernel(size):
    return Lossfct.kernelsample_gaussian(size,device,dim,bandwidth) # !HERE!
def sample_kernel_cpu(size):
    return Lossfct.kernelsample_gaussian(size,"cpu",dim,bandwidth) #needed to conduct evaluation on cpu


########## Define the model
hidden_architecture=[]
for w in widths:
    hidden_architecture.append( (w,torch.nn.ReLU()) )
model=model_generator.MLP(dim,dim,hidden_architecture).to(device)


########## Set optimizer
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate_Adam )
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[2000,4000],gamma=0.1)

# #Summary writer for Tensorboard initialization
# summary(model,input_data=torch.tensor(np.zeros(shape=(1,dim)),dtype=torch.float32),device=device)
# writer = SummaryWriter()

################# Training loop ##########################
now=datetime.now()
for epoch in np.arange(iterations):

    # Sample from the generator
    U=np.matrix(np.random.normal(size=dim * batchsize)).reshape(batchsize, dim)
    U=torch.tensor(U,dtype=torch.float32).to(device)

    # Sample to approximation the kernel (every 20th iteration)
    if epoch%20==0:
        W = sample_kernel(s_size_approx)
    
    # Make predictions
    samples=model(U)

    # Empty gradient
    optimizer.zero_grad()
    
    # Calculate Loss
    loss = Lossfct.lossfunction_vec(samples, kernelmatrix, charfct_vec, W, device, False) #True means using exact representation of kernel where possible, if false kernelmatrix is irrelevant
    
    # Calculate gradient
    loss.backward()
    
    # Take one SGD step
    optimizer.step()
    scheduler.step() 
    
    # #logging for tensorboard
    # writer.add_scalar('Loss', loss.item(), epoch)

    if epoch%25==0:
        print("epoch: ",epoch," loss=",loss)
        
################# end of training loop ##########################
print("Training time: ",datetime.now()-now)


# Set model to evaluation mode
model.eval()

with torch.no_grad():
    
    # #visualize the model in tensorboard
    # writer.add_graph(model, torch.tensor(np.random.normal(size=dim),dtype=torch.float32).to(device))
    # #start tensorboard with tensorboard --logdir=runs

    #Send model to cpu for calculation on larger matrices
    model.cpu()
    
    # Estimatef true loss by using a large sample size and including the term in the MMD estimate that does not depend on the generator
    U=np.matrix(np.random.normal(size=dim * testsize)).reshape(testsize, dim)
    U=torch.tensor(U,dtype=torch.float32).to("cpu")
    samples=model(U)
    W=sample_kernel_cpu(testsize).to("cpu") #Used to approximate d-dimensional kernel
    
    approx_loss= Lossfct.lossfunction_vec(samples,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

    print("The approximate loss of the final model is equal to ",approx_loss,". It should be very close to 0 (<0.01). Otherwise the model has not learned the distribution correctly")

    # Visualize the distribution of the generator by plotting marginal/bivariate densities
    U=np.matrix(np.random.normal(size=dim * plotsize)).reshape(plotsize, dim)
    U=torch.tensor(U,dtype=torch.float32).to("cpu")
    samples=model(U)
    

    plot=sns.jointplot(x=samples.cpu().detach().numpy()[:,0],y=samples.cpu().detach().numpy()[:,1],kind="kde")
    plot.savefig("Plots/bivariate_contour_plot.pdf",format="pdf")
    plt.close()
    plot=plt.hist(samples.cpu().detach().numpy(),bins=30)
    plt.savefig("Plots/marginal_histogram.pdf",format="pdf")
    plt.close()
    plot=sns.displot(samples.cpu().detach().numpy(), kind="kde")
    plot.savefig("Plots/marginal_densities.pdf",format="pdf")

# Save the model
widthstring='_'.join([str(w) for w in widths])
filenamemodel="saved_models/model_"+str(dim)+"-dim_"+widthstring+"widths_"+date.today().strftime('%d-%m-%Y')+".pth"
filenamecovandmean="saved_models/covandmean_"+str(dim)+"-dim_"+date.today().strftime('%d-%m-%Y'+".npz")
torch.save([model.kwargs,model.state_dict()], filenamemodel)
np.savez(filenamecovandmean,cov=c,mean=m,dgf=dgf)

