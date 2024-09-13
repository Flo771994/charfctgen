# Used to create graphs for paper (run on cpu)

from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg") #otherwise plots not shown with error --- has to be removed for computations on server
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import torch
from torchinfo import summary
import Lossfct
import model_generator
import scipy
from scipy.special import gamma
import time
import tabulate
#from collections.abc import Mapping

np.random.seed(771994)
torch.manual_seed(771994)


plotsize=1000000
size_sample_W=20000
s_size_int=6000
noww=time.time()
bandwidth=[0.02,0.5,1,5,100]
k=2
device="cpu"

def kernelmatrix_cpu(X):
    return Lossfct.gaussian_kernelmatrix(X,"cpu",bandwidth) #needed to conduct evaluation on cpu





def paramsproj(b,alpha,cov,mean,shift,n_approx):
    """ Return the parameters of the projection of the multivariate alpha stable distribution to the vector b """
    gauss_sample=np.random.multivariate_normal(mean=mean,cov=cov,size=n_approx)
    norm=np.sum(gauss_sample**2,axis=1)**(-1/2)
    gauss_sample_norm=np.multiply(gauss_sample,norm.reshape(-1,1))
    
    s=np.mean(np.power( np.abs( np.sum(np.multiply(gauss_sample_norm, b) ,axis=1) ) ,alpha) )
    sigma=s**(1/alpha)
    beta=np.mean( np.multiply( np.power( np.abs( np.sum(np.multiply(gauss_sample_norm, b) ,axis=1) ) ,alpha) ,  np.sign(np.sum(np.multiply(gauss_sample_norm, b) ,axis=1) ) ) )/s

    if alpha==1:
        mu=np.sum(np.multiply(b,shift))-2*np.mean( np.multiply(   np.sum(np.multiply(gauss_sample_norm, b) ,axis=1)   ,  np.log(np.abs( np.sum(np.multiply(gauss_sample_norm, b) ,axis=1) ) ) ) )/np.pi
    else:
        mu=np.sum(np.multiply(b,shift))
    return [sigma,beta,mu]




for dim in ["2","5","10"]:
    for alpha in ["0.5","1"]:

        # Load model
        filenamemodel="saved_models/alpha_stab_"+str(alpha)+"_"+str(dim)+"-dim_300_50widths_19-08-2024.pth"
        kwargs, state_dict=torch.load(filenamemodel)
        model = model_generator.MLP(**kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        dim=kwargs["output_size"]

        filenamespdmatrix="saved_models/params_alpha_stab_"+str(alpha)+"_"+str(dim)+"-dim_19-08-2024.npz"#"saved_models/params_alpha_stab_"+str(alpha)+"_"+str(dim)+"-dim_22-07-2024.npz"
        covandmean= np.load(filenamespdmatrix)
        mean=covandmean["mean"]
        cov=covandmean["cov"]
        alpha=float(covandmean["alpha"])
        shift=covandmean["shift"]
        shift_cpu=torch.tensor(shift,device="cpu",dtype=torch.float32)
        mean_cpu=torch.tensor(covandmean["mean"],dtype=torch.float32).to("cpu")
        cov_cpu=torch.tensor(covandmean["cov"],dtype=torch.float32).to("cpu")
        eps=0.1
        num_b=3
        q=[0.05,0.1,0.3,0.5,0.7,0.9,0.95]
        n_approx=plotsize


                    
        def sample_kernel_cpu(size):
            return Lossfct.kernelsample_gaussian(size,"cpu",dim,bandwidth) #needed to conduct evaluation on cpu
        


        with torch.no_grad():
            
          
            now=time.time()
            U=np.matrix(np.random.normal(size=k*dim * plotsize)).reshape(plotsize, k*dim)
            U=torch.tensor(U,dtype=torch.float32).to("cpu")
            samples=model(U)
            npsamples=samples.detach().numpy()
            training_time=time.time()-now
            print("average time per sample generator", training_time/plotsize)


             # Define the sample to approximate the compensator in the prm simulation
            gauss_sample=np.random.multivariate_normal(mean=mean,cov=cov,size=s_size_int)
            norm=np.sum(gauss_sample**2,axis=1)**(-1/2)
            gauss_sample_norm=np.multiply(gauss_sample,norm.reshape(-1,1))
            gauss_sample_norm_torch=torch.tensor(gauss_sample_norm,device=device,dtype=torch.float32)


            def charfct_vec_cpu(W):
                return Lossfct.char_fct_alpha_stab(W,alpha,shift_cpu,gauss_sample_norm_torch,"cpu")

            c_alpha=np.sqrt(np.pi)*(2**(-alpha))*(alpha**(-1))*gamma( (2-alpha)/2 )*((gamma( (1+alpha)/2 ))**(-1)) #constant needed to normalize the PRm rep
            factor=alpha*c_alpha
            
            # sampling algorithm
            now=time.time()
            sample_prm=np.zeros((plotsize,dim))
            counter=0
            # need compensator for the correct approximation via the Levy Ito decomposition
            compensator=np.zeros(dim)
            if alpha>1:
                compensator= -eps**(1-alpha)/(c_alpha*(1-alpha))*np.sum(gauss_sample_norm,axis=0)/plotsize
            if alpha==1:
                compensator= -np.log(eps)/c_alpha*np.sum(gauss_sample_norm,axis=0)/plotsize
            for i in np.arange(plotsize):
                sum=0
                count=0
                delta=np.random.exponential(size=1)
                while (factor*delta)**(-1/alpha)>eps:
                    g_sample_prm=np.random.multivariate_normal(mean=mean,cov=cov,size=1)
                    n_prm=np.sum(g_sample_prm**2)**(-1/2)
                    g_sample_n_prm= g_sample_prm*n_prm
                    obs=g_sample_n_prm*(factor*delta)**(-1/alpha)
                    sum=sum+obs
                    delta=delta+np.random.exponential(size=1)
                    counter=counter+1
                sample_prm[i,]=sum+shift-compensator
            training_time=time.time()-now

            
            table_rows=[]
            
            b=[np.ones(dim)]
            for i in np.arange(num_b-1):
                if i==0:
                    b.append([1,-1,1,-1,1,-1,1,-1,1,-1][0:dim])
                if i==1:
                    b.append([-1,2,-1,2,-1,2,-1,2,-1,2][0:dim])
            for j in np.arange(len(b)):
                sigma,beta,mu=paramsproj(b[j],alpha,cov,mean,shift,n_approx)
                quantiles_stab=scipy.stats.levy_stable.ppf(q, alpha, beta, mu, sigma)
                proj_gen=np.dot(npsamples,b[j])
                proj_prm=np.dot(sample_prm,b[j])
                quantiles_gen=np.quantile(proj_gen,q)  
                quantiles_prm=np.quantile(proj_prm,q) 
                # Append a row with the quantiles for the current b[j]
                    # Format each quantile value to two decimal places and convert to lists
                quantiles_stab = [f"{x:.2f}" for x in quantiles_stab]
                quantiles_gen = [f"{x:.2f}" for x in quantiles_gen]
                quantiles_prm = [f"{x:.2f}" for x in quantiles_prm]
 
                # Add rows for each set of quantiles with labels
                table_rows.append([" ", "Generator"] + quantiles_gen)
                table_rows.append([f"b_\u007b{j+1}\u007d", "True"] + quantiles_stab)
                table_rows.append([" ", "Approximate"] + quantiles_prm)
                table_rows.append("\hline")

            # Define the number of quantiles dynamically
            num_quantiles = len(q)

            # Define the table headers dynamically based on the number of quantiles
            headers=[" ", "Quantile"] + [f"{i}" for i in q]
            # Create the LaTeX table using tabulate
            latex_table = tabulate.tabulate(table_rows, headers, tablefmt="latex")

            # Print or save the LaTeX table
            #print(latex_table)
            with open(f"Latex/table_{alpha}_stab_{dim}_dim_quantiles.txt", "w") as f:
                f.write(latex_table)
    
            print("Training time PRM: ",training_time)
            print("average time per sample", training_time/plotsize)
            print("average number of atoms per sample", counter/plotsize)
            print("expected number of atoms per sample", (1/eps)**alpha/factor)

            ########## benchmark the estimated loss of the final model with the estimated loss of an exact sample
            W=sample_kernel_cpu(size_sample_W).to("cpu") #Used to approximate d-dimensional kernel

            samples_short=samples[0:size_sample_W,]
            approx_loss= Lossfct.lossfunction_vec(samples_short,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

            print("The approximate loss of the final model is equal to ",approx_loss,". It should be close to 0. Otherwise the model has not learned the distribution correctly")
            
            dist_to_approx_sam=0
            sample_prm_torch_short=torch.tensor(sample_prm[0:size_sample_W,],device="cpu",dtype=torch.float32)
            for b in bandwidth:
                dist_to_approx_sam=Lossfct.MMD_equal_case(samples_short,device,b)- Lossfct.MMD_mixed_case(samples_short,sample_prm_torch_short,device,b) +Lossfct.MMD_equal_case(sample_prm_torch_short,device,b)
            dist_to_approx_sam=dist_to_approx_sam/len(bandwidth)
            
            print("The estimated MMD distance of a sample from the model to the PRM approximation sample is ",dist_to_approx_sam,". It should be close to 0. Otherwise the model has not learned the distribution correctly")

            prm_loss= Lossfct.lossfunction_vec(sample_prm_torch_short,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

            print("The approximate loss of the PRM approximation is equal to ",prm_loss,". It should be close to 0. Otherwise the model has not learned the distribution correctly")
        

            # Save losses
            table_rows=[]
            table_rows.append([f"{approx_loss}"]+[f"{prm_loss}"]+[f"{dist_to_approx_sam}"])
 
            # Define the table headers dynamically based on the number of quantiles
            headers=["loss_gen", "prm_loss" ,"mmd_comp"] 
            tabulate.LATEX_ESCAPE_RULES = {}
            # Create the LaTeX table using tabulate
            latex_table = tabulate.tabulate(table_rows, headers, tablefmt="latex")

            # Print or save the LaTeX table
            #print(latex_table)
            with open(f"Latex/table_{alpha}_stab_{dim}_dim_losses.txt", "w") as f:
                f.write(latex_table)

        
    
            

print("time for execution:",time.time()-noww )

