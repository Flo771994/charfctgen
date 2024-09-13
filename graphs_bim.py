# Used to create graphs for paper (run on cpu)

from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib
import tabulate
matplotlib.use("Qt5Agg") #otherwise plots not shown with error --- has to be removed for computations on server
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import torch
from torchinfo import summary
import Lossfct
import model_generator
import scipy
#from collections.abc import Mapping

np.random.seed(771994)
torch.manual_seed(771994)


size_sample_W=20000
plotsize=1000000
now=datetime.now()
bandwidth=[0.02,0.5,1,5,100]
k=2
device="cpu"

def kernelmatrix_cpu(X):
    return Lossfct.gaussian_kernelmatrix(X,"cpu",bandwidth) #needed to conduct evaluation on cpu

def format_matrix(matrix, environment="pmatrix", formatter=str):
    """Format a matrix using LaTeX syntax"""

    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception:
            raise TypeError("Could not convert to Numpy array")

    if len(shape := matrix.shape) == 1:
        matrix = matrix.reshape(1, shape[0])
    elif len(shape) > 2:
        raise ValueError("Array must be 2 dimensional")

    body_lines = [" & ".join(map(formatter, row)) for row in matrix]

    body = "\\\\\n".join(body_lines)
    return f"""\\begin{{{environment}}}{body}\\end{{{environment}}}"""





for dim in ["2","5","10"]:
    for n in ["2","5","10"]:

        # Load model
        filenamemodel="saved_models/Gauss_mix_"+str(n)+"-comp_"+str(dim)+"-dim_300_50widths_31-07-2024.pth"
        kwargs, state_dict=torch.load(filenamemodel)
        model = model_generator.MLP(**kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        
        dim=kwargs["output_size"]

        filenamespdmatrix="saved_models/covandmean_Gauss_mix_"+str(n)+"-comp_"+str(dim)+"-dim_31-07-2024.npz"
        covandmean= np.load(filenamespdmatrix)
        n_mix=covandmean["mixcom"]
        mean_list=covandmean["mean"]
        cov_list=covandmean["cov"]
        mean_cpu=torch.tensor(covandmean["mean"],dtype=torch.float32).to("cpu")
        cov_cpu=torch.tensor(covandmean["cov"],dtype=torch.float32).to("cpu")

        def charfct_vec_cpu(W):
            l=Lossfct.gaussian_char_fct_vec(W,mean_cpu[0],cov_cpu[0],"cpu")/n_mix
            for j in np.arange(n_mix-1):
                l= l+Lossfct.gaussian_char_fct_vec(W,mean_cpu[j+1],cov_cpu[j+1],"cpu")/n_mix      
            return l
        
        def sample_kernel_cpu(size):
            return Lossfct.kernelsample_gaussian(size,"cpu",dim,bandwidth) #needed to conduct evaluation on cpu

        

        with torch.no_grad():
            


            ######create bivariate contour plots
            U=np.matrix(np.random.normal(size=k*dim * plotsize)).reshape(plotsize, k*dim)
            U=torch.tensor(U,dtype=torch.float32).to("cpu")
            samples=model(U)
            npsamples=samples.detach().numpy()

            ########## benchmark the generated samples with an exact sample
            category=np.random.choice(n_mix,size=plotsize)
            true_sample=np.zeros((plotsize,dim))
            for j in np.arange(n_mix):
                indicator=np.equal(category,j)
                indicator=np.tile(indicator.reshape((plotsize,1)),(1,dim))
                true_sample=true_sample+np.multiply(indicator,np.random.multivariate_normal(mean=mean_list[j],cov=cov_list[j],size=plotsize))

            margin=dim-1
            #kdestimation
            kde_exact=scipy.stats.gaussian_kde(true_sample[:,[margin-1,margin]].T,bw_method="scott")
            #use same estimator as seaborn displot https://stackoverflow.com/questions/64560379/how-to-use-the-same-kde-used-by-seaborn
            kde_sim=scipy.stats.gaussian_kde(npsamples[:,[margin-1,margin]].T,bw_method="scott")
                    
            lowerx=0.
            upperx=0.
            lowery=0.
            uppery=0.
            for i in np.arange(n_mix):
                lowerx=min(lowerx,mean_list[i][margin-1]-5)
                upperx=max(upperx,mean_list[i][margin-1]+5)
                lowery=min(lowery,mean_list[i][margin]-5)
                uppery=max(uppery,mean_list[i][margin]+5)
        

            X,Y=np.meshgrid(np.linspace(lowerx, upperx, 100),np.linspace(lowery, uppery, 100))
            positions = np.vstack([X.ravel(), Y.ravel()])

            fig,ax=plt.subplots()
            
            def true_dens(X,Y):
                dens=scipy.stats.multivariate_normal(mean=mean_list[0][[margin-1,margin]],cov=cov_list[0][:,[margin-1,margin]][[margin-1,margin],:]).pdf(np.stack([X, Y],axis=-1))/n_mix
                for j in np.arange(n_mix-1):
                   dens=dens+scipy.stats.multivariate_normal(mean=mean_list[j+1][[margin-1,margin]],cov=cov_list[j+1][:,[margin-1,margin]][[margin-1,margin],:]).pdf(np.stack([X, Y],axis=-1))/n_mix
                return dens

            
            Z=np.reshape(kde_exact(positions).T, X.shape)
            p1=ax.contour(X,Y,Z,levels=[0.0005,0.001,0.0025,0.005,0.01,0.05,0.1],linestyles="dashdot",colors="b",linewidths=0.5) #Plot samples from exact sim alg
            Z=np.reshape(kde_sim(positions).T, X.shape)
            p2=ax.contour(X,Y,Z,levels=[0.0005,0.001,0.0025,0.005,0.01,0.05,0.1],linestyles="dashed",colors="g",linewidths=0.5) #Plot samples from generator
            Z= np.reshape(true_dens(X,Y),X.shape)
            p3=ax.contour(X,Y,Z,levels=[0.0005,0.001,0.0025,0.005,0.01,0.05,0.1],linestyles="solid",colors="r",linewidths=0.5)  #Plot true contour levels
            plt.clabel(p3,inline=1, fontsize=8,inline_spacing=1)

            h1,_ = p1.legend_elements()
            h2,_ = p2.legend_elements()
            h3,_ = p3.legend_elements()
            ax.legend([h1[0], h2[0],h3[0]], ["Exact", "Generator","True"])
            ax.set_title(str(dim)+"-dim")
            plt.xlim([lowerx,upperx])
            plt.ylim([lowery,uppery])
            plt.savefig("Plots/"+str(dim)+"-dim_Gauss_mix_"+str(n_mix)+"comp_contourplot.pdf",format="pdf")
            plt.close()

        
            ####Create marginal density plots
            # Define a color palette
            palette = sns.color_palette("husl", mean_list[0].size)    
            #sns.palettes(palette)
            plot=sns.displot(npsamples, kind="kde",legend=False,linewidth=0.7,alpha=0.8,common_norm=False,linestyle="dashed",palette=palette)
            
            lowerx,upperx= plt.gca().get_xlim()


            plt.xlim([lowerx,upperx])
            xlim=plt.xlim()
            plt.title(str(dim)+"-dim")
            plot.fig.subplots_adjust(top=0.9)
            #legend=[mlines.Line2D([], [], color='black', label='Generator',linestyle="dashed"),mlines.Line2D([], [], color='black', label='True',linestyle="solid"),mlines.Line2D([], [], color='black', label='Exact',linestyle="dotted")]
            plt.legend(title="Margin", labels=[str(dim-x) for x in np.arange(dim)])#plt.legend(handles=legend)
            for i in np.arange(mean_list[0].size):
                color = palette[i]
                x=np.arange(xlim[0],xlim[1],0.01)
                y=scipy.stats.norm.pdf(x,loc=mean_list[0][i],scale=np.sqrt(cov_list[0][i,i]))/n_mix
                for j in np.arange(n_mix-1):
                    y=y+scipy.stats.norm.pdf(x,loc=mean_list[j+1][i],scale=np.sqrt(cov_list[j+1][i,i]))/n_mix
                plt.plot(x,y,linestyle="solid",linewidth=0.8,alpha=0.8,color=color)
                kde_exact=scipy.stats.gaussian_kde(true_sample[:,i],bw_method="scott")
                y=kde_exact(x)
                plt.plot(x,y,linestyle="dotted",linewidth=0.7,alpha=0.8,color=color)
            plt.savefig("Plots/"+str(dim)+"-dim_Gauss_mix_"+str(n_mix)+"mix-comp_density_marginal.pdf",format="pdf")


            ########## benchmark the estimated loss of the final model with the estimated loss of an exact sample
            W=sample_kernel_cpu(size_sample_W).to("cpu") #Used to approximate d-dimensional kernel
            samples_short=samples[0:size_sample_W,]
            category=np.random.choice(n_mix,size=size_sample_W)
            true_sample=np.zeros((size_sample_W,dim))
            for j in np.arange(n_mix):
                indicator=np.equal(category,j)
                indicator=np.tile(indicator.reshape(size_sample_W,1),(1,dim))
                true_sample=true_sample+np.multiply(indicator,np.random.multivariate_normal(mean=mean_list[j],cov=cov_list[j],size=size_sample_W))

            true_sample_torch=torch.tensor(true_sample,dtype=torch.float32).to("cpu")
            ex_loss= Lossfct.lossfunction_vec(true_sample_torch,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

            print("The approximate loss of the exact sample is equal to ",ex_loss,". It should be close to 0. Otherwise the model has not learned the distribution correctly")
        
            dist_to_true_sam=0
            for b in bandwidth:
                dist_to_true_sam=Lossfct.MMD_equal_case(samples_short,device,b)- Lossfct.MMD_mixed_case(samples_short,true_sample_torch,device,b) +Lossfct.MMD_equal_case(true_sample_torch,device,b)
            dist_to_true_sam=dist_to_true_sam/len(bandwidth)
            
            print("The estimated MMD distance of a sample from the model to an exact sample is ",dist_to_true_sam,". It should be close to 0. Otherwise the model has not learned the distribution correctly")


            ########### Calculate estimate of the loss by using a large sample size and including the term in the MMD estimate that does not depend on the generator
            
            
            approx_loss= Lossfct.lossfunction_vec(samples_short,kernelmatrix_cpu,charfct_vec_cpu,W,"cpu")+Lossfct.estimate_constant_vec(W,charfct_vec_cpu)

            print("The approximate loss of the final model is equal to ",approx_loss,". It should be close to 0. Otherwise the model has not learned the distribution correctly")

            # Save losses
            table_rows=[]
            table_rows.append([f"{approx_loss}"]+[f"{ex_loss}"]+[f"{dist_to_true_sam}"])
 
            # Define the table headers dynamically based on the number of quantiles
            headers=["loss_gen", "loss_exact","mmd_comp"] 
            # Create the LaTeX table using tabulate
            latex_table = tabulate.tabulate(table_rows, headers, tablefmt="latex")

            # Print or save the LaTeX table
            #print(latex_table)
            with open(f"Latex/table_gauss_mix_{dim}_dim-{n_mix}_mix_comp.txt", "w") as f:
                f.write(latex_table)
    




print("time for execution:",datetime.now()-now )

