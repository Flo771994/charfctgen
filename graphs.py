# Used to create graphs for paper (run on cpu)

from datetime import datetime, date
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg") #otherwise plots not shown with error --- has to be removed for computations on server
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import torch
import Lossfct
import model_generator
import scipy
#from collections.abc import Mapping

np.random.seed(771994)
torch.manual_seed(771994)


plotsize=1000000
now=datetime.now()
k=2




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

    ##### Load model and paramters of the model
    filenamemodel="saved_models/Gauss_dist_2x"+str(dim)+"-dim_300_50-widths_08-01-2024.pth" 
    kwargs, state_dict=torch.load(filenamemodel)
    model = model_generator.MLP(**kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    dim=kwargs["output_size"]

    filenamespdmatrix="saved_models/covandmean_Gauss_dist_2x_"+str(dim)+"-dim_08-01-2024.npz"
    covandmean= np.load(filenamespdmatrix)
    

    with torch.no_grad():
        
        ##### write mean and scale matrix to file
        m=covandmean["mean"]
        c=covandmean["cov"]
        #dgf=covandmean["dgf"] #only for t-distribution
        mlatex=format_matrix(m,formatter=lambda x: f'{x:.2f}')
        clatex=format_matrix(c,formatter=lambda x: f'{x:.2f}')
        with open("Latex/double_"+str(dim)+"-dim_gauss"+".txt","w+") as fh:
            fh.writelines(mlatex+ "\n")
            fh.writelines(clatex)

        ######create bivariate contour plots
            
        # sample from generator
        U=np.matrix(np.random.normal(size=k*dim * plotsize)).reshape(plotsize, k*dim)
        U=torch.tensor(U,dtype=torch.float32).to("cpu")
        samples=model(U)
        npsamples=samples.detach().numpy()

        #  sample from exact simulation algorithm
        truesamples=np.random.multivariate_normal(mean=covandmean["mean"],cov=covandmean["cov"],size=plotsize)#
        

        #  estimate densities
        margin=dim-1
        #kdestimation
        kde_true=scipy.stats.gaussian_kde(truesamples[:,[margin-1,margin]].T,bw_method="scott")
        #use same estimator as seaborn displot https://stackoverflow.com/questions/64560379/how-to-use-the-same-kde-used-by-seaborn
        kde_sim=scipy.stats.gaussian_kde(npsamples[:,[margin-1,margin]].T,bw_method="scott")
        
        # set boundaries of x/y axis
        match dim:
            case 2:
                lowerx=-5
                upperx=8
                lowery=0
                uppery=6.5
            case 5:
                lowerx=-11
                upperx=0
                lowery=5
                uppery=15
            case 10:
                lowerx=-13
                upperx=4
                lowery=-10
                uppery=3.5


        X,Y=np.meshgrid(np.linspace(lowerx, upperx, 100),np.linspace(lowery, uppery, 100))
        positions = np.vstack([X.ravel(), Y.ravel()])

        fig,ax=plt.subplots()
        
        #plot=sns.jointplot(x=samples.cpu().detach().numpy()[:,0],y=samples.cpu().detach().numpy()[:,1],kind="kde") 

        Z=np.reshape(kde_true(positions).T, X.shape)
        p1=ax.contour(X,Y,Z,levels=[0.0005,0.001,0.0025,0.005,0.01,0.05,0.1],linestyles="dashdot",colors="b",linewidths=0.5) #Plot samples from exact sim alg
        Z=np.reshape(kde_sim(positions).T, X.shape)
        p2=ax.contour(X,Y,Z,levels=[0.0005,0.001,0.0025,0.005,0.01,0.05,0.1],linestyles="dashed",colors="g",linewidths=0.5) #Plot samples from generator
        Z=np.reshape(scipy.stats.multivariate_normal(mean=covandmean["mean"][[margin-1,margin]],cov=covandmean["cov"][:,[margin-1,margin]][[margin-1,margin],:]).pdf(np.stack([X, Y],axis=-1)),X.shape)
        p3=ax.contour(X,Y,Z,levels=[0.0005,0.001,0.0025,0.005,0.01,0.05,0.1],linestyles="solid",colors="r",linewidths=0.5)  #Plot true contour levels
        plt.clabel(p2,inline=1, fontsize=10,inline_spacing=1)

        h1,_ = p1.legend_elements()
        h2,_ = p2.legend_elements()
        h3,_ = p3.legend_elements()
        ax.legend([h1[0], h2[0],h3[0]], ["Exact", "Generator","True"])
        ax.set_title(str(dim)+"-dim")
        plt.xlim([lowerx,upperx])
        plt.ylim([lowery,uppery])
        plt.savefig("Plots/2x_"+str(dim)+"-dim_gauss-"+"contourplot.pdf",format="pdf")
        plt.close()




        ######   create marginal density plots
        

        # set boundaries of x-axis
        match dim:
            case 2:
                lowerx=-5
                upperx=10
            case 5:
                lowerx=-10
                upperx=15
            case 10:
                lowerx=-15
                upperx=15


        plot=sns.displot(npsamples, kind="kde",legend=False,linewidth=1,common_norm=False,linestyle="dashed")
        plt.xlim([lowerx,upperx])
        xlim=plt.xlim()
        plt.title(str(dim)+"-dim")
        plot.fig.subplots_adjust(top=0.9)
        legend=[mlines.Line2D([], [], color='black', label='Generator',linestyle="dashed"),mlines.Line2D([], [], color='black', label='True',linestyle="solid")]
        plt.legend(title="Component", labels=[str(dim-x) for x in np.arange(dim)])#plt.legend(handles=legend)
        for i in np.arange(m.size):
            x=np.arange(xlim[0],xlim[1],0.01)
            y=scipy.stats.norm.pdf(x,loc=m[i],scale=np.sqrt(covandmean["cov"][i,i]))# compare with true density
            plt.plot(x,y,linestyle="solid",linewidth=0.5,alpha=0.7)
        plt.savefig("Plots/2x_"+str(dim)+"-dim_gauss_"+"density_marginal.pdf",format="pdf")



print("time for execution:",datetime.now()-now )

