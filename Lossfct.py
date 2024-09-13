import cmath
import numpy as np
import torch
import scipy
from datetime import datetime

def gaussian_kernel(x,y,bandwidth=1):
    """Gaussian kernel
    Input must be tensors of shape [1,dimension]
    """
    exponent=( torch.sum((x-y)**2) )/bandwidth
    return torch.exp(-exponent)

def gaussian_kernelmatrix(x,device,bandwidth: list=[1]):
    """Gaussian kernelmatrix
    x is of shape [n,d]
    bandwidth must be a list
    """
    sum=torch.zeros(size=(x.shape[0],x.shape[0])).to(device)
    for b in bandwidth:
        exponent = torch.cdist(x, x, p=2).to(device)**2 / b
        sum=sum+torch.triu(torch.exp(-exponent), diagonal=1)
    return sum/len(bandwidth)

def laplace_kernel(x,y,bandwidth=1):
    """Laplace kernel
    Input must be tensors of shape [1,dimension]
    """
    exponent=torch.sum(torch.abs(x-y))/bandwidth
    return torch.exp(-exponent)

def laplace_kernelmatrix(x,device, bandwidth: list=[1]):
    """Laplace kernelmatrix
    x is of shape [n,d]
    bandwidth must be a list
    """
    sum=torch.zeros(size=(x.shape[0],x.shape[0])).to(device)
    for b in bandwidth:
        exponent = torch.cdist(x, x, p=1).to(device) / b
        sum=sum+torch.triu(torch.exp(-exponent), diagonal=1)
    return sum/len(bandwidth)

def kernelsample_gaussian(size,device,dim=1,bandwidth: list=[1]):
    """ Sample from the random vectors behind the Gaussian kernel with various badnwidths
    If len(bandwidth=1 then it is just a sample for the Gaussian kernel (Gaussian distribution) with bandwidth=bandwidth[0]
    If len(bandwidth>1 then it a sample for the average of (len(bandwidth) Gaussian kernels with bandwidths given by bandwidth
    """
    r_bandwidth=1/np.sqrt(np.random.choice(bandwidth,size=size))
    b=np.sqrt(2)*torch.tensor(r_bandwidth,dtype=torch.float32).reshape((size,1)).to(device)
    return torch.mul(b,torch.tensor(np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=size), dtype=torch.float32).to(device))


def kernelsample_laplace(size,device,dim=1,bandwidth: list=[1]):
    """ Sample from the random vectors behind the Laplace kernel with various badnwidths
        If len(bandwidth=1 then it is just a sample for the Laplace kernel (Cauchy distribution) with bandwidth=bandwidth[0]
        If len(bandwidth>1 then it a sample for the average of (len(bandwidth) Laplace kernels with bandwidths given by bandwidth
        """
    r_bandwidth=1/np.random.choice(bandwidth,size=size)
    b=torch.tensor(r_bandwidth,dtype=torch.float32).reshape((size,1)).to(device)
    return torch.mul(b,torch.tensor(np.random.standard_cauchy(size*dim).reshape(size,dim), dtype=torch.float32).to(device))



def gaussian_char_fct(z,mu,sigma):
    """ z is a tensor of shape (1,d) and corresponds arguments of characteristic function of d-dimensional Gaussian distribution
        mu is a tensor of shape (1,d) and corresponds to mean vector
        sigma is a tensor of shape (d,d)
    """
    # z=np.matrix(z)
    # mu=np.matrix(mu)
    # sigma=np.matrix(sigma)
    if not ((z.size() == sigma[0,:].size()) & (z.size() == sigma[:,0].size()) ):
        print("Dimensions inconsistent" ,[z.shape, sigma.shape, mu.shape])
        return  cmath.nan

    m=torch.sum(torch.mul(mu,z),dtype=torch.float32)
    m=torch.complex(real=torch.tensor([0],dtype=torch.float32),imag=m)

    v= - torch.matmul( torch.matmul(z,sigma) ,torch.t(z) ) /2
    v=torch.complex(real=v ,imag=torch.tensor([0],dtype=torch.float32))

    return(torch.exp(m+v))

def gaussian_char_fct_vec(z,mu,sigma,device):
    """ z is a tensor of shape (n,d) and corresponds arguments of characteristic function of d-dimensional Gaussian distribution
        mu is a tensor of shape (1,d) and corresponds to mean vector
        sigma is a tensor of shape (d,d) and corresponds to the covariance matrix
    """
    m=torch.matmul(z,mu)
    m=torch.complex(real=torch.tensor([0],dtype=torch.float32).to(device),imag=m)
    v = - torch.sum ( torch.mul( torch.matmul(z,sigma),z ) ,dim=1 )/2
    v = torch.complex(real=v, imag=torch.zeros(v.size(), dtype=torch.float32).to(device))
    return(torch.exp(m+v))



def lossfunction(Y,kernel,charfct,W):
    """ This function estimates the part of the MMD estimate that depends on the parameters of the generator
    Y is a tensor of dimension [n,d] of n samples from the model used in the d-dimensional characteristic function charfct
    charfct is the target characteristic function from which we want to simulate. It must return a complex tensor.
    kernel is a functional parameter to evaluate the kernel corresponding to the chosen MMD
    W is a tensor  of dimension [m,d] with an m-sample from the random vector corresponding to the kernel used to approximate the kernel evaluations
    """

    n=Y.shape[0]
    m=W.shape[0]

    sumkernel=torch.tensor([0],dtype=torch.float32)
    sumcharfct=torch.complex(real=torch.tensor([0],dtype=torch.float32), imag=torch.tensor([0],dtype=torch.float32))

    for i in np.arange(n):
        for j in np.arange(i+1,n):
            sumkernel= sumkernel + 2*kernel(Y[i,:],Y[j,:])
    for i in np.arange(n):
        for j in np.arange(m):
            helpsum= torch.sum( torch.mul(W[j,:],Y[i,:]) )
            sumcharfct= sumcharfct+ torch.exp( torch.complex(real=torch.tensor([0],dtype=torch.float32), imag=-helpsum) )*charfct(W[j,:])
    # We only have to consider the real part of charfct
    sumcharfct = -2 * sumcharfct.real / (n * m)
    sumkernel=sumkernel/(n*(n-1))

    return sumkernel+sumcharfct


def lossfunction_vec(Y,kernelmatrix,charfct,W,device,exact=True):
    """ This function estimates the part of the MMD estimate that depends on the parameters of the generator
    Y is a tensor of dimension [n,d] of n samples from the model used in the d-dimensional characteristic function charfct
    charfct is the target characteristic function from which we want to simulate. It must return a complex tensor.
    kernelmatrix returns the kernelmatrix with the evaluations of the kernel corresponding to the chosen MMD
    W is a tensor  of dimension [m,d] with an m-sample from the random vector corresponding to the kernel used to approximate the kernel evaluations
    """

    n=Y.shape[0]
    m=W.shape[0]

    helpsum = torch.mm(Y, torch.t(W))
    exp_imag = torch.exp(-torch.complex(torch.tensor([0.0], dtype=torch.float32).to(device), helpsum))
    sumcharfct = torch.sum(torch.matmul(exp_imag, charfct(W)))
    # We only have to consider the real part of charfct
    sumcharfct = -2 * sumcharfct.real / (n * m)

    if(exact):
        sumkernel = 2 * torch.sum(kernelmatrix(Y))
        sumkernel=sumkernel/(n*(n-1))
    else:
        helpsum = torch.mm(Y, torch.t(W))
        exp_imag = torch.exp( torch.complex(torch.tensor([0.0], dtype=torch.float32).to(device), helpsum))
        exp_imag_1 = torch.exp( torch.complex(torch.tensor([0.0], dtype=torch.float32).to(device), -helpsum))
        sumkernel = torch.sum( torch.matmul(exp_imag,torch.t(exp_imag_1)) ) - n*m  #diag is (m,...,m)
        # We only have to consider the real part of charfct since imaginary part should only be numerical error
        sumkernel = sumkernel.real/(n*(n-1)*m)

    return sumkernel+sumcharfct

def estimate_constant(W,charfct):
    """ This function estimates the term in the MMD estimation that does not depend on the parameters of the generator
        W is a tensor of dimension[m, d] with an m-sample from the random vector corresponding to the kernel used to approximate the kernel evaluations
        charfct is the target characteristic function that is supposed to be simulated"""
    sumcharfct=torch.tensor([0],dtype=torch.complex64)
    m=W.shape[0]
    for i in np.arange(m):
        sumcharfct = sumcharfct + charfct(-W[i,:])*charfct(W[i, :])
    # In theory, sumcharfct is purely real-valued, but due to numerical errors it can happen that sumcharfct has non-zero imaginary value, which is why we drop the imaginary part here
    sumcharfct = sumcharfct.real / m
    return sumcharfct

def estimate_constant_vec(W,charfct):
    """ This function estimates the term in the MMD estimation that does not depend on the parameters of the generator
        W is a tensor of dimension[m, d] with an m-sample from the random vector corresponding to the kernel used to approximate the kernel evaluations
        charfct is the target characteristic function that is supposed to be simulated"""
    m=W.shape[0]
    sumcharfct= torch.sum(charfct(W)*charfct(-W))
    sumcharfct = sumcharfct.real / m
    return sumcharfct




def t_char_fct_vec(z,mu,sigma,nu,device):
    """ z is a tensor of shape (n,d) and corresponds arguments of characteristic function of d-dimensional t-distribution with odd degree of freedom
        mu is a tensor of shape (1,d) and corresponds to non-centrality vector
        sigma is a tensor of shape (d,d) and corresponds to the scale matrix
    """
    if not ((nu-1)%2==0 ):
        print("Degrees of freedom are not odd" ,nu)
        return  cmath.nan
    const=np.sqrt(np.pi)*scipy.special.gamma((nu+1)/2)/((2**(nu-1))*scipy.special.gamma(nu/2))
    m=torch.matmul(z,mu).to(device)
    m=torch.complex(real=torch.tensor([0],dtype=torch.float32).to(device),imag=m).to(device)
    sqrt= np.sqrt(nu)*torch.sqrt(torch.sum ( torch.mul( torch.matmul(z,sigma),z ) ,dim=1 ) ).to(device) #to be checked again
    u=int((nu+1)/2)
    sum=torch.tensor(np.zeros(shape=z.shape[0]),dtype=torch.float32).to(device)
    for i in np.arange(u):
        fac=torch.tensor( np.math.factorial(2*u-(i+1)-1)/(np.math.factorial(2*u-(i+1)-1 -(u-(i+1)))*np.math.factorial(u-(i+1))) ,dtype=torch.float32).to(device)
        sum=sum+fac*torch.pow(2*sqrt,i)/ np.math.factorial(i)
    sqrt= torch.complex(real=-sqrt,imag=torch.tensor([0], dtype=torch.float32).to(device))
    return(const*torch.mul(sum,torch.exp(m+sqrt)))

def char_fct_rational_quad_vec(z,alpha,beta,mu,device):
    """characteristic function derived from rational quadratic kernel
    z is input tensor of shape [n,d]
    alpha,beta are the parameters of the rational quadratic kernel
    mu is a tensor of shape [1,d]  the shift of the distribution
    """
    m=torch.matmul(z,mu).to(device)
    imag_part=torch.exp(torch.complex(real=torch.tensor([0],dtype=torch.float32).to(device),imag=m)).to(device) # exp(i mu^t * z) 
    norm=torch.sum(torch.mul(z,z),dim=1).to(device)
    inner=(1+beta*norm/(2*alpha)).to(device)
    real_part=torch.complex(real=torch.pow(inner,-alpha),imag=torch.tensor([0],dtype=torch.float32).to(device)).to(device) #rational quad kernel without shift
    return torch.mul(real_part,imag_part)  

def char_fct_alpha_stab(z,alpha,shift,sample,device):
    """characteristic function of the multivariate alpha-stable distribution and Gaussian measure on the unit-sphere
    z is input tensor of shape [n,d]
    alpha is the parameter of the stable distribution
    shift is a tensor of shape [1,d]  the shift of the distribution
    c is the covariance matrix of the underlying gaussian distribution
    m is the mean vector of the gaussian distribution
    size is the sample size of the monte carlo approximation of the integral over the unit sphere
    The rerpesentation of the alpha stable characteristic function is based on Sato (1999) Theorem 14.10
    """
    size=sample.shape[0]
    if alpha==1:
        v=torch.matmul(sample,torch.t(z)).to(device)
        vv=torch.abs(v).to(device)
        log=torch.log(vv).to(device)
        prod=torch.complex(imag= (2/np.pi)*torch.mul(v,log), real=torch.zeros(vv.shape).to(device)).to(device)
        sum=vv+prod
        #very rarely there are numerical problems that have to be handeled
        if (torch.sum(torch.isnan(sum))>0) or (torch.sum(torch.isinf(sum)>0)):
            sum[torch.isnan(sum)]=0
            sum[torch.isinf(sum)]=0
        tz=torch.complex(imag=torch.matmul(z,shift),real=torch.zeros(z.shape[0]).to(device)).to(device)
        s=torch.exp(-torch.t(torch.sum(sum,dim=0))/size+tz).to(device)
    else:
        v=torch.matmul(sample,torch.t(z)).to(device)
        vv=torch.pow(torch.abs(v),alpha).to(device)
        fac=torch.complex(imag=-torch.tan(np.pi*alpha*torch.sign(v)/2),real=torch.ones(v.shape).to(device)).to(device)
        prod=torch.mul(vv,fac)
        tz=torch.complex(imag=torch.matmul(z,shift),real=torch.zeros(z.shape[0]).to(device)).to(device)
        s=torch.exp(-torch.t(torch.sum(prod,dim=0))/size+tz).to(device)
    return s
    

def MMD_mixed_case(x:torch.tensor,y:torch.tensor,device,b=1):

    """calculates the MMD between the sample and the target distribution 

    Args:
        x (torch.tensor): the current sample distribution 
        y (torch.tensor): the target distribution which we want to model
        device (_type_): device object for PyTorch
        b (int, optional): the bandwith value. Defaults to 1.

    Returns:
        _type_: the mixed case of the MMD which is a value
    """

    n = x.shape[0]
    m = y.shape[0]

    # calculates euclidean distance tensor
    cdist = torch.cdist(x,y)
 
    #calculate kernel values
    kernel_values = kernel_gauss_cdist(cdist,b).to(device)

  
    #get the sum
    sum = torch.sum(kernel_values) 

    #weighting the sum
    res = 2/(m*n) * sum

    return res

def MMD_equal_case(x:torch.tensor,device,b:float=1):
    """calculates the sample x sample and response x response case for the MMD

    Args:
        x (torch.tensor): the input tensor
        device (_type_): the PyTorch device object
        b (float, optional): the bandwith value . Defaults to 1.

    Returns:
        _type_: the MMD value for the equal case (a single value)
    """

    n = x.shape[0]
    
    cdist = torch.cdist(x,x,p=2).to(device)
    
    
    #calculate kernel values
    kernel_values = kernel_gauss_cdist(cdist,b)
    
    #get the diagonal
    diag_kernel_values = kernel_values.diagonal()

    #remove the diagonal j==i elements
    sum = torch.sum(kernel_values) - torch.sum(diag_kernel_values)
    
    res = 1/(n*(n-1)) * sum
    

    return res


def kernel_gauss_cdist(cdist:torch.tensor, b:int=1)->torch.tensor:
    """calculates the tensor (matrix) of gaussian kernel values calculated using cdist

    Args:
        cdist (torch.tensor): a 2d tensor of cdist output values
        b (int, optional): the bandwidth. Defaults to 1.

    Returns:
        torch.tensor: the resulting gaussian kernel value tensor 
    """

    res = torch.exp(-b*((cdist)**2))

    return res