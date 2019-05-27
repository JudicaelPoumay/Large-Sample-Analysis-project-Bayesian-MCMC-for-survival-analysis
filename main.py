import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt
from scipy.stats import gamma

def BMCMC(n, ys, fs, verbose = None):
    #init augmented data array
    ysAug   = np.copy(ys)
    
    #diffuse prior HP
    a=1
    b=1

    print("start MCMC")
    theta   = []
    for i in range(n):
        shape = a + len(ysAug)
        rate  = b + np.sum(ysAug)
        theta.append(1/np.random.gamma(shape,1/rate))
        for i,(y,f) in enumerate(zip(ys,fs)):        
            if(f != 1):
                ysAug[i] = y + np.random.exponential(theta[-1])
              
    #plot
    plt.plot(theta)
    plt.ylabel('theta')
    plt.xlabel('iteration')
    if(verbose is not None):
        plt.savefig(f"theta_convergence_{verbose}.png")
    plt.close()
    
    ret = plt.hist(theta,50)
    plt.close()
    plt.hist(theta[len(theta)//10:],50)
    plt.ylabel('theta')
    if(verbose is not None):
        plt.savefig(f"theta_distribution_{verbose}.png")
    plt.close()
    return theta, ret[0], ret[1]
    
def genUniCensData(n, cens, expt):
    print("gen data")
    ys      = np.random.exponential(expt,n)
    fs      = np.zeros_like(ys)
    fs[(ys < cens)] = 1
    ys[(ys >= cens)] = cens
    print(np.sum(ys >= cens))
    print(np.mean(ys))
    
    return ys, fs
    
def genRdCensData(n, cens, expt):
    print("gen data")
    ys      = np.random.exponential(expt,n)
    censs    = np.random.exponential(cens,n)
    fs      = np.zeros_like(ys)
    fs[(ys < censs)] = 1
    ys[(ys >= censs)] = censs[fs == 0] 
    print(np.sum(ys >= censs))
    print(np.mean(ys))
    
    return ys, fs
    
def probExp(y, f, param):
    if(f == 1):
        return param*math.exp(-y*param)
    else:
        return math.exp(-y*param)
    
def loglikelyhood(ys, fs, theta):
    res = 1
    for (y, f) in zip(ys, fs):
        res += math.log(probExp(y, f, 1/theta))
    return res

def probTheta(theta, binNb, binBound):# approximated since no easy way to compute gamma pdf (scipy is useless don't even try)
    for i,b in enumerate(binBound):
        if theta >= b:      
            return binNb[i]/sum(binNb)
    
def integratedLogLikelyhood(ys, fs, thethas, binNb, binBound):
    res = 0
    for theta in thethas:
        res += loglikelyhood(ys, fs, theta)*probTheta(theta, binNb, binBound)
    return res

def getExplanatoryVar(df):
    age     = df['age'].values.tolist()
    sex     = df['sex'].values.tolist()
    hgb     = df['hgb'].values.tolist()
    creat   = df['creat'].values.tolist()
    mspike  = df['mspike'].values.tolist()    
    
    return np.asarray([np.asarray([float(a),0.0 if s=='F' else 1.0,float(h),float(c),float(m)]) for (a,s,h,c,m) in zip(age,sex,hgb,creat,mspike)])
    
def transformToUniform(df,cens):
    df = df.copy()
    df = df[(df.futime >= cens) | (df.death == 1)]
    df.loc[(df.futime >= cens) & (df.death == 0), 'futime'] = cens
    
    return df
    
def evaluateBMCMC(unif):   
    #select generation method
    label = ""
    if(unif):
        gen = genUniCensData
        label = "uniform"
    else:
        gen = genRdCensData
        label = "random"

    #analyze effect of number of iteration on BMCMC performance
    ys, fs = gen(3000, 30, 100)
    ILL = []
    for i in range(10,300,10):
        thetas, binNb, binBound = BMCMC(i, ys,fs)
        ILL.append(integratedLogLikelyhood(ys, fs, thetas, binNb, binBound))
        
    plt.plot(range(10,300,10),ILL)
    plt.ylabel('integrated LogLikelyhood')
    plt.xlabel('number of iteration of B-MCMC')
    plt.savefig(f"iteration_{label}.png")
    plt.close()

    #analyze effect of censoring on BMCMC performance
    ILL = []
    for i in range(0,500,10):
        ys, fs = gen(3000, i, 100)
        thetas, binNb, binBound = BMCMC(500, ys,fs)
        ILL.append(integratedLogLikelyhood(ys, fs, thetas, binNb, binBound))
        
    plt.plot(range(0,500,10),ILL)
    plt.ylabel('integrated LogLikelyhood')
    plt.xlabel('right censoring on data')
    plt.savefig(f"censoring_{label}.png")
    plt.close()

    #analyze effect of data size on BMCMC performance
    ILL = []
    for i in range(100,4000,100):
        ys, fs = gen(i, 30, 100)
        thetas, binNb, binBound = BMCMC(500, ys,fs)
        ILL.append(integratedLogLikelyhood(ys, fs, thetas, binNb, binBound))
        
    plt.plot(range(100,4000,100),ILL)
    plt.ylabel('integrated LogLikelyhood')
    plt.xlabel('data size')
    plt.savefig(f"size_{label}.png")
    plt.close()
    
#load data and drop na
data = pd.read_csv("mgus2.csv")
data = data.drop("useless", axis=1)
data = data.drop("id", axis=1)
data = data.dropna(axis=0, how='any')

xs = getExplanatoryVar(data)


#evaluate
with open("results.txt", 'w+', encoding='utf-8') as out:
    ys = data['futime'].values
    fs = data['death'].values
    thetas, binNb, binBound = BMCMC(500, ys,fs,verbose = "real_data")
    print("Real data", file=out)
    print("####################################", file=out)
    print("Mean theta :"+str(np.mean(thetas)), file=out)
        
    dataUnif = transformToUniform(data,90)
    ys = dataUnif['futime'].values
    fs = dataUnif['death'].values
    thetas, binNb, binBound = BMCMC(500, ys,fs,verbose = "uniform real_data")
    print("Uniform real data", file=out)
    print("####################################", file=out)
    print("Mean theta :"+str(np.mean(thetas)), file=out)
    
    ys, fs = genUniCensData(5000, 15, 100)
    thetas, binNb, binBound = BMCMC(500, ys,fs,verbose = "low censoring")
    print("Low censoring", file=out)
    print("####################################", file=out)
    print("Mean theta :"+str(np.mean(thetas)), file=out)
    
    ys, fs = genUniCensData(5000, 70, 100)
    thetas, binNb, binBound = BMCMC(500, ys,fs,verbose = "mid censoring")
    print("Mid censoring", file=out)
    print("####################################", file=out)
    print("Mean theta :"+str(np.mean(thetas)), file=out)
    
    ys, fs = genUniCensData(5000, 200, 100)
    thetas, binNb, binBound = BMCMC(500, ys,fs,verbose = "high censoring")
    print("High censoring", file=out)
    print("####################################", file=out)
    print("Mean theta :"+str(np.mean(thetas)), file=out)   
# evaluateBMCMC(True)
# evaluateBMCMC(False)