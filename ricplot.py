import numpy as np
import scipy.linalg as sc
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib as mpl

fig_width_pt = 413.85834  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 11,
          'text.fontsize': 11,
          'legend.fontsize': 11,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
           'text.usetex': True,
           'figure.figsize': fig_size}
mpl.rcParams.update(params)
mpl.rcParams['text.latex.unicode'] = True

#########################################
#########################################

def norm(x):
    return np.amax(np.absolute(np.linalg.eig(x)[0]))

#########################################
#########Q=G=1, T=1, H=L2[0,1]###########
#########################################

def s(N):
    #spatial step-size
    h = 1.0/float(N)
    #temporal step-size
    k = h**(2.1)
    #number of temporal steps
    K = int(1.0/k)
    #overlapp/mass matrix
    S = np.zeros((N+1,N+1))
    for i in range(N+1):
        S[i,i] = 2.0/3*h
    for i in range(N):
        S[i,i+1] = 1.0/6*h
        S[i+1,i] = S[i,i+1]    
    #stiffness matrix
    A = np.zeros((N+1,N+1))
    for i in range(N+1):
        A[i,i] = 2.0/h
    for i in range(N):
        A[i,i+1] = -1.0/h
        A[i+1,i] = A[i,i+1]
    #S**(-1)(.)S**(-1) with schur decomposition
    s1,s2 = sc.schur(S, output='real')
    def tran(E):
        X = np.zeros((N+1,N+1))
        for i in range(N+1):
            for j in range(N+1):
                X[i,j] = np.dot(s2.transpose(),np.dot(E,s2))[i,j]/(s1[i,i]*s1[j,j])
        return np.dot(s2,np.dot(X,s2.transpose()))
    # silvester parameters
    I = np.identity(N+1)
    R2 = k*np.linalg.solve(S,A)
    R1 = I+k*np.transpose(R2)
    # Compute the Schur decomp
    r,u = sc.schur(R1, output='real')
    s,v = sc.schur(R2.transpose(), output='real')
    # Define the solver for each temporal step
    def sylvester(q):
        f = np.dot(np.dot(u.transpose(), q), v)
        y = np.zeros((N+1,N+1))
        for i in range(N+1):
            for j in range(N+1):
                y[i,j]=f[i,j]/(r[i,i]+s[j,j])
        return np.dot(np.dot(u, y), v.transpose())
    # Solve one step
    P = S
    i = 0
    while i <= K-1:
        D1 = k*S+P-k*np.dot(P,np.linalg.solve(S,P))
        D2 = k*np.zeros((N+1,N+1))
        PT = P
        D2[0,0] = h**2*(PT[0,0]*(1.0/4+1.0/72)+PT[0,1]*(1.0/12+1.0/72)+PT[1,1]*3/144)
        D2[N,N] = h**2*(PT[N,N]*(1.0/4+1.0/72)+PT[N-1,N]*(1.0/12+1.0/72)+PT[N-1,N-1]/144)
        for n in range(1,N):
            D2[n,n] = h**2*(PT[n,n]*(1.0/4+1.0/72)+PT[n,n+1]*(1.0/12+1.0/72)+PT[n-1,n]*(1.0/12+1.0/72)+PT[n-1,n+1]/72+PT[n-1,n-1]/144+PT[n+1,n+1]*3/144)
        for n in range(N):
            D2[n,n+1] = h**2*(PT[n+1,n]/12+PT[n+1,n+1]*(1.0/24+1.0/144)+PT[n,n]*(1.0/24+1.0/144)+PT[n,n+1]/72)
            D2[n+1,n] = D2[n,n+1]
        for n in range(N-1):
            D2[n,n+2] = h**2*(PT[n+2,n]+P[n+2,n+1]+PT[n+1,n]+PT[n+1,n+1])/144
            D2[n+2,n] = D2[n,n+2]
        P = sylvester(D1+D2)
        i = i+1
        print i
    return P
    
#####################################################################
#####################################################################

#reference solution

refN = 200
ref = s(refN)

#error in norm

def deltaP(N):
    h = 1/float(N)
    #overlapp/mass matrix
    S = np.zeros((N+1,N+1))
    for i in range(N+1):
        S[i,i] = 2.0/3*h
    for i in range(N):
        S[i,i+1] = 1.0/6*h
        S[i+1,i] = S[i,i+1]    
    #mesh and Functionspace
    mesh = np.zeros(N+1)
    for i in range(N+1):
        mesh[i]=i*h
    #Lagrange Functions
    def phi(i,x):
        return max(0,1-abs(mesh[i]-x)/h)
    #Ref mesh
    refmesh = np.zeros(refN+1)
    for i in range(refN+1):
        refmesh[i]=i/float(refN)
    #Ref Lagrange Functions
    def psi(i,x):
        return max(0,1-abs(refmesh[i]-x)*float(refN))
    #Lagrange Functions
    def phi(i,x):
        return max(0,1-abs(mesh[i]-x)/h)
    #Ref overlapp/mass matrix
    Sref = np.zeros((N+1,refN+1))
    for i in range(N+1):
        for j in range(refN+1):
            Sref[i,j] = integrate.quad(lambda x: phi(i,x)*psi(j,x), max(0,refmesh[j]-1/float(refN)), refmesh[j]+1/float(refN))[0]
    # ~extrapolate~ the solution
    Extra = np.linalg.solve(S,Sref)
    # Input solution
    P = s(N)
    # Differenz input solution, referenz solution
    B = np.dot(np.transpose(Extra),np.dot(P,Extra))-ref
    # return norm
    return norm(B)
    
########################################################################
########################################################################


#N = (10**np.linspace(0.5,1.5,20)).astype(int)
#X = 1.0/N
#Y = [deltaP(1,1,reftsteps,n,1) for n in N]

N = 2*np.array([2,4,8,16,32])
X = 1.0/N
Y = [deltaP(n) for n in N]

plt.loglog(N,Y, 'ro', label=r"Error")
plt.loglog(N,X,color='k',ls='--', label=r"$h^1$")
plt.xlabel(r"$N_h$")
plt.legend()
plt.show()

