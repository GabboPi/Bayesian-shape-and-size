include("mcmc.jl")
Random.seed!(123)

N = 20; #Numero di configurazioni
K = 3; #Numero di landmark per ogni configurazione
d = 2; #Numero di covariate
p = 3; # Numero di coordinate
z = repeat([1 2],N) #Matrix of covariates





#Matrice di design
Z = zeros(N,K,K*d)
for i =1:N
    Z[i,:,:] = kron(I(K),reshape(z[i,:],1,d)) #Mi assicuro che il vettore z[i,:] sia riga
end
#Matrice di varianza e covarianza
Sigma_true = 0.01*[1.0 0.5 0.3; 0.5 2.0 0.7; 0.3 0.7 1.0]

VarCov = kron(I(p),Sigma_true)


#Media vera
Beta1 = [-7.0; 1.0; 15.0]
Beta2 = [6.0; -9.0; -2.0]
Beta3 = [-5.0; 12.0; 7.0] 

mu  = [Beta1; Beta2; Beta3];

B_true = repeat(reshape(mu,K,p),outer = [1, 1, d])
B_true = permutedims(B_true,(3,1,2))

#Identifico la media
#mu = GS(reshape(mu,3,3))
mu = reshape(mu,9)


#Build dataset
samples, Y, R_true, theta_true = makedataset(N,K,p,mu,VarCov);
theta_sim = [0 0 0]
beta_sim = 0
Sigma_sim = 1


I_max = 30000
burn_in = 20000
thin = 1
original = 0 #Uso le rotazioni vere

@time B, Sigma_est, theta, R = mcmc(I_max, burn_in, thin, d,K,p,N,Z,Y, original, samples,B_true, Sigma_true, theta_true,theta_sim,beta_sim,Sigma_sim);

m = mean(B,dims =1);
m = reshape(m,d,K,p);

#plot_mcmc(identify(B),Sigma_est,GS(reshape(mu,3,3)),Sigma,R,R_true,"Pippo/")


T1 = [0]
T2 = [0]
T3 = [0]
B_v = [1]
S_v = [1]
#@time grid_mcmc(T1,T2,T3,B_v,S_v,I_max, burn_in, thin, d,K,p,N,Z,Y, original, samples,theta_true, R_true,mu)
