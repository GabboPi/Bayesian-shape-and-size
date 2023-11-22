include("mcmc.jl")
Random.seed!(123)

N = 20; #Numero di configurazioni
K = 3; #Numero di landmark per ogni configurazione
d = 1; #Numero di covariate
p = 3; # Numero di coordinate
z = [1] #Matrix of covariates

#Matrice di design
Z = kron(I(K),z)
#Matrice di varianza e covarianza
Sigma = [1 0.5 0.3; 0.5 2 0.7; 0.3 0.7 1]

VarCov = kron(I(p),Sigma)


#Media vera
Beta1 = [-7; 1; 15]
Beta2 = [6; -9; -2]
Beta3 = [-5; 12; 7] 

mu  = [Beta1; Beta2; Beta3];

#Identifico la media
mu = GS(reshape(mu,3,3))
mu = reshape(mu,9)


#Build dataset
samples, Y, R_true, theta_true = makedataset(N,K,p,mu,VarCov);
theta_sim = [1 1 1]
beta_sim = 0
Sigma_sim = 0


I_max = 30000
burn_in = 20000
thin = 1
original = 0 #Uso le rotazioni vere
@time B, Sigma_est, theta, R = mcmc(I_max, burn_in, thin, d,K,p,N,Z,Y, original, samples,theta_true,theta_sim,beta_sim,Sigma_sim);
if original == 1
    plot_mcmc(identify(B),Sigma_est,GS(reshape(mu,3,3)),Sigma,identify_samples(theta),identify_angles(theta_true))
end

m = mean(B,dims =1);
m = reshape(m,3,3);
GS(m)
reshape(mean(Sigma_est, dims = 1),3,3)

plot_mcmc(identify(B),Sigma_est,GS(reshape(mu,3,3)),Sigma,R,R_true)
#plot_mcmc(identify(B),Sigma_est,GS(reshape(mu,3,3)),Sigma,identify_samples(theta),identify_angles(theta_true))