include("mcmc.jl")
Random.seed!(123)

N = 20; #Numero di configurazioni
K = 3; #Numero di landmark per ogni configurazione
d = 2; #Numero di covariate
p = 3; # Numero di coordinate
z = repeat([1.0 2.0],N) #Matrix of covariates

#Second covariate is sampled from a normal distribution
for i = 1:N
    mu = 10
    sigma = 1
    z[i,2] = rand(Normal(mu,sigma))
end





#Matrice di design
Z = zeros(N,K,K*d)
for i =1:N
    Z[i,:,:] = kron(I(K),reshape(z[i,:],1,d)) #Mi assicuro che il vettore z[i,:] sia riga
end

#Matrice di varianza e covarianza
nu = 5.0
A = rand(Uniform(0,2),K,K)
Psi = A*A'
Sigma_true = rand(InverseWishart(nu,Psi))

#Sigma_true = 0.1*Matrix(I(K))
VarCov = kron(I(p),Sigma_true)


#Media vera
mu = rand(Uniform(0,10),d*K*p)
B_true = reshape(mu,d,K,p)

#Se d = 1 bisogna fare attenzione a trasformare beta in un tensore [N,d,K,p]
#B_true = repeat(reshape(mu,K,p),outer = [1, 1, d])
#B_true = permutedims(B_true,(3,1,2))


#Build dataset
samples, Y, R_true, theta_true = makedataset(N,d,K,p,z,B_true,VarCov);


#MCMC parameters
I_max = 30000
burn_in = 20000
thin = 1
original = 0 #Se = 1, l'algoritmo usa ad ogni passo le rotazioni vere anzichè quelle simulate


#---- PRIORS PARAMETERS ---#
#Sigma
nu_prior = K+1;
Psi_prior = Matrix(1.0*I(K));
#Beta
M_prior = zeros(p,K*d);
V_prior = 10.0^4*Matrix(I(K*d))



#MCMC 
theta_sim = [1 1 1] #Se p =2, l'angolo da plottare è il primo--> [1 0 0]
beta_sim = 1
Sigma_sim = 1

plot_flag = [1 1 1 1] #Flag per sceglier equali parametri plottare, l'ordine è [B,Sigma,R,Theta]


@time B, Sigma_est, theta, R, X = mcmc(I_max, burn_in, thin, d,K,p,N,z,Z,Y,nu_prior,Psi_prior,M_prior,V_prior, original, samples,B_true, Sigma_true, theta_true,theta_sim,beta_sim,Sigma_sim);


B_true_tensor = permutedims(reshape(B_true,d,K,p,1),(4,1,2,3))

samples_id,X_id, B_id, B_true_id, R_id, R_true_id, theta_id, theta_true_id =identify_params(Y,B, B_true_tensor, Sigma_est, Sigma_true, R, R_true)

plot_mcmc(B_id,Sigma_est,B_true_id,Sigma_true,R_id,R_true_id,theta_id,theta_true_id,plot_flag)
#Uncomment to plot unidentified data
#plot_mcmc(B,Sigma_est, B_true_tensor,Sigma_true, R, R_true,theta,theta_true,plot_flag,"Unidentified/")


compare(X_id,samples_id)


#=
T1 = [0]
T2 = [0]
T3 = [0]
B_v = [0,1]
S_v = [0,1]
@time grid_mcmc(T1,T2,T3,B_v,S_v,I_max, burn_in, thin, d,K,p,N,Z,Y,nu_prior,Psi_prior,M_prior,V_prior, original, samples,B_true,Sigma_true,theta_true, R_true)
=#