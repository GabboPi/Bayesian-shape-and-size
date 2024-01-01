include("mcmc.jl")
Random.seed!(292215)

N = 50; #Numero di configurazioni
K = 3; #Numero di landmark per ogni configurazione
d = 2; #Numero di covariate
p = 3; # Numero di coordinate

if d ==1
    z = repeat([1.0],N) #Matrix of covariates
    #=
    for i = 1:N
        mu = 10
        sigma = 1
        z[i] = rand(Normal(mu,sigma))
    end
    =#
end

#Second covariate is sampled from a normal distribution
if d == 2
    z = repeat([1.0 0.0],N) #Matrix of covariates
    for i = 1:N
        mu = 10
        sigma = 1
        z[i,2] = rand(Normal(mu,sigma))
    end
    #=m = mean(z[:,2])
    sigma = sqrt(var(z[:,2]))
    for i = 1:N
        z[i,2] = (z[i,2]-m)/sigma
    end
    =#
end





#Matrice di design
Z = zeros(N,K,K*d)
for i =1:N
    #Z[i,:,:] = kron(I(K),reshape(z[i,:],1,d)) #Mi assicuro che il vettore z[i,:] sia riga
    Z[i,:,:] = kron(I(K),z[i,:]')
end

#Matrice di varianza e covarianza


nu = 5.0
A = rand(Uniform(0,2),K,K)
Psi = A*A'
Sigma_true = rand(InverseWishart(nu,Psi))

#=
k = 0.1
Sigma_true = k*Matrix(I(K))
=#
VarCov = kron(I(p),Sigma_true)


#Media vera
mu = rand(Normal(5,1),d*K*p)
B_true = reshape(mu,d,K,p)
mu_true = zeros(K,p)


#Build dataset
samples, Y, R_true, theta_true = makedataset(N,d,K,p,z,B_true,VarCov);


#MCMC parameters
I_max = 30000
burn_in = 20000
thin = 5
original = 0 #Se = 1, l'algoritmo usa ad ogni passo le rotazioni vere anzichè quelle simulate


#---- PRIORS PARAMETERS ---#
#Sigma
nu_prior = K+1;
Psi_prior = Matrix(1.0*I(K));
#Beta
M_prior = zeros(p,K*d);
V_prior = 10.0^6*Matrix(I(K*d))



#MCMC 
theta_sim = [1 1 1] #Se p =2, l'angolo da plottare è il primo--> [1 0 0]
beta_sim = 1
Sigma_sim = 1

plot_flag = [1 1 1 0] #Flag per sceglier equali parametri plottare, l'ordine è [B,Sigma,R,Theta]


@time B, Sigma_est, theta, R, X = mcmc(I_max, burn_in, thin, d,K,p,N,z,Z,Y,nu_prior,Psi_prior,M_prior,V_prior, original, samples,B_true, Sigma_true, theta_true,theta_sim,beta_sim,Sigma_sim);


B_true_tensor = permutedims(reshape(B_true,d,K,p,1),(4,1,2,3))

samples_id,X_id, B_id, B_true_id, R_id, R_true_id, theta_id, theta_true_id =identify_params(Y,B, B_true_tensor, Sigma_est, Sigma_true, R, R_true)

plot_mcmc(B_id,Sigma_est,B_true_id,Sigma_true,R_id,R_true_id,theta_id,theta_true_id,plot_flag)
#compare(X_id,samples_id)




#Siccome non ero sicuro dei risultati ottenuti, ho provato a confrontare la configurazione media usando le rotazioni vere e quelle simulate 
X_m = reshape(mean(X_id,dims=[1,2]),K,p)
samples_m = reshape(mean(samples_id,dims=1),K,p)
print("riemann distance is:")
Riemann_distance(samples_m,X_m)

#Uncomment to plot unidentified data
#plot_mcmc(B,Sigma_est, B_true_tensor,Sigma_true, R, R_true,theta,theta_true,plot_flag,"Unidentified/")
#Compare unidentified data
#compare(X,samples)

#=
T1 = [0]
T2 = [0]
T3 = [0]
B_v = [0,1]
S_v = [0,1]
@time grid_mcmc(T1,T2,T3,B_v,S_v,I_max, burn_in, thin, d,K,p,N,Z,Y,nu_prior,Psi_prior,M_prior,V_prior, original, samples,B_true,Sigma_true,theta_true, R_true)
=#