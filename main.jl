using LinearAlgebra
using Distributions
using Random
using Plots
Random.seed!(123)
# ------ FUNZIONI --------- #
function Helm(k)
    H = zeros(k,k);
    for j = 1:k 
        dj = 1/sqrt(j*(j+1))
        H[j,1:j-1] = -dj*ones(1,j-1);
        H[j,j] = j*dj
    end
    return H
end

function Rotation(theta1,theta2,theta3)
    #Funzione che prende in input i 3 angoli di Eulero con convenzione ZXZ e restituisce la corrispondente matrice di rotazione in R3
    R1 = [
        cos(theta1) sin(theta1) 0;
        -sin(theta1) cos(theta1) 0;
        0 0 1
    ]   

    R2 = [
     1 0 0;
     0 cos(theta2) sin(theta2);
     0 -sin(theta2) cos(theta2)
    
    ]

    R3 = [
        cos(theta3) sin(theta3) 0;
        -sin(theta3) cos(theta3) 0;
        0 0 1
    ]

    return R3*R2*R1
end

function sample(B,type)
    #Funzione che campiona dalla full conditional degli angoli di Eulero
    #Le full conditional sono ricavate a partire dalla amtrix Fisher definita su SO(3), dove la densità è proporzionale a exp(R'B)
    #Se type = 1 allora campiona da una VonMises con densità proporzionale a exp( rho * cos(theta-gamma) )
    # Se type = 2 allora campiona da una densità simil-VonMises di densità prop. a exp( rho * cos(theta-gamma) ) * sin(theta), con dominio [0,pi)
    flag = 0
    i = 0

    if type == 1
        a = B[1,1]+B[2,2]
        b = B[2,1]-B[1,2]
    elseif  type == 2
        a = B[2,2]+B[3,3]
        b = B[3,2]-B[2,3]
    else
        return NaN
    end

    #Calcolo rho, sin(gamma) e cos(gamma)
    rho = sqrt(a^2 + b^2)
    cgamma = a/rho
    sgamma = b/rho

    #Siccome la funzione arcocoseno ha come dominio (-pi/2, pi/2) devo controllare, nel caso in cui stia campionando da 
    #una Von-Mises, che l'angolo non sia fupri da questo dominio. A tale scopo, controllo il segno del seno. 
    if type == 1
        if(sgamma > 0)
            gamma = acos(cgamma)
        else
            gamma = 2*pi-acos(cgamma)
        end
    else
        #Se sto campionando da una simil-Von mises di dominio [o,π) il problema non si pone 
        gamma = acos(cgamma)
    end



    while flag == 0
        #Se rho vale zero ottengo un caso degenere, cioè un0uniforme su [o, 2π]
        if rho == 0
            y = rand(Uniform(0,2pi))
        else
            #Campiono da una VonMises
            y = rand(VonMises(gamma,rho))
        end
        #Campiono da un'uniforme (0,1)
        u = rand(Uniform(0,1))

        #Se type = 2 uso la VonMises come Kernel e faccio accept-reject, in cui il rapporto è pari a sin(y)
        if( (u <= sin(y)) & (type == 2))
            return y
        #Se type = 1 uso semplciemente il campione della VonMises
        elseif type == 1
            return y
        i = i+1
        end
    
    end
end

function metropolis(R,Y,mu,I_Sigma,last)
    #Prototipo di step Metropolis: da rivedere perchè molto lento.
    X = Y*R
    flag = 0
    while flag == 0
        
        #Calcolo le proposte (simmetriche) per gli angoli
        theta1 = rand(Normal(last[1],0.1))
        theta2 = rand(Normal(last[2],0.1))
        #Mi assicuro che l'angolo 2 sia tra [0,π)
        theta2 = theta2%pi
        theta3 = rand(Normal(last[3],0.1))

        global R1 = Rotation(theta1,theta2,theta3)
        global T = [theta1, theta2, theta3]
        X1 = Y*R1

        num = -0.5*tr( (X1-mu)'*I_Sigma*(X1-mu))
        den = -0.5*tr( (X-mu)'*I_Sigma*(X-mu))

        alpha = min(0, num-den)

        u = rand(Uniform(0,1))
        if log(u) < alpha
            flag = 1
        end
    end
    return [R1, T]
end

function GS(B)
    #Funzione che restituisce una versioen identificata ddella mtrice di rotazione considerata
    #Si tratta di una semplice ortogonalizzazione di Gram-Schmidt
    p = size(B)[2]
    A = B'
    V = zeros(p,p);
    V[:,1] = A[:,1]/norm(A[:,1])

    V[:,2] = A[:,2] - (A[:,2]'*V[:,1])*V[:,1]
    V[:,2] = V[:,2]/norm(V[:,2])

    V[:,3] = A[:,3] - ((A[:,3]'*V[:,1])*V[:,1]) - ((A[:,3]'*V[:,2])*V[:,2])
    V[:,3] = V[:,3]/norm(V[:,3])

    #Scelgo l'ultima colonna in modo da avere una amtric ein SO(3)
    if (det(V)!=1)
        V[:,3] = -V[:,3]
    end

    return B*V
end

function Riemann_distance(mu, mu_approx)
    #Funzione che calcola la distanza riemanniana tar due confgurazioni di forma
    Z1 = mu/norm(mu)
    Z2 = mu_approx/norm(mu_approx)

    A = Z2'Z1

    F = svd(A);

    return acos(sum(F.S))
end

function makedataset(N,K,p,mu,VarCov)
    #Tensore dei campioni
    samples = zeros(N,K,p);
    #Tensore delle configurazioni forma-scala
    Y = zeros(N,K,K)
    #RTensore delle amtrici di rotazione vere
    R_true = zeros(N,K,p)
    #Campiono N elementi da una normale multivariata
    for i = 1:N
        samples[i,:,:] = reshape(
            rand(
            MvNormal(mu, VarCov)
            ),
            K,p)
    end

    #Rimuovo la rotazione
    for i in 1:N
        global P = zeros(3,3)
        F = svd(samples[i,:,:])

        #Se la matrice V non è in SO(3) effettuo una permutazione
        #per assicurarmi di ottenere una rotazione
        if(det(F.Vt)== -1)
            P = [0 1 0; 1 0 0; 0 0 1]
            elseif (det(F.Vt)== 1)
                P = I(p)
            elseif (det(F.Vt == 0))
                print("Matrix is singular!")
            end
        V = F.V*P
        U = F.U*P
        Y[i,:,:] = U*Diagonal(F.S)
        R_true[i,:,:] = V'
    end
    return samples, Y, R_true
end

function mcmc_setup(I_max,burn_in,thin,d,K,p,N)
    #----Inizializzazione die parametri-----#
    # - Coefficienti regressivi - #
    #Tensore per le matrici dei coefficienti regressivi
    B = zeros(I_max,d,K,p);
    #Valore iniziale: scelgo la amtrice identità per evitare errori legati alla non hermitianità
    B[1,1,:,:] = I(p)
    # Parametri necessari per la full conditional
    M = zeros(p,K*d);
    V = zeros(p,K*d,K*d);
    for l = 1:p
        V[l,:,:] = 10^4*I(K*d);
    end

    # - Matrice di varianza e covarianza - #
    #Parametri necessari pe rla full conditional
    nu = K+1;
    Psi = I(K);
    #Tensore per la matrice di avrianza e covarianza
    Sigma_est = zeros(I_max,K,K)
    #Valore iniziale -> matrice identità
    Sigma_est[1,:,:] = I(K)


    # - Angoli e rotazioni - #
    #Tensore per le matrici di rotazione: una per ogni configurazione 
    R = zeros(I_max,N,p,p);
    #Matrice per gli angoli
    theta = zeros(I_max,N,3)
    #Inizializzo gli angoli a zero
    theta[1,:,1] = zeros(N);
    theta[1,:,2] = zeros(N);
    theta[1,:,3] = zeros(N);

    #Valori iniziali delle rotazioni
    for s = 1:N 
        R[1,s,:,:] = Rotation(theta[1,s,1], theta[1,s,2], theta[1,s,3]);
    end

    #Tensore dove salvare le configurazioni
    X = zeros(I_max,N,K,p)
    return B,M,V,nu,Psi,Sigma_est,theta,R,X
end

function sample_update!(X,R,Y)
    for s = 1:N
        X[s,:,:] = Y[s,:,:]*R[s,:,:]'
    end
    return X
end

function mcmc_B!(N,p,K,d,Sigma,Z,V,M,X)
    B = zeros(d,p,p)
    M_s = zeros(p,K*d,1);
    V_s = zeros(p,K*d,K*d);
    I_Sigma = inv(Sigma)
    for l = 1:p
        for s = 1:N
            V_s[l,:,:] = V_s[l,:,:] + Z'*I_Sigma*Z
            M_s[l,:] = M_s[l,:] + Z'*I_Sigma*X[s,:,l]
        end
        V_s[l,:,:] = V_s[l,:,:] +inv(V[l,:,:])
        M_s[l,:] = M_s[l,:] +inv(V[l,:,:])*M[l,:]

        V_s[l,:,:] = inv(V_s[l,:,:]);
        V_s[l,:,:] = Hermitian(V_s[l,:,:])
        M_s[l,:] = V_s[l,:,:]*M_s[l,:];
    
    #Campiono dalla full-conditional di beta_l
        B[:,:,l] = rand(MvNormal(
            M_s[l, :], V_s[l, :, :]
        ))
    end
    return B
end

function mcmc_Sigma!(N,K,p,nu,Psi,X,Z,B)
        #Ricavo i parametri necessari per campionare dalla full conditional di Sigma
        nu_s = nu+N*p;

        Psi_s = zeros(K,K);
        for s = 1:N
            for l = 1:p
                Psi_s = Psi_s + (X[s,:,l]-Z*B[:,:,l]')*(X[s,:,l]-Z*B[:,:,l]')'
            end
        end
        Psi_s = Psi_s + Psi
    
        #Campiono dalla full di Sigma
        Sigma = rand(
            InverseWishart(nu_s, Psi_s)
        )
        return Sigma
end

function mcmc_theta!(N,B,Sigma,theta_last)
    m = B[1,:,:]
    theta = zeros(N,3)
    R = zeros(N,p,p)
    for s = 1:N

        A = m'*inv(Sigma[:,:])*Y[s,:,:]

        theta1 = theta_last[s,1]
        theta2 = theta_last[s,2]
        theta3 = theta_last[s,3]

        #Matrici di rotazione associate ai 3 angoli di Eulero, con convenzione ZXZ
        R1 = [
            cos(theta1) sin(theta1) 0;
            -sin(theta1) cos(theta1) 0;
            0 0 1
        ]

        R2 = [
        1 0 0;
        0 cos(theta2) sin(theta2);
        0 -sin(theta2) cos(theta2)
        
        ]

        R3 = [
        cos(theta3) sin(theta3) 0;
        -sin(theta3) cos(theta3) 0;
        0 0 1
        ]


        L = R2*R1*A'
        H = A'*R3*R2
        D = R1*A'*R3


        #Campiono theta_1
        theta[s,1] = sample(H,1)
        #Campiono theta_2
        theta[s,2] = sample(D,2)
        #Campiono theta_3
        theta[s,3] = sample(L,1)
        
        #Utilizzo gli angoli appena campionati per costruire la matrice di rotazione
        R[s,:,:] = Rotation(theta[s,1], theta[s,2], theta[s,3])

    end
    return theta, R
end
    

function mcmc(I_max, burn_in, thin, d,K,p,N,Z,Y, original = 0,samples=zeros(N,K,p))

    B,M,V,nu,Psi,Sigma_est,theta,R,X = mcmc_setup(I_max,burn_in,thin,d,K,p,N);

    for i = 2:I_max
        if original == 0
            X[i,:,:,:] = sample_update!(X[i,:,:,:],R[i-1,:,:,:],Y)
        elseif original == 1
            X[i,:,:,:] = samples
        end
        B[i,:,:,:]=mcmc_B!(N,p,K,d,Sigma_est[i-1,:,:],Z,V,M,X[i-1,:,:,:]);
        Sigma_est[i,:,:]=mcmc_Sigma!(N,K,p,nu,Psi,X[i-1,:,:,:],Z,B[i-1,:,:,:]);
        if original == 0
            theta[i,:,:], R[i,:,:,:] = mcmc_theta!(N,B[i-1,:,:,:],Sigma_est[i-1,:,:],theta[i-1,:,:]);
        end
        if i%1000 == 0
            print("Iteration counter: ",i, '\n')
        end
    end

    return B[burn_in:thin:end,:,:,:], Sigma_est[burn_in:thin:end,:,:], theta[burn_in:thin:end,:,:], R[burn_in:thin:end,:,:,:]
end

function plot_mcmc(B,Sigma,B_true,Sigma_true)
    I = size(B)[1]
    K = size(B)[3]
    p = size(B)[4]

    p_B = Array{Plots.Plot{Plots.GRBackend},1}()
    p_S = Array{Plots.Plot{Plots.GRBackend},1}()
    for i = 1:K
        for j = 1:p
        p1 = hline!(plot(B[:,1,i,j], size = (1920,1080),legend = true),[B_true[i,j]])
        p2 = hline!(plot(Sigma[:,i,j], size = (1920,1080), legend = true),[Sigma_true[i,j]])
        push!(p_B, p1)
        push!(p_S, p2)
        end
    end

    lab = reshape(repeat(["sample";"true"],K*p),1,2*K*p)
    name_B = reshape(["B"*"_"*string(i)*"_"*string(j) for i =1:3 for j = 1:3],1,K*p)
    name_S = reshape(["S"*"_"*string(i)*"_"*string(j) for i =1:3 for j = 1:3],1,K*p)
    p_B = plot(p_B..., layout = K*p, title = name_B, labels = lab)
    p_S = plot(p_S..., layout = K*p, title = name_S, labels = lab)
    savefig(p_B,"Beta.png")
    savefig(p_S,"Sigma.png")

end

function identify(B)
    dims = size(B)
    I_max = dims[1]
    d = dims[2]
    K = dims[3]
    p = dims[4]
    B_identified = zeros(I_max,d,K,p)
    for i = 1:I_max
        B_identified[i,1,:,:] = GS(B[i,1,:,:])
    end
    return B_identified
end

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
samples, Y, R_true = makedataset(N,K,p,mu,VarCov);

I_max = 30000
burn_in = 20000
thin = 1
original = 1 #Uso le rotazioni vere
@time B, Sigma_est, theta = mcmc(I_max, burn_in, thin, d,K,p,N,Z,Y, original, samples);
if original == 1
    plot_mcmc(identify(B),Sigma_est,GS(reshape(mu,3,3)),Sigma)
end

m = mean(B,dims =1);
m = reshape(m,3,3);
GS(m)
reshape(mean(Sigma_est, dims = 1),3,3)