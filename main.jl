using LinearAlgebra
using Distributions
using Random
using Plots

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

function makedataset(N,K,p,mu,VarCov,)
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

N = 20; #Numero di configurazioni
K = 3; #Numero di landmark per ogni configurazione
d = 1; #Numero di covariate
p = 3; # Numero di coordinate
z = [1] #Matrix of covariates

#Matrice di design
Z = kron(I(K),z)
#Matrice di varianza e covarianza
k = 0.1 
Sigma = k*I(K)
VarCov = kron(I(p),Sigma)


#Media vera
Beta1 = [60; 1; 100]
Beta2 = [10; 30; 180]
Beta3 = [20; 400; 0.5] 

mu  = [Beta1; Beta2; Beta3];

#Identifico la media
mu = GS(reshape(mu,3,3))
mu = reshape(mu,9)


#Build dataset
samples, Y, R_true = makedataset(N,K,p,mu,VarCov);
