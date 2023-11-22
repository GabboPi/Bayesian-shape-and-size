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

    #Siccome la funzione arcocoseno ha come codominio [0, π] devo controllare, nel caso in cui stia campionando da 
    #una Von-Mises, che l'angolo non sia fupri da questo dominio. A tale scopo, controllo il segno del seno. 
    if type == 1
        #=if(sgamma > 0)
            gamma = acos(cgamma)
        else
            gamma = 2pi-acos(cgamma)
        end=#
        gamma = atan(sgamma,cgamma)
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

        if type == 2
            y = y%pi
        end


        #Se type = 2 uso la VonMises come Kernel e faccio accept-reject, in cui il rapporto è pari a sin(y)
        if( (u <= sin(y)) && (type == 2))
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
    #Tensore degli angoli
    theta_true = zeros(N,3)
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
        if(det(F.Vt) < 0)
            P = [-1 0 0; 0 1 0; 0 0 1]
            elseif (det(F.Vt) > 0)
                P = I(p)
            elseif (det(F.Vt == 0))
                print("Matrix is singular!")
                break
            end
        V = F.V*P
        U = F.U*P
        Y[i,:,:] = U*Diagonal(F.S)
        R_true[i,:,:] = V
        theta_true[i,:] = angles(R_true[i,:,:])
        end
    return samples, Y, R_true, theta_true
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

function mcmc_theta!(N,B,Sigma,theta_last, theta_true::Union{Array,Nothing}=nothing, theta_sim::Union{Array,Nothing}=nothing)
    
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

        if isnothing(theta_true)
            #Campiono theta_1
            theta[s,1] = sample(H,1)
            #Campiono theta_2
            theta[s,2] = sample(D,2)
            #Campiono theta_3
            theta[s,3] = sample(L,1)
            #Utilizzo gli angoli appena campionati per costruire la matrice di rotazione
            R[s,:,:] = Rotation(theta[s,1], theta[s,2], theta[s,3])
        else
            #=if theta_sim[1] == 0
                theta[s,1] = theta_true[s,1]
            else
                theta[s,1] = sample(H,1)
            end
            
            if theta_sim[2] == 0
                theta[s,2] = theta_true[s,2]
            else
                theta[s,2] = sample(D,2)
            end

            if theta_sim[3] == 0
                theta[s,3] = theta_true[s,3]
            else
                theta[s,3] = sample(L,3)
            end
            =#

            theta[s,1] = theta_true[s,1]*(1-theta_sim[1]) + sample(H,1)*theta_sim[1]
            #Campiono theta_2
            theta[s,2] = theta_true[s,2]*(1-theta_sim[2]) + sample(D,2)*theta_sim[2] 
            #Campiono theta_3
            theta[s,3] = theta_true[s,3]*(1-theta_sim[3]) + sample(L,1)*theta_sim[3]
            #Utilizzo gli angoli appena campionati per costruire la matrice di rotazione
            R[s,:,:] = Rotation(theta[s,1], theta[s,2], theta[s,3])
        end
        

    end
    return theta, R
end
    
function mcmc(I_max, burn_in, thin, d,K,p,N,Z,Y, original = 0,samples=zeros(N,K,p), theta_true::Union{Array,Nothing} = nothing, theta_sim::Union{Array,Nothing} = nothing, beta_sim::Int = 0, Sigma_sim::Int = 0 )

    B,M,V,nu,Psi,Sigma_est,theta,R,X = mcmc_setup(I_max,burn_in,thin,d,K,p,N);

    for i = 2:I_max
        if original == 0
            X[i,:,:,:] = sample_update!(X[i,:,:,:],R[i-1,:,:,:],Y)
        elseif original == 1
            X[i,:,:,:] = samples
        
        end

        if beta_sim == 1
            B[i,:,:,:]=mcmc_B!(N,p,K,d,Sigma_est[i-1,:,:],Z,V,M,X[i-1,:,:,:]);
        else
            Beta1 = [-7; 1; 15]
            Beta2 = [6; -9; -2]
            Beta3 = [-5; 12; 7] 
            mu  = reshape([Beta1; Beta2; Beta3],3,3)
            #mu = GS(mu)
            B[i,:,:,:]= mu
        end

        if Sigma_sim == 1
            Sigma_est[i,:,:]=mcmc_Sigma!(N,K,p,nu,Psi,X[i-1,:,:,:],Z,B[i-1,:,:,:]);
        else
            Sigma_est[i,:,:]= [1 0.5 0.3; 0.5 2 0.7; 0.3 0.7 1]
        end
        
        if original == 0 && isnothing(theta_true)
            theta[i,:,:], R[i,:,:,:] = mcmc_theta!(N,B[i-1,:,:,:],Sigma_est[i-1,:,:],theta[i-1,:,:]);
        elseif !isnothing(theta_true)
            theta[i,:,:], R[i,:,:,:] = mcmc_theta!(N,B[i-1,:,:,:],Sigma_est[i-1,:,:],theta[i-1,:,:], theta_true, theta_sim)
        end
        
        if i%1000 == 0
            print("Iteration counter: ",i, '\n')
        end
    end

    return B[burn_in:thin:end,:,:,:], Sigma_est[burn_in:thin:end,:,:], theta[burn_in:thin:end,:,:], R[burn_in:thin:end,:,:,:]
end

function plot_mcmc(B,Sigma,B_true,Sigma_true,R,R_true)
    if isdir("Plots/") == false
        mkdir("Plots/")
    end
    I = size(B)[1]
    K = size(B)[3]
    p = size(B)[4]

    print("Plotting B and Sigma...")
    p_B = Array{Plots.Plot{Plots.GRBackend},1}()
    p_S = Array{Plots.Plot{Plots.GRBackend},1}()
    for i = 1:K
        for j = 1:p
        m = minimum(B[:,1,i,j])-1
        M = maximum(B[:,1,i,j])+1
        p1 = hline!(plot(B[:,1,i,j], size = (1920,1080),legend = true,xlims=(0,I), ylims = (m,M)),[B_true[i,j]])
        m = minimum(Sigma[:,i,j])-1
        M = maximum(Sigma[:,i,j])+1
        p2 = hline!(plot(Sigma[:,i,j], size = (1920,1080), legend = true,xlims = (0,I)),[Sigma_true[i,j]])
        push!(p_B, p1)
        push!(p_S, p2)
        end
    end

    lab = reshape(repeat(["sample";"true"],K*p),1,2*K*p)
    name_B = reshape(["B"*"_"*string(i)*"_"*string(j) for i =1:3 for j = 1:3],1,K*p)
    name_S = reshape(["S"*"_"*string(i)*"_"*string(j) for i =1:3 for j = 1:3],1,K*p)
    p_B = plot(p_B..., layout = K*p, title = name_B, labels = lab)
    p_S = plot(p_S..., layout = K*p, title = name_S, labels = lab)
    savefig(p_B,"Plots/Beta.png")
    savefig(p_S,"Plots/Sigma.png")
    print("Done! \n")

    print("Plotting angles...")
    theta = identify_R_angles(B,R)
    theta_true = identify_R_angles_true(B_true,R_true)
    plot_angles(theta,theta_true)
    print("Done! \n")
    print("Plotting R...")
    R1 = identify_R(B,R)
    R2 = identify_R_true(B_true,R_true)
    plot_R(R1,R2)
    print("Done!")

end

function plot_R(R,R_true)

    I = size(R)[1]
    N = size(R)[2]
    K = size(R)[3]
    p = size(R)[4]

    if isdir("Plots/R") == false
        mkdir("Plots/R")
    end

    p_R = Array{Plots.Plot{Plots.GRBackend},1}()
    for s =1:N
        for i = 1:K
            for j = 1:p
            m = minimum(R[:,s,i,j])-1
            M = maximum(R[:,s,i,j])+1
            p1 = hline!(plot(R[:,s,i,j], size = (1920,1080),legend = true, xlims = (0,I), ylims = (m,M)),[R_true[s,i,j]])
            push!(p_R, p1)
            end
        end
    end

    N_p = N*p*p
    lab = reshape(repeat(["sample";"true"],N_p),1,2*N_p)
    name_R = reshape(["R"*"_"*string(s)*"_"*string(i)*"_"*string(j) for s =1:N for i =1:3 for j = 1:3],1,N_p)
    for k = 0:N-1
        p_S = p_R[9k+1:9k+9]
        title_S = reshape(name_R[9k+1:9k+9],1,p*p)
        labels_S = reshape(lab[18k+1: 18+18k],1,2*p*p)
        p_S1 = plot(p_S..., layout = 9, title = title_S, labels = labels_S)
        savefig(p_S1,"Plots/R/R_"*string(k+1)*".png")
        print("Plot "*string(k+1)*" finished! \n")
    end

end

function plot_angles(theta,theta_true)

    I = size(theta)[1]
    N = size(theta)[2]
    K = 3
    if isdir("Plots/Theta") == false
        mkdir("Plots/Theta")
    end
    p_R = Array{Plots.Plot{Plots.GRBackend},1}()
    for s =1:N
        for i = 1:K
            if i != 2
                p1 = hline!(plot(theta[:,s,i], size = (1920,1080),legend = true,ylims=(0,2pi),xlims=(0,I)),[theta_true[s,i]])
            else
                p1 = hline!(plot(theta[:,s,i], size = (1920,1080),legend = true,ylims=(0,pi), xlims = (0,I)),[theta_true[s,i]])
            end
            push!(p_R, p1)
        end
    end

    N_p = N*K
    lab = reshape(repeat(["sample";"true"],N_p),1,2*N_p)
    name_R = reshape(["theta"*"_"*string(s)*"_"*string(i) for s =1:N for i =1:3],1,N_p)
    for k = 0:N-1
        p_S = p_R[3k+1:3k+3]
        title_S = reshape(name_R[3k+1:3k+3],1,K)
        labels_S = reshape(lab[6k+1: 6+6k],1,2*K)
        p_S1 = plot(p_S..., layout = 3, title = title_S, labels = labels_S)
        savefig(p_S1,"Plots/Theta/theta_"*string(k+1)*".png")
        println("Plot"*string(k+1)*" finished!")
    end

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

function angles(R)
        theta2 = acos(R[3,3])
        #theta = pi - theta
        theta3 = atan(R[1,3]/sin(theta2),R[2,3]/sin(theta2))
        theta1 = atan(R[3,1]/sin(theta2), -R[3,2]/sin(theta2))
        #theta1 = acos(-R[3,2]/sqrt(1-R[3,3]^2))
        #psi = atan(R[3,2]/cos(theta), R[3,3]/cos(theta))
        #theta3 = acos(R[2,3]/sqrt(1-R[3,3]^2))
        #=if theta1 < 0
            theta1 += 2pi
        end
        if theta3 < 0
            theta3 += 2pi
        end
        if theta2 < 0
            theta2+= pi/2
        end=#
    return [theta1,theta2,theta3]
end

function identify_t!(theta)
    
    if theta[1] < -2pi
        theta[1] %= 2pi
    end
    if theta[2] < -pi
        theta[2] %= pi
    end
    if theta[3] < -2pi
        theta[3] %= 2pi
    end
    
    
    if theta[1] <0
        theta[1] += 2pi
    end
    if theta[2] <0
        theta[2] += pi
    end
    if theta[3] <0
        theta[3] += 2pi
    end

    if theta[1] > 2pi
        theta[1] %= 2pi
    end
    if theta[2] > pi
        theta[2] %= pi
    end
    if theta[3] > 2pi
        theta[3] %= 2pi
    end


    return theta
end

function identify_angles(theta)
    N = size(theta)[1]
    for i = 1:N
        theta[i,:] = identify_t!(theta[i,:])
    end
    return theta
end

function identify_samples(theta)
    N = size(theta)[1]
    for i = 1:N
        theta[i,:,:] = identify_angles(theta[i,:,:])
    end
    return theta
end

function verify_svd(X,Y,theta)
    R = Rotation(theta[1],theta[2],theta[3])
    return X-Y*R'
end

function a_mises(rho,gamma,x)
    return exp(rho*cos(x-gamma))/(2pi*besseli(0,x))*sin(gamma)
end

function get_V(B)
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
    return V
end

function identify_R_angles(B,R)
    R_new = copy(R)
    dim = size(R)
    N = dim[1]
    S = dim[2]
    K = dim[3]
    p = dim[4]
    thetas = zeros((N,S,3))
    for i = 1:N
        for s = 1:S
            R_new[i,s,:,:] = R[i,s,:,:]*get_V(reshape(B[i,1,:,:],3,3))
            #R_new[i,s,:,:] = R[i,s,:,:]*get_V(reshape(B[i,1,:,:],3,3))
            thetas[i,s,:] = identify_t!(angles(R_new[i,s,:,:]))
        end
    end
    return thetas
end

function identify_R(B,R)
    R_new = copy(R)
    dim = size(R)
    N = dim[1]
    S = dim[2]
    K = dim[3]
    p = dim[4]
    thetas = zeros((N,S,3))
    for i = 1:N
        for s = 1:S
            R_new[i,s,:,:] = R[i,s,:,:]*get_V(reshape(B[i,1,:,:],3,3))
        end
    end
    return R_new
end

function identify_R_true(B_true,R_true)
    dim = size(R_true)
    S = dim[1]
    K = dim[2]
    p = dim[3]

    R_true_new = copy(R_true)
    G = get_V(B_true)
    for s = 1:S
        R_true_new[s,:,:] = R_true[s,:,:]*G
    end
    return R_true_new
end

function identify_R_angles_true(B_true,R_true)
    dim = size(R_true)
    S = dim[1]
    K = dim[2]
    p = dim[3]

    R_true_new = copy(R_true)
    thetas = zeros((S,3))
    G = get_V(B_true)
    for s = 1:S
        R_true_new[s,:,:] = R_true[s,:,:]*G
        thetas[s,:] = identify_t!(angles(R_true_new[s,:,:]))
    end
    return thetas
end
