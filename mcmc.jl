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

function sample(B::Array{Float64},type::Int)
    #Funzione che campiona dalla full conditional degli angoli di Eulero
    #Le full conditional sono ricavate a partire dalla amtrix Fisher definita su SO(3), dove la densità è proporzionale a exp(RA')
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
    #una Von-Mises, che l'angolo non sia fupri da questo dominio. A tale scopo, uso arcotangente 2 e sommo 2pi se serve
    if type == 1
        gamma = atan(sgamma,cgamma)
        if(gamma <0)
            gamma += 2pi
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
            #Campiono da una VonMises.Attenzione: la VonMises su Julia campiona da [-π, π]
            y = rand(VonMises(gamma,rho))
        end
        #Campiono da un'uniforme (0,1)
        u = rand(Uniform(0,1))

        if type == 2
            #Mi assicuro che l'angolo campionato stia tra [o,π)
            if (y<0)
                y+=pi
            end
            y = y%pi
        end


        #Se type = 2 uso la VonMises come Kernel e faccio accept-reject, in cui il rapporto è pari a sin(y)
        if( (u <= sin(y)) && (type == 2))
            return y
        #Se type = 1 uso semplciemente il campione della VonMises, assicurandomi che stia tra [o, 2π]
        elseif type == 1
            if(y < 0)
                y += 2*pi
            end
            return y
        i = i+1
        end
    
    end
end

function GS(B::Array{Float64})
    #Funzione che restituisce una versioen identificata ddella mtrice di rotazione considerata
    #Si tratta di una semplice ortogonalizzazione di Gram-Schmidt
    p = size(B)[2]
    A = B'
    V = zeros(p,p);
    V[:,1] = A[:,1]/norm(A[:,1])

    if p == 3
        V[:,2] = A[:,2] - (A[:,2]'*V[:,1])*V[:,1]
        V[:,2] = V[:,2]/norm(V[:,2])
        V[:,3] = A[:,3] - ((A[:,3]'*V[:,1])*V[:,1]) - ((A[:,3]'*V[:,2])*V[:,2])
        V[:,3] = V[:,3]/norm(V[:,3])

        #Scelgo l'ultima colonna in modo da avere una amtric ein SO(3)
        if (det(V) < 0)
            V[:,3] = -V[:,3]
        end
    elseif p== 2
        V[:,2] = A[:,2] - (A[:,2]'*V[:,1])*V[:,1]
        V[:,2] = V[:,2]/norm(V[:,2])

        if det(V) < 0
            V[:,2] = -V[:,2]
        end
    end

    return B*V
end

function Riemann_distance(mu::Array{Float64}, mu_approx::Array{Float64})
    #Funzione che calcola la distanza riemanniana tar due confgurazioni di forma
    Z1 = mu/norm(mu)
    Z2 = mu_approx/norm(mu_approx)

    A = Z2'Z1

    F = svd(A);

    return acos(sum(F.S))
end

function makedataset(N::Int64,d::Int64,K::Int64,p::Int64,z::Array{Float64},B::Array{Float64},VarCov::Array{Float64})
    #Tensore dei campioni
    samples = zeros(N,K,p);
    #Tensore delle configurazioni forma-scala
    Y = zeros(N,K,p)
    #RTensore delle amtrici di rotazione vere
    R_true = zeros(N,p,p)
    #Tensore degli angoli
    theta_true = zeros(N,3)
    #Campiono N elementi da una normale multivariata
    for i = 1:N
        mu = zeros(K,p)
        for h = 1:d
            mu += z[i,h]*B[h,:,:]
        end
        samples[i,:,:] = reshape(
            rand(
            MvNormal(vec(mu), VarCov)
            ),
            K,p)
    end

    #Rimuovo la rotazione
    for i in 1:N
        global P = zeros(p,p)
        F = svd(samples[i,:,:])

        #Se la matrice V non è in SO(3) effettuo una permutazione
        #per assicurarmi di ottenere una rotazione
        if(det(F.Vt) < 0)
                if p == 3
                    P = [-1 0 0; 0 1 0; 0 0 1]
                elseif p == 2
                    P = [-1 0; 0 1]
                end
            elseif (det(F.Vt) > 0)
                P = I(p)
            elseif (det(F.Vt == 0))
                print("Matrix is singular!")
                break
            end
        V = F.V*P
        U = F.U*P
        Y[i,:,:] = U*Diagonal(F.S)
        R_true[i,:,:] = V'
        theta_true[i,:] = angles(R_true[i,:,:])
        end
    return samples, Y, R_true, theta_true
end

function mcmc_setup(I_max::Int64,burn_in::Int64,thin::Int64,d::Int64,K::Int64,p::Int64,N::Int64,M_prior::Array{Float64},V_prior::Array{Float64})
    #----Inizializzazione die parametri-----#
    # - Coefficienti regressivi - #
    #Tensore per le matrici dei coefficienti regressivi
    B = zeros(I_max,d,K,p);
    #Valore iniziale: scelgo la amtrice identità per evitare errori legati alla non hermitianità
    B[1,1,:,:] = ones(K,p)

    # Parametri necessari per la full conditional
    #M = zeros(p,K*d);
    M = M_prior
    V = zeros(p,K*d,K*d);
    for l = 1:p
        V[l,:,:] = V_prior;
    end

    # - Matrice di varianza e covarianza - #
    #Parametri necessari pe rla full conditional
    #nu = K+1;
    #Psi = Matrix(1.0*I(K));
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
    if p == 3
        for s = 1:N 
            #R[1,s,:,:] = Rotation(theta[1,s,1], theta[1,s,2], theta[1,s,3]);
            R[1,s,:,:] = Matrix(I(3))
        end
    elseif p == 2
        for s = 1:N 
            #R[1,s,:,:] = [cos(theta) sin(theta); -sin(theta) cos(theta)];
            R[1,s,:,:] = Matrix(I(2))
        end
    end

    #Tensore dove salvare le configurazioni
    X = zeros(I_max,N,K,p)
    return B,M,V,Sigma_est,theta,R,X
end

function sample_update!(X::Array{Float64},R::Array{Float64},Y::Array{Float64})
    for s = 1:N
        X[s,:,:] = Y[s,:,:]*R[s,:,:]
    end
    return X
end

function mcmc_B!(N::Int64,p::Int64,K::Int64,d::Int64,Sigma::Array{Float64},Z::Array{Float64},V::Array{Float64},M::Array{Float64},X::Array{Float64})
    B = zeros(d,K,p)
    M_s = zeros(p,K*d,1);
    V_s = zeros(p,K*d,K*d);
    I_Sigma = inv(Sigma)
    for l = 1:p
        for s = 1:N
            V_s[l,:,:] = V_s[l,:,:] + Z[s,:,:]'*I_Sigma*Z[s,:,:]
            M_s[l,:] = M_s[l,:] + Z[s,:,:]'*I_Sigma*X[s,:,l]
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

function mcmc_Sigma!(N::Int64,K::Int64,p::Int64,nu::Int64,Psi::Array{Float64},X::Array{Float64},Z::Array{Float64},B::Array{Float64})
        #Ricavo i parametri necessari per campionare dalla full conditional di Sigma
        nu_s = nu+N*p;

        Psi_s = zeros(K,K);
        for s = 1:N
            for l = 1:p
                Psi_s = Psi_s + (X[s,:,l]-Z[s,:,:]*vec(B[:,:,l]))*(X[s,:,l]-Z[s,:,:]*vec(B[:,:,l]))'
            end
        end
        Psi_s = Psi_s + Psi
    
        #Campiono dalla full di Sigma
         Sigma = rand(
            InverseWishart(nu_s, Psi_s)
        )
        return Sigma
end

function mcmc_theta!(N::Int64,B::Array{Float64},z::Array{Float64},Sigma::Array{Float64},theta_last::Array{Float64}, theta_true::Union{Array{Float64},Nothing}=nothing, theta_sim::Union{Array{Int64},Nothing}=nothing)
    d = size(B)[1]
    K = size(B)[2]
    p = size(B)[3]
    theta = zeros(N,3)
    R = zeros(N,p,p)
    for s = 1:N


        m = zeros(K,p)
        for h = 1:d
            m += z[s,h]*B[h,:,:]
        end

        A = m'*inv(Sigma[:,:])*Y[s,:,:]

        theta1 = theta_last[s,1]

        if p == 2
            theta2 = 0
            theta3 = 0
        elseif p > 2
            theta2 = theta_last[s,2]
            theta3 = theta_last[s,3]
        end

        #Matrici di rotazione associate ai 3 angoli di Eulero, con convenzione ZXZ
        R1 = [
            cos(theta1) sin(theta1) 0;
            -sin(theta1) cos(theta1) 0;
            0 0 1
        ]

        if p == 3
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

            #=L = A'*R1'*R2'
            H = R2'*R3'*A'
            D = R3'*A'*R1'  
            =#

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
                if theta_sim[1] == 0
                    #Controllo che gli angoli siano tra [0,2pi]
                    if(theta_true[s,1]<0)
                        theta_true[s,1] += 2pi
                    end
                    theta[s,1] = theta_true[s,1]
                else
                    theta[s,1] = sample(H,1)
                end
                
                if theta_sim[2] == 0
                    #Controllo che gli angoli siano tra [0,pi]
                    if(theta_true[s,2]<0)
                        theta_true[s,2] += pi
                    end
                    theta[s,2] = theta_true[s,2]
                else
                    theta[s,2] = sample(D,2)
                end

                if theta_sim[3] == 0
                    #Controllo che gli angoli siano tra [0,2pi]
                    if(theta_true[s,3]<0)
                        theta_true[s,3] += 2pi
                    end
                    theta[s,3] = theta_true[s,3]
                else
                    theta[s,3] = sample(L,1)
                end
                R[s,:,:] = Rotation(theta[s,1], theta[s,2], theta[s,3])
            end
        end
        if p == 2
            if theta_sim[1] == 0
                if(theta_true[s,1]<0)
                    theta_true[s,1] += 2pi
                end
                theta[s,1] = theta_true[s,1]
                theta[s,2] = 0
                theta[s,3] = 0
            else
                a = tr(A)
                b = A[1,2]-A[2,1]
                rho = sqrt(a^2+b^2)
                gamma = atan(b/rho,a/rho)
                if(gamma < 0)
                    gamma += 2pi
                end
                theta[s,1] = rand(VonMises(gamma,rho))
                if(theta[s,1]<0)
                    theta[s,1] += 2*pi
                end
                R[s,:,:] = [cos(theta[s,1]) sin(theta[s,1]); -sin(theta[s,1]) cos(theta[s,1]) ]
            end
        end
    end
    return theta, R
end
    
function mcmc(I_max::Int64, burn_in::Int64, thin::Int64, d::Int64,K::Int64,p::Int64,N::Int64,z::Array{Float64},Z::Array{Float64},Y::Array{Float64},nu::Int64,Psi::Array{Float64},M_prior::Array{Float64},V_prior::Array{Float64}, original::Int64 = 0,samples::Array{Float64}=zeros(N,K,p), B_true::Union{Array{Float64},Nothing} = nothing, Sigma_true::Union{Array{Float64},Nothing} = nothing, theta_true::Union{Array,Nothing} = nothing, theta_sim::Union{Array,Nothing} = nothing, beta_sim::Int64 = 0, Sigma_sim::Int64 = 0 )

    B,M,V,Sigma_est,theta,R,X = mcmc_setup(I_max,burn_in,thin,d,K,p,N,M_prior,V_prior);

    for i = 2:I_max
        if original == 0
            X[i,:,:,:] = sample_update!(X[i,:,:,:],R[i-1,:,:,:],Y)
        elseif original == 1
            X[i,:,:,:] = samples
        
        end

        if beta_sim == 1
            B[i,:,:,:]=mcmc_B!(N,p,K,d,Sigma_est[i-1,:,:],Z,V,M,X[i,:,:,:]);
        else
            B[i,:,:,:] = B_true
        end

        if Sigma_sim == 1
            Sigma_est[i,:,:]=mcmc_Sigma!(N,K,p,nu_prior,Psi_prior,X[i,:,:,:],Z,B[i,:,:,:]);
        else
            Sigma_est[i,:,:]= Sigma_true
        end
        
        if original == 0 && isnothing(theta_true)
            theta[i,:,:], R[i,:,:,:] = mcmc_theta!(N,B[i,:,:,:],z,Sigma_est[i,:,:],theta[i-1,:,:]);
        elseif !isnothing(theta_true)
            theta[i,:,:], R[i,:,:,:] = mcmc_theta!(N,B[i,:,:,:],z,Sigma_est[i,:,:],theta[i-1,:,:], theta_true, theta_sim)
        end
        
        if i%1000 == 0
            print("Iteration counter: ",i, '\n')
        end
    end

    return B[burn_in:thin:end,:,:,:], Sigma_est[burn_in:thin:end,:,:], theta[burn_in:thin:end,:,:], R[burn_in:thin:end,:,:,:], X[burn_in:thin:end,:,:,:]
end

function plot_mcmc(B::Array{Float64},Sigma::Array{Float64},B_true::Array{Float64},Sigma_true::Array{Float64},R::Array{Float64},R_true::Array{Float64},theta::Array{Float64},theta_true::Array{Float64},plot_flag::Array{Int64},dir::String="Plots/")
    if isdir(dir) == false
        mkdir(dir)
    else
        rm(dir,recursive=true)
        mkdir(dir)
    end
    I = size(B)[1]
    d = size(B)[2]
    K = size(B)[3]
    p = size(B)[4]

    B_flag = plot_flag[1]
    Sigma_flag = plot_flag[2]
    R_flag = plot_flag[3]
    theta_flag = plot_flag[4]
    if B_flag == 1 || Sigma_flag == 1
        print("Plotting B and Sigma...")
    end
    p_B = Array{Plots.Plot{Plots.GRBackend},1}()
    p_S = Array{Plots.Plot{Plots.GRBackend},1}()


    if B_flag == 1
            if isdir(dir*"Beta/") == false
                mkdir(dir*"Beta/")
            end
            lab = reshape(repeat(["sample";"true"],K*p),1,2*K*p)
            name_B = reshape(["B"*"_"*string(h)*"_"*string(i)*"_"*string(j) for h = 1:d for i =1:K for j = 1:p],1,d*K*p)
            for h = 1:d
                for i = 1:K
                    for j = 1:p
                        m = minimum( [minimum(B[:,h,i,j]) B_true[1,h,i,j] ])-1
                        M = maximum( [maximum(B[:,h,i,j]) B_true[1,h,i,j] ])+1
                        p1 = hline!(plot(B[:,h,i,j], size = (1920,1080),legend = true,xlims=(0,I), ylims = (m,M)),[B_true[1,h,i,j]])
                        push!(p_B, p1)
                    end
                end
                p_B_plot = plot(p_B[(h-1)*K*p+1:h*K*p]..., layout = (K,p), title = reshape(name_B[(h-1)*K*p+1:h*K*p],1,K*p), labels = lab)
                savefig(p_B_plot,dir*"Beta/Beta_"*string(h)*".png")
            end
    end
    
    if Sigma_flag == 1
        for i = 1:K
            for j = 1:K
                m = minimum(Sigma[:,i,j])-1
                M = maximum(Sigma[:,i,j])+1
                p2 = hline!(plot(Sigma[:,i,j], size = (1920,1080), legend = true,xlims = (0,I)),[Sigma_true[i,j]])
        push!(p_S, p2)
            end
        end
        name_S = reshape(["S"*"_"*string(i)*"_"*string(j) for i =1:K for j = 1:K],1,K*K)
        lab = reshape(repeat(["sample";"true"],K*K),1,2*K*K)
        p_S = plot(p_S..., layout = K*K, title = name_S, labels = lab)
        savefig(p_S,dir*"Sigma.png")
    end

    if B_flag == 1 || Sigma_flag == 1
        print("Done! \n")
    end

    if theta_flag == 1
        print("Plotting angles...")
        plot_angles(theta,theta_true,dir)
        print("Done! \n")
    end

    if R_flag == 1
        print("Plotting R...")
        plot_R(R,R_true,dir)
        print("Done!")
    end

end

function plot_R(R::Array{Float64},R_true::Array{Float64},dir::String="Plots/")

    I = size(R)[1]
    N = size(R)[2]
    p = size(R)[3]
    if isdir(dir*"R/") == false
        mkdir(dir*"R/")
    end

    p_R = Array{Plots.Plot{Plots.GRBackend},1}()
    for s =1:N
        for i = 1:p
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
    name_R = reshape(["R"*"_"*string(s)*"_"*string(i)*"_"*string(j) for s =1:N for i =1:p for j = 1:p],1,N_p)
    for k = 0:N-1
        p_S = p_R[p*p*k+1:p*p*k+p*p]
        title_S = reshape(name_R[p*p*k+1:p*p*k+p*p],1,p*p)
        labels_S = reshape(lab[2*p*p*k+1: 2*p*p+2*p*p*k],1,2*p*p)
        p_S1 = plot(p_S..., layout = p*p, title = title_S, labels = labels_S)
        savefig(p_S1,dir*"R/R_"*string(k+1)*".png")
        print("Plot "*string(k+1)*" finished! \n")
    end

end

function plot_angles(theta::Array{Float64},theta_true::Array{Float64},dir::String = "Plots/")

    I = size(theta)[1]
    N = size(theta)[2]
    p = size(theta)[3]
    if isdir(dir*"Theta/") == false
        mkdir(dir*"Theta/")
    end
    p_R = Array{Plots.Plot{Plots.GRBackend},1}()
    for s =1:N
        for i = 1:p
            if i != 2
                p1 = hline!(plot(theta[:,s,i], size = (1920,1080),legend = true,ylims=(0,2pi),xlims=(0,I)),[theta_true[s,i]])
            else
                p1 = hline!(plot(theta[:,s,i], size = (1920,1080),legend = true,ylims=(0,pi), xlims = (0,I)),[theta_true[s,i]])
            end
            push!(p_R, p1)
        end
    end

    N_p = N*p
    lab = reshape(repeat(["sample";"true"],N_p),1,2*N_p)
    name_R = reshape(["theta"*"_"*string(s)*"_"*string(i) for s =1:N for i =1:3],1,N_p)
    for k = 0:N-1
        p_S = p_R[3k+1:3k+3]
        title_S = reshape(name_R[3k+1:3k+3],1,p)
        labels_S = reshape(lab[6k+1: 6+6k],1,2*p)
        p_S1 = plot(p_S..., layout = 3, title = title_S, labels = labels_S)
        savefig(p_S1,dir*"Theta/theta_"*string(k+1)*".png")
        println("Plot"*string(k+1)*" finished!")
    end

end

function identify(B::Array{Float64})
    dims = size(B)
    I_max = dims[1]
    d = dims[2]
    K = dims[3]
    p = dims[4]
    B_identified = zeros(I_max,d,K,p)
    for i = 1:I_max
        for h=1:d
            B_identified[i,h,:,:] = GS(B[i,h,:,:])
        end
    end
    return B_identified
end

function angles(R::Array{Float64})
        p = size(R)[2]
        if p == 3
            theta2 = acos(R[3,3])
            theta3 = atan(R[1,3]/sin(theta2),R[2,3]/sin(theta2))
            theta1 = atan(R[3,1]/sin(theta2), -R[3,2]/sin(theta2))
        elseif p == 2
            theta1 = atan(R[1,2],R[1,1])
            theta2 = 0
            theta3 = 0
        end
    return [theta1,theta2,theta3]
end

function identify_t!(theta::Array{Float64})
    
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

function identify_angles(theta::Array{Float64})
    N = size(theta)[1]
    for i = 1:N
        theta[i,:] = identify_t!(theta[i,:])
    end
    return theta
end

function identify_samples(theta::Array{Float64})
    N = size(theta)[1]
    for i = 1:N
        theta[i,:,:] = identify_angles(theta[i,:,:])
    end
    return theta
end

function verify_svd(X::Array{Float64},Y::Array{Float64},theta::Array{Float64})
    R = Rotation(theta[1],theta[2],theta[3])
    return X-Y*R'
end

function a_mises(rho::Float64,gamma::Float64,x::Float64)
    return exp(rho*cos(x-gamma))/(2pi*besseli(0,x))*sin(x)
end

function get_V(B::Array{Float64})
    #Funzione che restituisce una versioen identificata ddella mtrice di rotazione considerata
    #Si tratta di una semplice ortogonalizzazione di Gram-Schmidt
    p = size(B)[2]
    A = B'
    V = zeros(p,p);
    V[:,1] = A[:,1]/norm(A[:,1])
    V[:,2] = A[:,2] - (A[:,2]'*V[:,1])*V[:,1]
    V[:,2] = V[:,2]/norm(V[:,2])
    if p == 3 
        V[:,3] = A[:,3] - ((A[:,3]'*V[:,1])*V[:,1]) - ((A[:,3]'*V[:,2])*V[:,2])
        V[:,3] = V[:,3]/norm(V[:,3])
        #Scelgo l'ultima colonna in modo da avere una amtric ein SO(3)
        if (det(V) < 0)
            V[:,3] = -V[:,3]
        end
    elseif p == 2
        if(det(V) < 0)
            V[:,2] = -V[2,:]
        end
    end
    return V
end

function identify_R_angles(B::Array{Float64},R::Array{Float64})
    R_new = copy(R)
    dim = size(R)
    N = dim[1]
    S = dim[2]
    K = size(B)[3]
    p = dim[3]
    thetas = zeros((N,S,3))
    for i = 1:N
        for s = 1:S
            R_new[i,s,:,:] = R[i,s,:,:]*get_V(reshape(B[i,1,:,:],K,p))
            thetas[i,s,:] = identify_t!(angles(copy(R_new[i,s,:,:])))
        end
    end
    return R_new, thetas
end

function identify_R(B::Array{Float64},R::Array{Float64})
    R_new = copy(R)
    dim = size(R)
    N = dim[1]
    S = dim[2]
    K = dim[3]
    p = dim[4]
    for i = 1:N
        G = get_V(reshape(B[i,1, :,:],K,p))
        for s = 1:S
            R_new[i,s,:,:] = R[i,s,:,:]*G
        end
    end
    return R_new
end

function identify_R_true(B_true::Array{Float64},R_true::Array{Float64})
    dim = size(R_true)
    S = dim[1]
    K = dim[2]
    p = dim[3]

    R_true_new = copy(R_true)
    G = get_V(reshape(B_true[1,1,:,:],K,p))
    for s = 1:S
        R_true_new[s,:,:] = R_true[s,:,:]*G
    end
    return R_true_new
end

function identify_R_angles_true(B_true::Array{Float64},R_true::Array{Float64})
    dim = size(R_true)
    S = dim[1]
    K = dim[2]
    p = dim[3]

    R_true_new = copy(R_true)
    thetas = zeros((S,3))
    G = get_V(B_true[1,1,:,:])
    for s = 1:S
        R_true_new[s,:,:] = R_true[s,:,:]*G
        thetas[s,:] = identify_t!(angles(R_true_new[s,:,:]))
    end
    return R_true_new, thetas
end

function grid_mcmc(T1::Vector{Int64},T2::Vector{Int64},T3::Vector{Int64},B_v::Vector{Int64},S_v::Vector{Int64},I_max::Int64, burn_in::Int64, thin::Int64, d::Int64,K::Int64,p::Int64,N::Int64,Z::Array{Float64},Y::Array{Float64},nu_prior::Int64,Psi_prior::Array{Float64},M_prior::Array{Float64},V_prior::Array{Float64}, original::Int, samples::Array{Float64},B_true::Array{Float64}, Sigma_true::Array{Float64},theta_true::Array{Float64}, R_true::Array{Float64})
    if isdir("Plots/") == false
        mkdir("Plots/")
    else
        rm("Plots/",recursive=true)
        mkdir("Plots/")
    end
    for t1 in T1
        if t1 == 1
            dir1 = "Theta1_"
        else 
            dir1 = ""
        end
        for t2 in T2
            if t2 == 1
                dir2 = "Theta2_"
            else 
                dir2 = ""
            end
            for t3 in T3
                if t3 == 1
                    dir3 = "Theta3_"
                else 
                    dir3 = ""
                end
                for b in B_v
                    if b == 1
                        dirb = "Beta_"
                    else 
                        dirb = ""
                    end
                    for s in S_v
                        if s == 1
                            dirs = "Sigma_"
                        else 
                            dirs = ""
                        end
                        dir_m = dirb*dirs*dir1*dir2*dir3
                        if dir_m == ""
                            dir = "Plots/Original/"
                        else
                            if last(dir_m) == '_'
                                dir_m = chop(dir_m)
                            end
                            dir = "Plots/"*dir_m*"/"
                        end
                        if isdir(dir) == false
                            mkdir(dir)
                        end
                        theta_sim = [t1 t2 t3]
                        beta_sim = b 
                        Sigma_sim = s
                        plot_flag = [1 1 1 1]
                        B, Sigma_est, theta, R, X = mcmc(I_max, burn_in, thin, d,K,p,N,z,Z,Y,nu_prior,Psi_prior,M_prior,V_prior, original, samples,B_true, Sigma_true, theta_true,theta_sim,beta_sim,Sigma_sim);
                        B_true_tensor = permutedims(reshape(B_true,d,K,p,1),(4,1,2,3))
                        samples_id,X_id, B_id, B_true_id, R_id, R_true_id, theta_id, theta_true_id =identify_params(Y,B, B_true_tensor, Sigma_est, Sigma_true, R, R_true)
                        plot_mcmc(B_id,Sigma_est,B_true_id,Sigma_true,R_id,R_true_id,theta_id,theta_true_id,plot_flag,dir)
                    end
                end
            end
        end
    end

    
end

function plot_data(D::Array{Float64})

    N,K,p = size(D)
    if p == 3
        X = D[1,:,1]
        Y = D[1,:,2]
        Z = D[1,:,3]

        p1 = scatter(X,Y,Z)

        for i=2:N
            X = D[i,:,1]
            Y = D[i,:,2]
            Z = D[i,:,3]
            scatter!(p1,X,Y,Z)
        end
    elseif p == 2
        X = D[1,:,1]
        Y = D[1,:,2]
        p1 = scatter(X,Y)
        for i=2:N
            X = D[i,:,1]
            Y = D[i,:,2]
            scatter!(p1,X,Y)
        end
    end
    return p1
end

function compare(X::Array{Float64},samples::Array{Float64})
    dim = size(X)
    N = dim[2]
    K = dim[3]
    p = dim[4]
    data = reshape(mean(X,dims = 1),N,K,p)
    p1 = plot_data(data)
    p2 = plot_data(samples)

    p = [p1,p2]

    plot(p...,layout = 2,titles = reshape(["Simulated","Real"],1,2),legend = false)
end

function identify_params(Y::Array{Float64},B::Array{Float64}, B_true::Array{Float64}, Sigma::Array{Float64},Sigma_true::Array{Float64}, R::Array{Float64},R_true::Array{Float64})
    
    I_max,K,p = size(B)[[1,3,4]]
    N = size(Y)[1]
    B_id = identify(B)
    B_true_id = identify(B_true)
    R_id, theta_id = identify_R_angles(B,R)
    R_true_id,theta_true_id = identify_R_angles_true(B_true,R_true)

    X_id = zeros(I_max,N,K,p)
    for i = 2:I_max
        X_id[i,:,:,:] = sample_update!(X_id[i,:,:,:],R_id[i-1,:,:,:],Y)
    end

    samples_id = zeros(N,K,p)
    samples_id = sample_update!(samples_id,R_true_id,Y)
    return samples_id, X_id, B_id, B_true_id, R_id, R_true_id, theta_id, theta_true_id
end