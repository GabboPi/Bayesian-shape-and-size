using LinearAlgebra
using Distributions
using Random
using Statistics
using Plots
include("mcmc.jl")

function matrix_fisher(R::Array{Float64},A::Array{Float64})
    return exp(tr(RA'))
end

function R_metropolis(R::Array{Float64}, A::Array{Float64})
    sample = copy(R)
    theta_last = angles(R)

    p = size(R)[1]

    if p ==2

    elseif p ==3
        flag = 0

        accepted = 0
        rejected = 0
        while flag == 0 & rejected < 1000
            theta1 = rand(Normal(theta_last[1],0.1))
            theta2 = rand(Normal(theta_last[2],0.1))
            theta3 = rand(Normal(theta_last[3],0.1))

            while theta1 < 0
                theta1 += 2pi
            end
            theta1 %= 2pi
    
            while theta2 < 0
                theta2 += pi
            end
            theta2 %= pi
    
            while theta3 < 0
                theta3 += 2pi
            end
            theta3 %= 2pi

            R_new = Rotation(theta1,theta2,theta3)
            ratio = tr(R_new*A') - tr(R*A')

            u = rand(Uniform(0,1))
            if log(u) < ratio
                accepted += 1
                sample = R_new
                flag = 1
            else
                rejected +=1
            end
        end
    end
    theta_sample = angles(sample)
    #print("Acceptance ratio: ",string(accepted/(accepted+rejected),"\n"))
    return theta_sample, sample
end


function mcmc_R!(N::Int64,B::Array{Float64},z::Array{Float64},Sigma::Array{Float64},theta_last::Array{Float64}, theta_true::Union{Array{Float64},Nothing}=nothing, theta_sim::Union{Array{Int64},Nothing}=nothing)
    d = size(B)[1]
    K = size(B)[2]
    p = size(B)[3]
    theta = zeros(N,3)
    R = zeros(N,p,p)
    I_Sigma = inv(Sigma[:,:])
    for s = 1:N

        m = zeros(K,p)
        for h = 1:d
            m += z[s,h]*B[h,:,:]
        end

        A = m'*I_Sigma*Y[s,:,:]

        if p == 3
            R_last = Rotation(theta_last[1],theta_last[2],theta_last[3])
            theta[s,:], R[s,:,:] = R_metropolis(R_last,A)
        end
        if p == 2
            if theta_sim[1] == 0
                if(theta_true[s,1]<0)
                    theta_true[s,1] += 2pi
                end
                theta[s,1] = theta_true[s,1]
                theta[s,2] = 0
                theta[s,3] = 0
                R[s,:,:] = [cos(theta[s,1]) sin(theta[s,1]); -sin(theta[s,1]) cos(theta[s,1]) ]
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

function mcmc_theta_new!(N::Int64,B::Array{Float64},z::Array{Float64},Sigma::Array{Float64},theta_last::Array{Float64}, theta_true::Union{Array{Float64},Nothing}=nothing, theta_sim::Union{Array{Int64},Nothing}=nothing)
    d = size(B)[1]
    K = size(B)[2]
    p = size(B)[3]
    theta = zeros(N,3)
    R = zeros(N,p,p)
    I_Sigma = inv(Sigma[:,:])
    for s = 1:N

        m = zeros(K,p)
        for h = 1:d
            m += z[s,h]*B[h,:,:]
        end

        A = m'*I_Sigma*Y[s,:,:]

        theta1 = theta_last[s,1]

        if p == 2
            theta2 = 0
            theta3 = 0
        elseif p == 3
            theta2 = theta_last[s,2]
            theta3 = theta_last[s,3]
        end

        #Matrici di rotazione associate ai 3 angoli di Eulero, con convenzione ZXZ

        if p == 3
            if isnothing(theta_true)
                #Campiono theta_1
                R1, R2, R3 = make_rotations(theta1,theta2,theta3)
                H = A'*R3*R2
                theta[s,1] = sample(H,1)
                #Campiono theta_2
                R1, R2, R3 = make_rotations(theta[s,1],theta2,theta3)
                D = R1*A'*R3
                theta[s,2] = sample(D,2)
                #Campiono theta_3
                R1, R2, R3 = make_rotations(theta[s,1],theta[s,2],theta3)
                L = R2*R1*A'
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
                    R1, R2, R3 = make_rotations(theta1,theta2,theta3)
                    H = A'*R3*R2
                    theta[s,1] = sample(H,1)
                end
                
                if theta_sim[2] == 0
                    #Controllo che gli angoli siano tra [0,pi]
                    if(theta_true[s,2]<0)
                        theta_true[s,2] += pi
                    end
                    theta[s,2] = theta_true[s,2]
                else
                    R1, R2, R3 = make_rotations(theta[s,1],theta2,theta3)
                    D = R1*A'*R3
                    theta[s,2] = sample(D,2)
                end

                if theta_sim[3] == 0
                    #Controllo che gli angoli siano tra [0,2pi]
                    if(theta_true[s,3]<0)
                        theta_true[s,3] += 2pi
                    end
                    theta[s,3] = theta_true[s,3]
                else
                    R1, R2, R3 = make_rotations(theta[s,1],theta[s,2],theta3)
                    L = R2*R1*A'
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
                R[s,:,:] = [cos(theta[s,1]) sin(theta[s,1]); -sin(theta[s,1]) cos(theta[s,1]) ]
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

function make_rotations(theta1::Float64,theta2::Float64,theta3::Float64)
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

    return R1, R2, R3
end