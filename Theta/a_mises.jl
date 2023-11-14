using SpecialFunctions
using Plots
using LinearAlgebra
using NumericalIntegration
using Distributions

function a_mises(rho,gamma,x)
    return exp(rho*cos(x-gamma))*sin(x)
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
        if(sgamma > 0)
            gamma = acos(cgamma)
        else
            gamma = 2pi-acos(cgamma)
        end
        #gamma = atan(sgamma,cgamma)
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
            if y <0
                y += 2pi
            end
            return y
        i = i+1
        end
    
    end
end

N = 100000
x = collect(LinRange(0,pi,N));
#x = collect(LinRange(-1,1,N));
y = zeros(N,1);
z = zeros(N,1)

B = ones(3,3)+I(3)
B[3,2] += 3
a = B[2,2]+B[3,3]
b = B[3,2]-B[2,3]
rho = sqrt(a^2+b^2)
gamma = acos(a/rho)

for i =1:N
    z[i] = sample(B,2)
    y[i] = a_mises(rho,gamma,x[i])
    #y[i] = acos(x[i])
end

c = integrate(x,y)

histogram(z,normalize=true)
plot!(x,(/).(y,c))