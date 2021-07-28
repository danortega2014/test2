# test2
```
cd("C:\\Users\\danor\\Desktop\\")



using POMDPs,POMDPModelTools
using Random, Distributions 
using CuArrays


Random.seed!(1234)

Base.@kwdef mutable struct firmgame <: MDP{Tuple{Float64,Float64,Float64,Float64,Float64,Float64, Float64, Float64, Int64}, Symbol} # POMDP{State, Action, Observation
    discount_factor::Float64 = 0.98 # discount
    terminaly::Bool = 0 
end 

#observations are nomprice, wealth, debt, capital, interest rate,labor9
m = firmgame()

# actions  np2 to p2 represent what the agent expects the change in demand to be in the price, selling less when he expects low demand, and vice versa.

#action set available given state 


function POMDPs.actions(m::firmgame, s::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}) 
    nomprice, wealth, debt, capital, interest, labor, inventory, consumption, timestep  = s
    a1 = nothing
    a2 = nothing 
    a3 = nothing 
    a4 = nothing 
    a5 = nothing
    a6 = nothing
    a7 = nothing 
    a8 = nothing 
    a9 = nothing 
    a10 = nothing 
    a11 = nothing
    a12 = nothing 
    a13 = nothing 
    a14 = nothing 
    a15 = nothing  
    while inventory >= 1.0 
        a1 =:np2 
    end 
    
    while inventory >= 2.0
        a2 =:np1 
    end

    while inventory >= 3.0
        a3 =:p0 
    end

    while inventory >= 4.0
        a4 =:p1 
    end 

    while inventory >= 5.0
        a5 =:p2
    end

    while  wealth >= (50.0 + debt*interest/6.0)
        a6 = :smallinvest 
    end

    while wealth >= (100.0 + debt*interest/6.0) 
        a7 == :largeinvest
    end

    while wealth >= 0.0 && debt <= 1000
        a8 = :smallborrow 
        a9 = :largeborrow 
    end


    while debt >=  50 && wealth >= (50.0 + debt*interest/6.0)
        a10 = :smallpayback
    end

    while debt >= 100.0  && wealth >= (100.0 + debt*interest/6.0)
        a11 = :largepayback 
    end

    while labor >= 12.0  
        a12 =:smallcraft 
    end


    while labor >= 24.0  
        a13 =:largecraft 
    end


    while wealth >= (50.0+.10*wealth) + debt*(1.0+interest/6.0)
        a14 =:smallconsume
    end

    while wealth >= (100.0 +.2*wealth) + debt*(1.0+interest/6.0)
        a15 =:largeconsume 
    end

    m = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15]
    return filter(!isnothing, m)
end


z = 0.0  
timestep = 0 
function POMDPs.gen(m::firmgame, s::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64, Float64, Int64}, a::Symbol, rng)
    global z_t =  round(rand(Normal(0.0, .7))) #-2:2
    global inflationshock =  round(rand(Normal(0.0, .5))) #-1:1
    #round function causes creation of -0.0, which has to be corrected for later

    nomprice, wealth, debt, capital, interest, labor, inventory, consumption, timestep  = s
    
    interest_ = round(max(0.0, interest - (inflationshock/130.0) + (debt/150000.0)), digits = 8) #interest rate function: higher inflation, lower interest rate & higher debt/higher interest rate 
    
 #if agent fails to do an action within his action space, restart sim.   

    global timestep += 1  #step counter
    global y = 3.0 + z_t  #demand function 0:4
    m.terminaly = 0
  

    #transition function: wealth, debt, capital depends on action
    if a==:np2 && inventory >= 1.0 
        Zstar = -2.0
        Qstar = Zstar + 3.0#how much agent is attempting to sell based on Zstar prediction
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)#current price based on previous price, demand shock, and inflation shock.
        debt_ = max(0.0, debt*(1+interest/6.0))
        Q = min(y, Qstar) # actual quantity supplied function: based on principle you can't sell more than demand only <=
        wealth_ = max(0.0, (wealth + nomprice*Q - debt*interest/6.0)*(1.0 + interest/12.0)) #agent sells Q 
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory - Q) #agent loses Q
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption

        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 
    
    elseif a==:np1 && inventory >= 2.0
        Zstar = -1.0
        Qstar = Zstar + 3.0#how much agent is attempting to sell based on Zstar prediction
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)#current price based on previous price, demand shock, and inflation shock.
        debt_ = max(0.0, debt*(1+interest/6.0))
        Q = min(y, Qstar) # actual quantity supplied function: based on principle you can't sell more than demand only <=
        wealth_ = max(0.0, (wealth + nomprice*Q - debt*interest/6.0)*(1.0 + interest/12.0)) #agent sells Q 
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory - Q) #agent loses Q
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption

        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 
   
    elseif a==:p0 && inventory >= 3.0
        Zstar = 0.0
        Qstar = Zstar + 3.0#how much agent is attempting to sell based on Zstar prediction
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)#current price based on previous price, demand shock, and inflation shock.
        debt_ = max(0.0, debt*(1+interest/6.0))
        Q = min(y, Qstar) # actual quantity supplied function: based on principle you can't sell more than demand only <=
        wealth_ = max(0.0, (wealth + nomprice*Q - debt*interest/6.0)*(1.0 + interest/12.0)) #agent sells Q 
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory - Q) #agent loses Q
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption

        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a==:p1 && inventory >= 4.0
        Zstar = 1.0
        Qstar = Zstar + 3.0#how much agent is attempting to sell based on Zstar prediction
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)#current price based on previous price, demand shock, and inflation shock.
        debt_ = max(0.0, debt*(1+interest/6.0))
        Q = min(y, Qstar) # actual quantity supplied function: based on principle you can't sell more than demand only <=
        wealth_ = max(0.0, (wealth + nomprice*Q - debt*interest/6.0)*(1.0 + interest/12.0)) #agent sells Q 
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory - Q) #agent loses Q
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption
        
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a==:p2 && inventory >= 5.0
        Zstar = 2.0
        Qstar = Zstar + 3.0#how much agent is attempting to sell based on Zstar prediction
        nomprice_ = nomprice + z_t + inflationshock #current price based on previous price, demand shock, and inflation shock.
        debt_ = debt*(1+interest/6.0) 
        Q = min(y, Qstar) # actual quantity supplied function: based on principle you can't sell more than demand only <=
        wealth_ = (wealth + nomprice*Q - debt*interest/6.0)*(1.0 + interest/12.0) #agent sells Q 
        capital_ = capital*.95 
        inventory_ = inventory - Q #agent loses Q
        labor_ = labor + 3.0
        consumption_ = consumption 

        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a== :smallinvest && wealth >= (50.0 + debt*interest/6.0)
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)#current price based on previous price, demand shock, and inflation shock.
        debt_ = max(0.0, debt*(1+interest/6.0)) # actual quantity supplied function: based on principle you can't sell more than demand only <=
        wealth_ = max(0.0, (wealth  - debt*interest/6.0 - 50)*(1.0 + interest/12.0)) #agent sells Q 
        capital_ = max(0.0, capital*.95 + 5.0) 
        inventory_ = max(0.0, inventory) #agent loses Q
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption
        
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a== :largeinvest && wealth >= (100.0 + debt*interest/6.0) 
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0))
        wealth_ = max(0.0, (wealth - debt*interest/6.0 - 100)*(1.0 + interest/12.0))
        capital_ = max(0.0, capital*.95 + 10.0) 
        inventory_ = max(0.0, inventory)
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption
         
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a== :smallborrow && wealth >= 0.0
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0)+50)
        wealth_ = max(0.0, (wealth - debt*interest/6.0 + 100)*(1.0 + interest/12.0))
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory)
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption
        
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a== :largeborrow && wealth >= 0.0
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0)+100)
        wealth_ = max(0.0, (wealth - debt*interest/6.0 + 100)*(1.0 + interest/12.0))
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory)
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption
        

        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a == :smallpayback && debt >=50 && wealth >= (50.0 + debt*interest/6.0)
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0)- 50)
        wealth_ = max(0.0, (wealth- debt*interest/6.0 - 50*(1.0 + interest/12.0)))
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory)
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption
        
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a == :largepayback && debt >= 100.0  && wealth >= (100.0 + debt*interest/6.0)
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0)-100)
        wealth_ = max(0.0, (wealth- debt*interest/6.0 - 00)*(1.0 + interest/12.0)) 
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory)
        labor_ = min(24, labor + 3.0)
        consumption_ = consumption
        

        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a==:smallcraft && labor >= 12.0  
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)#current price based on previous price, demand shock, and inflation shock.
        debt_ = max(0.0, debt*(1+interest/6.0))
        wealth_ = max(0.0, (wealth - debt*interest/6.0)*(1.0 + interest/12.0)) #agent sells Q 
        capital_ = max(0.0, capital*.95) 
        prod = ((12)^.5)*(capital)^.5
        inventory_ = max(0.0, inventory + prod) 
        labor_ = min(24, labor + 3.0 - 12.0)
        consumption_ = consumption
        
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a==:largecraft && labor >= 24.0  
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0))
        wealth_ = max(0.0, (wealth - debt*interest/6.0)*(1.0 + interest/12.0)) #agent sells Q 
        capital_ = max(0.0, capital*.95) 
        prod = ((24)^.5)*(capital)^.5
        inventory_ = max(0.0, inventory + prod) #agent loses Q
        labor_ = min(24, labor + 3.0 - 24.0)
        consumption_ = consumption
        
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a ==:smallconsume && wealth >= (50.0+.10*wealth) + debt*(1.0+interest/6.0)
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0))
        consumption_ = consumption + 50.0 + .10*wealth
        wealth_ = max(0.0, (wealth - debt*interest/6.0 - (consumption_ - consumption))*(1.0 + interest/12.0)) 
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory) 
        labor_ = min(24, labor + 3.0)
     
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    elseif a ==:largeconsume && wealth >= (100.0 +.2*wealth) + debt*(1.0+interest/6.0)
        nomprice_ = max(0.0, nomprice + z_t + inflationshock)
        debt_ = max(0.0, debt*(1+interest/6.0))
        consumption_ = consumption + 100.0 + .20*wealth
        wealth_ = max(0.0, (wealth - debt*interest/6.0 - (consumption_ - consumption))*(1.0 + interest/12.0)) 
        capital_ = max(0.0, capital*.95) 
        inventory_ = max(0.0, inventory) 
        labor_ = min(24, labor + 3.0)
        
        sp = nomprice_, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    else #action outside of actionset
        m.terminaly = 1
        nomprice_ = 30.0
        wealth_ = 100.0
        debt_ = 0.0
        capital_ = 5.0
        inventory_ = 4.0
        labor_ = 12.0
        consumption_ = 0.0 
        timestep = 1

        sp = nomprice, wealth_, debt_, capital_, interest_, labor_, inventory_, consumption_, timestep 

    end
    
    if wealth_ == 0 
        r = -5000 - (debt)^2 
    elseif m.terminaly == 0 
            r = (consumption_ - consumption)^2 + (wealth_ - wealth) + (labor_ - labor)^3
    else
        r = -10000
    end

    # create and return a NamedTupleSE
    return (sp=sp, r=r)
end

POMDPs.isterminal(m::firmgame, s::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64, Float64, Int64}) = m.terminaly == 1 || s[9] == 100 || s[2] == 0.0 

POMDPs.discount(pomdp::firmgame) = pomdp.discount_factor
POMDPs.initialstate(pomdp::firmgame) = Deterministic((30.0, 100.0, 0.0, 5.0, 0.02, 12.0, 4.0, 0.0, 1))  #initialprice_, wealth_, debt_, capital_, interest_, labor, inventory, consumption, step


function POMDPs.convert_s(::Type{A}, s::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64, Int64}, m::firmgame) where A<:AbstractArray
    s = convert(A, [s...])
    return s
end


env = firmgame()


using DeepQLearning
using Flux
using POMDPPolicies
using CuArrays #load if you want to send weights to gpu in solve 
using BSON: @save, @load


# @load "qnetwork.bson" model_27
using Knet


qn2 = Chain(LSTM(9, 45), LSTM(45, 30), Dense(30, 15), Knet.softmax)

#Flux.loadparams!(qn2, map(p -> p .= randn.(), Flux.params(qn2)))
# @save "mymodely.bson" weights

stepsx = 90000
exploration2 = SoftmaxPolicy(m, LinearDecaySchedule(start = 16.0, stop = .001, steps = stepsx))
# exploration2 = EpsGreedyPolicy(env, LinearDecaySchedule(start=1.0, stop=0.1, steps=.9*stepsx))
solvrz= DeepQLearningSolver(qnetwork = qn2, max_steps= stepsx, target_update_freq = 350,  batch_size = 75, train_freq = 4, 
                             exploration_policy = exploration2,
                             learning_rate= .005,log_freq=500, eval_freq = 25 ,  num_ep_eval = 5, 
                             recurrence=true, double_q=true, dueling=false, prioritized_replay=false, prioritized_replay_alpha= 1, prioritized_replay_epsilon = .0000001,
                             prioritized_replay_beta = .4 ,  buffer_size = 700,  max_episode_length= 100, train_start = 100 ,save_freq = stepsx/8) #logdir = "C:\\Users\\danor\\Desktop\\models\\model"
policyzx = solve(solvrz,env)

#Flux.reset!(qnet17)
#resetstate!(policyzx)
#Flux.params(qnet17)#

using POMDPSimulators 
rsum = 0.0 
for (s,a,r) in stepthrough(env, policyzx, "s,a,r", max_steps=100 )
    println("$z_t, $inflationshock, $s, $a, $r")
    global rsum += r
end  
println("Undiscounted reward was $rsum.") 
```
