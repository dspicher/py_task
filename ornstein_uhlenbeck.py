import json
import warnings
from collections import OrderedDict

import brian2
import numpy as np
from brian2 import *
from IPython import embed
from scipy.stats import bernoulli

from helper import do, dump

assert(brian2.__version__ == '2.0')

#prefs.codegen.target = "weave"
#warnings.simplefilter(action = "ignore", category = FutureWarning)

@implementation("cpp", """
     #include<math.h>
     double phi(double U, double alpha, double beta, double r_max) {
        return r_max / (1 + exp(-beta*(U-alpha)));
     }
     """)
@check_units(U=volt, alpha=volt, beta=volt**-1, r_max=second**-1, result=second**-1)
def phi(U, alpha, beta, r_max):
    pass


@implementation("cpp", """
     #include<math.h>
     double phi_prime(double U, double alpha, double beta, double r_max) {
        double num = exp((U + alpha) * beta) * r_max * beta;
        double denom = pow(exp(U * beta) + exp(alpha * beta), 2.0);
        return num / denom;
     }
     """)
@check_units(U=volt, alpha=volt, beta=volt**-1, r_max=second**-1, result=(second * volt)**-1)
def phi_prime(U, alpha, beta, r_max):
    pass

def get_poisson_spikes(input_rate=10*Hz, cycle_dur=100*ms, N_input=100, use_indices=None):
    if use_indices == None:
        use_indices = range(N_input)
    spike_nrs = np.random.poisson(input_rate * cycle_dur, len(use_indices))
    indices = []
    times = np.array([])
    for i, nr in enumerate(spike_nrs):
        indices = indices + [use_indices[i] for _ in range(nr)]
        times = np.concatenate((times, cycle_dur / ms * np.random.rand(nr)))
    times = times * ms

    return indices, times

def get_chain_spikes(input_rate, cycle_dur=100*ms, N_input=100):
    indices = range(N_input)
    times = arange(0, cycle_dur / ms, cycle_dur / ms / N_input) * ms
    chain = SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)
    return SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)

def OU_estimation(dt=0.05, T=2500, sigma=0.01, mu=1, theta=0.01, S0=1, M=1):
    N = int(1.0*T/dt)
    S = np.zeros((M,N+1))
    S[:,0] = S0
    for n in range(N):
        #Partially vectorized
        eps = sqrt(dt)*np.random.normal(0, 1, (M,))
        S[:,n+1] = S[:,n] + theta*(mu - S[:,n])*dt + sigma*eps
    return S


def simulate((repetition_i, p)):
    neuron = json.load(open("default_neuron.json", "r"))
    group_size = p["n_group"]

    neuron["g_adap_delta"] = group_size*neuron["g_adap_delta"]
    np.random.seed(p.get("seed",42))  # seed for pseudo-randomness


    neuron["r_max"] = 0.1/group_size
    neuron["t_supp"] = p.get("t_supp", 0.0)

    ns = {}
    for key, value in neuron.items():
        if key[0] == "g":
            unit = msiemens
        elif key[0] == "E":
            unit = mV
        elif key[0] == "t":
            unit = ms
        elif key[0] == "I":
            unit = uamp
        elif key[:3] == "rho":
            unit = 1.0
        elif key == "alpha":
            unit = mV
        elif key == "beta":
            unit = mV**-1
        elif key == "r_max":
            unit = ms**-1
        else:
            raise Exception()
        ns[key] = value * unit


    ns["phi_prime"] = phi_prime
    ns["phi"] = phi
    defaultclock.dt = 0.05*ms

    dt_teacher = 1 * second
    ns["g_E"] = TimedArray(np.zeros(1) * msiemens, dt_teacher)
    ns["g_I"] = TimedArray(np.zeros(1) * msiemens, dt_teacher)

    ns["C"] = 1e-6 * farad

    f_in = 10*Hz
    N_input = p["N_input"]
    cycle = 500*ms

    N_presentations = 2
    durat = N_presentations*cycle

    # OU
    OUgLfactor = p["OUgLfactor"]
    OUgEmu = OUgLfactor*neuron['g_L']/15.0
    OUgImu = OUgLfactor*14.0*neuron['g_L']/15.0

    gOUE = OU_estimation(dt=defaultclock.dt/ms, T=durat/ms, sigma=8e-2*OUgEmu, mu=OUgEmu, theta=1e-1, S0=OUgEmu, M=2*group_size)
    gOUE[gOUE<=0] = 0.0
    gOUI = OU_estimation(dt=defaultclock.dt/ms, T=durat/ms, sigma=8e-2*OUgImu, mu=OUgImu, theta=1e-1, S0=OUgImu, M=2*group_size)
    gOUI[gOUI<=0] = 0.0

    """
    plot(gOUI[0,:]*neuron['E_I']/(gOUE[0,:]+gOUI[0,:]))
    figure()
    plot(gOUI[1,:]*neuron['E_I']/(gOUE[1,:]+gOUI[1,:]))
    show()
    """
    
    ns["gOUE"] = TimedArray(gOUE.T * msiemens, defaultclock.dt)
    ns["gOUI"] = TimedArray(gOUI.T * msiemens, defaultclock.dt)

    nrn_eqs = """
    #Teaching
    I_som = -g_E(t)*(U - E_E) - g_I(t)*(U - E_I) : amp
    """

    if p["Isp"]:
        nrn_eqs += """
        #Spike currents
        I_spike = - g_Na*(U - E_Na)*int((t-lastspike) < t_rise) - g_K*(U - E_K)*int((t-lastspike) > t_rise)*int((t-lastspike) < t_fall) : amp
        """
    else:
        nrn_eqs += """
        #Spike currents
        I_spike = 0.0*uamp : amp
        """

    nrn_eqs += """
    #Adaptation currents
    dg_adap/dt = -g_adap/tau_adap : siemens
    I_adap = -g_adap*(U - E_adap) : amp

    #Intra-group connections
    I_intra = -g_E_S_tot*(U - E_E) - g_E_S_tot*rho_I*(U - E_I) : amp

    #Inter-group connections
    I_inter = -g_I_S_tot*(U - E_I) - g_I_S_tot*rho_E*(U - E_E) : amp

    #Dendritic currents
    I_den = -g_D*(U - V) : amp

    #Leak current
    I_L = -g_L*(U - E_L) : amp

    #Somatic potential
    dU/dt = (I_L + I_den + I_intra + I_inter + I_adap + I_som + I_spike)/C : volt

    #Dendritic potential
    dV/dt = (-g_L*(V-E_L) - gOUE(t,i)*(V - E_E) - gOUI(t,i)*(V - E_I) - (g_E_D_tot + g_E_D_tot2)*(V - E_E) - g_S*(V-U))/C : volt

    #Dendritic prediction
    dV_star/dt = (-g_L*(V_star - E_L) - g_D*(V_star - V))/C : volt

    #sum of excitatory synaptic conductance onto dendrite
    g_E_D_tot : siemens

    #sum of excitatory synaptic conductance onto dendrite
    g_E_D_tot2 : siemens

    g_E_S_tot : siemens
    g_I_S_tot : siemens

    spiketrain = int((t-lastspike) <= 1.1*dt)/(dt/second) : 1
    """

    NoI = NeuronGroup(2*group_size, nrn_eqs, threshold="phi(U, alpha, beta, r_max)*dt>rand() or abs(t-5*ms)<=dt",
                    refractory=p["t_ref"]*ms, reset="g_adap += g_adap_delta/(tau_adap/ms)", method="euler")
    NoI.U = -70.0 * mV
    NoI.V = -70.0 * mV
    NoI.V_star = -70.0 * mV

    eqs_syn_som_exc = """
    #Input with weight
    dg_E_S/dt = -g_E_S/tau_exc_inh : siemens (clock-driven)

    g_E_S_tot_post = g_E_S : siemens (summed)

    w : 1
    """

    s_som_e = Synapses(NoI, NoI, eqs_syn_som_exc, on_pre="g_E_S += w/(tau_exc_inh/ms)*msiemens", method="euler")
    if group_size > 1:
        s_som_e.connect("i!=j and ((i < group_size) == (j < group_size))")
    else:
        s_som_e.connect("(i < group_size) == (j < group_size)")
    s_som_e.w = p["we"]

    eqs_syn_som_inh = """
    #Input with weight
    dg_I_S/dt = -g_I_S/tau_exc_inh : siemens (clock-driven)

    g_I_S_tot_post = g_I_S : siemens (summed)

    w : 1
    """

    s_som_i = Synapses(NoI, NoI, eqs_syn_som_inh, on_pre="g_I_S += w/(tau_exc_inh/ms)*msiemens", method="euler")
    s_som_i.connect("(i < group_size) != (j < group_size)")
    s_som_i.w = p["wi"]

    # equations for the synapses
    eqs_syn = """
    #Input without weight
    dg_s/dt = -g_s/tau_s : siemens (clock-driven)

    #Input with weight
    dg_E_D/dt = -g_E_D/tau_s : siemens (clock-driven)

    g_E_D_tot_post = g_E_D : siemens (summed)

    #Derivatives w.r.t the synaptic weight
    ddV_dw/dt = -(g_L + g_S + g_E_D)*dV_dw/C + g_S*dV_star_dw/C + (E_E - V)*g_s/C : volt (clock-driven)

    ddV_star_dw/dt = -(g_L + g_D)*dV_star_dw/C + g_D*dV_dw/C : volt (clock-driven)
    """

    if p.get("inst_delta", True):
        eqs_syn += """
        dw/dt = int(((t-lastspike) >= t_supp) or ((t-lastspike) <= 1.1*dt))*eta*((spiketrain - phi(V_star, alpha, beta, r_max)) * phi_prime(V_star, alpha, beta, r_max) / phi(V_star, alpha, beta, r_max) * dV_star_dw) : 1 (clock-driven)

        eta : 1
        """
    else:
        eqs_syn += """
        ddelta/dt = ((spiketrain - phi(V_star, alpha, beta, r_max)) * phi_prime(V_star, alpha, beta, r_max) / phi(V_star, alpha, beta, r_max) * dV_star_dw - delta)/tau_delta : 1 (clock-driven)
        """
        eqs_syn += """
        dw/dt = eta*delta : 1 (clock-driven)

        eta : 1
        """


    on_pre = """
    g_s += 1.*msiemens
    g_E_D += w*msiemens
    """

    p_pattern0 = 0.5
    patterns = np.array([0, 1]) #bernoulli.rvs(1.0-p_pattern0, size=N_presentations)

    rates = np.array([0*Hz/Hz*(1-patterns), f_in/Hz*patterns])
    rates = np.repeat(rates, [N_input/2,N_input/2], axis=0)

    rates = TimedArray(rates.T*Hz,dt=cycle)
    ns["rates"] = rates
    poisson = NeuronGroup(N_input, '',threshold='rand()<rates(t,i)*dt')

    syns = Synapses(poisson, NoI, eqs_syn, on_pre=on_pre, method="euler")
    syns.connect(True)
    syns.eta = group_size * p["eta"] / N_input
    w_base = 1e0 / N_input

    sigma_f = 0.1
    sigma = w_base*sigma_f

    syns.w = w_base + sigma*np.random.randn(2*N_input*group_size)

    min_weights = np.zeros(N_input*group_size*2)

    @network_operation(when="end", dt=20 * ms)
    def clip_weights():
        syns.w = np.maximum(syns.w, min_weights)

    M = StateMonitor(NoI, ["U", "V", "V_star", "I_adap","I_den", "I_L",  "g_adap", "I_inter", "I_intra"], record=True)
    SM = SpikeMonitor(NoI)
    SM2 = SpikeMonitor(poisson)
    SynMon = StateMonitor(syns, ["w"], record=True, dt=10*ms)
    ratem = PopulationRateMonitor(NoI)


    durat = len(patterns)*cycle

    run(durat, namespace=ns, report="text")
    plot(M.t/ms, M.U[0,:].T/mV)
    plot(M.t/ms, M.V[0,:].T/mV)

    hertz_factor = 1000.0/(cycle/ms)*1.0/(group_size)
    results = (M.t/ms, M.U/ms, M.V/ms, M.V_star/ms, gOUE, gOUI)
    #results = (SynMon1.t/ms, SynMon1.w, SynMon2.w, SM.spike_trains(), assigned_spikes, selectivities)
    dump(results, p["ident"])

def rate_learning():
    params = OrderedDict()
    params["Slrn"] = [False]
    params["Isp"] = [True]
    params["n_group"] = [1]
    params["t_ref"] = [20.0]
    params["we"] = [0.0]
    params["wi"] = [0.0]
    params["eta"] = [0.0]
    params["overlap"] = [0.0]
    params["seed"] = range(1)
    params["base"] = [1.0]
    params["N_input"] = [10,20,50,100]
    params["OUgLfactor"] = [1e-3,1e0,2e0, 4e0,1e1]

    file_prefix = "ornstein/p"
    do(simulate, params, file_prefix, withmp=False, create_notebooks=False)

rate_learning()
