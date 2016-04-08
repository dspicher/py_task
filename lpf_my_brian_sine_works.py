import json

import numpy as np
from brian2 import *


@implementation('cpp', '''
     #include<math.h>
     double phi(double U, double alpha, double beta, double r_max) {
        return r_max / (1 + exp(-beta*(U-alpha)));
     }
     ''')
@check_units(U=volt, alpha=volt, beta=volt**-1, r_max=second**-1, result=second**-1)
def phi(U, alpha, beta, r_max):
    pass


@implementation('cpp', '''
     #include<math.h>
     double phi_prime(double U, double alpha, double beta, double r_max) {
        double num = exp((U + alpha) * beta) * r_max * beta;
        double denom = pow(exp(U * beta) + exp(alpha * beta), 2.0);
        return num / denom;
     }
     ''')
@check_units(U=volt, alpha=volt, beta=volt**-1, r_max=second**-1, result=(second * volt)**-1)
def phi_prime(U, alpha, beta, r_max):
    pass


def simulate(neuron, g_fac):

    from IPython import embed

    neuron["g_S"] = 0.5

    ns = {}
    for key, value in neuron.items():
        if key[0] == 'g':
            unit = msiemens
        elif key[0] == 'E':
            unit = mV
        elif key[0] == 't':
            unit = ms
        elif key == 'alpha':
            unit = mV
        elif key == 'beta':
            unit = mV**-1
        elif key == 'r_max':
            unit = ms**-1
        else:
            raise Exception()
        ns[key] = value * unit

    ns['phi_prime'] = phi_prime
    ns['phi'] = phi

    defaultclock.dt = 0.1*ms

    np.random.seed(13)  # seed for pseudo-randomness

    input_rate = 20 * Hz
    N_input = 100
    cycle_dur = 100 * ms

    spike_nrs = np.random.poisson(input_rate * cycle_dur, N_input)
    indices = []
    times = np.array([])
    for i, nr in enumerate(spike_nrs):
        indices = indices + [i for _ in range(nr)]
        times = np.concatenate((times, cycle_dur / ms * np.random.rand(nr)))
    times = times * ms

    #indices = range(N_input)
    #times = arange(0, cycle_dur / ms, cycle_dur / ms / N_input) * ms

    Noise = SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)

    pre_epochs = 1
    learn_epochs = 30
    test_epochs = 3
    epochs = pre_epochs + learn_epochs + test_epochs
    l_c = 8
    eval_c = 2
    epoch_dur = cycle_dur * (l_c + eval_c)
    cycles = epochs * (l_c + eval_c)
    t_end = cycles * cycle_dur
    g_factor = g_fac
    exc_level = 7e-3

    def g_E_teacher(t):
        if t < pre_epochs * epoch_dur or t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c or t > (pre_epochs + learn_epochs) * epoch_dur:
            return 0.0
        else:
            return ((1 + np.sin(-np.pi / 2 + t / t_end * cycles * 2 * np.pi)) * exc_level + exc_level) * g_factor

    def g_I_teacher(t):
        if t < pre_epochs * epoch_dur or t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c or t > (pre_epochs + learn_epochs) * epoch_dur:
            return 0.0
        else:
            return 4e-2 * g_factor

    dt_teacher = 0.5 * ms
    N_values = int(t_end / (dt_teacher))
    g_E_s = np.zeros(N_values)
    g_I_s = np.zeros(N_values)
    for idx in xrange(N_values):
        g_E_s[idx] = g_E_teacher(idx * dt_teacher)
        g_I_s[idx] = g_I_teacher(idx * dt_teacher)

    ns['g_E'] = TimedArray(g_E_s * msiemens, dt_teacher)
    ns['g_I'] = TimedArray(g_I_s * msiemens, dt_teacher)

    ns['C'] = 1e-6 * farad

    nrn_eqs = '''
    #Teaching
    I_som = -g_E(t)*(U - E_E) - g_I(t)*(U - E_I) : amp

    #Spike currents
    #I_spike = 0.0*uamp : amp
    I_spike = - g_Na*(U - E_Na)*int((t-lastspike) < t_rise) - g_K*(U - E_K)*int((t-lastspike) > t_rise)*int((t-lastspike) < t_fall) : amp

    #Somatic potential
    dU/dt = (-g_L*(U-E_L) - g_D*(U - V))/C + I_som/C + I_spike/C : volt

    #Dendritic potential
    dV/dt = -g_L*(V-E_L)/C - g_E_D_tot*(V - E_E)/C - g_S*(V-U)/C : volt

    #Dendritic prediction
    dV_star/dt = -g_L*(V_star - E_L)/C - g_D*(V_star - V)/C : volt

    #sum of excitatory conductance onto dendrite
    g_E_D_tot : siemens
    '''

    NoI = NeuronGroup(1, nrn_eqs, threshold='phi(U, alpha, beta, r_max)*dt>rand()',
                      refractory=0 * ms)
    NoI.U = -70.0 * mV
    NoI.V = -70.0 * mV

    # equations for the synapses
    eqs_syn = '''
    #Input without weight
    dg_s/dt = -g_s/tau_s : siemens

    #Input with weight
    dg_E_D/dt = -g_E_D/tau_s : siemens

    g_E_D_tot_post = g_E_D : siemens (summed)

    #Derivatives w.r.t the synaptic weight
    ddV_dw/dt = -(g_L + g_S + g_E_D)*dV_dw/C + g_S*dV_star_dw/C + (E_E - V)*g_s/C : volt

    ddV_star_dw/dt = -(g_L + g_D)*dV_star_dw/C + g_D*dV_dw/C : volt

    ddelta/dt = ((int((t-lastspike) < 0.11*ms)/(dt/second) - phi(V_star, alpha, beta, r_max)) * phi_prime(V_star, alpha, beta, r_max) / phi(V_star, alpha, beta, r_max) * dV_star_dw - delta)/tau_delta : 1

    dw/dt = eta*delta : 1
    '''

    pre = '''
    g_s += 1.*msiemens
    g_E_D += w*msiemens
    '''

    S = Synapses(Noise, NoI, eqs_syn, pre=pre)

    min_weights = np.zeros(N_input)

    @network_operation(when='end', dt=20 * ms)
    def clip_weights():
        S.w = np.maximum(S.w, min_weights)

    S.connect(True)
    S.w = 5.5e-1 / N_input #+ 1e-2 / N_input * np.random.randn(N_input)
    ns['eta'] = 8e-4 / N_input

    M = StateMonitor(NoI, ['U', 'V', 'V_star'], record=True, dt=1*ms)
    SM = SpikeMonitor(NoI)
    SynMon = StateMonitor(S, ['w', 'dV_dw', 'dV_star_dw', 'g_s', 'g_E_D', 'delta'], record=True, dt=1*ms)

    run(t_end, namespace=ns, report='stdout')

    U_M = (g_E_s*nrn["E_E"] + g_I_s*nrn["E_I"] + nrn["g_L"]*nrn["E_L"]) / (g_E_s + g_I_s + nrn["g_L"])

    figure(figsize=(20, 6))
    subplot(1, 2, 1)
    plot(M.t / ms, M.U.T / mV)
    plot(M.t / ms, M.V.T / mV)
    subplot(1, 2, 2)
    plot(SynMon.t / ms, SynMon.w.T)

    figure(figsize=(20,6))
    nudge_ends = arange(epoch_dur/ms, t_end/ms, epoch_dur/ms)
    offset = 100
    curr_off = 0
    cycles_back = 1
    cycles_fw = 1
    for i, nudge_end in enumerate(nudge_ends[::4]):
        mask1 = M.t/ms < (nudge_end + cycles_fw*cycle_dur/ms)
        mask2 = M.t/ms > (nudge_end - cycles_back*cycle_dur/ms)
        mask = np.logical_and(mask1, mask2)
        plot(curr_off+arange(np.sum(mask)), M.U.T[mask]/mV,'b')
        plot(curr_off+arange(np.sum(mask)), M.V.T[mask]/mV,'k')
        curr_off = curr_off + offset + np.sum(mask)

    import cPickle
    #cPickle.dump((M.t/ms, M.U/mV, M.V/mV, SynMon.t/ms, SynMon.w.T), open('brian_reproduced.p','wb'))

    show()
    embed()

nrn = json.load(open('default_neuron.json', 'r'))
for g_factor in [20.0]:
    nrn["alpha"] = -50.0
    nrn["beta"] = 0.35
    nrn["r_max"] = 0.1
    simulate(nrn, g_factor)
