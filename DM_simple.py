import numpy as np
from numpy import pi
from scipy.special import erf
from scipy.integrate import trapz, cumtrapz, quad


rho0 = 0.3 #GeV/cm^3

#Load in the list of nuclear spins and atomic masses
target_list = np.loadtxt("Nuclei.txt", usecols=(0,), dtype=bytes).astype(str)
A_list = np.loadtxt("Nuclei.txt", usecols=(1,))
J_list = np.loadtxt("Nuclei.txt", usecols=(2,))

Jvals = dict(zip(target_list, J_list))
Avals = dict(zip(target_list, A_list))

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import math

#---------------------------------------------------------
# Velocity integral eta
def calcEta(vmin, vlag=230.0, sigmav=156.0,vesc=544.0):
    
  aplus = np.minimum((vmin+vlag), vmin*0.0 + vesc)/(np.sqrt(2)*sigmav)
  aminus = np.minimum((vmin-vlag), vmin*0.0 + vesc)/(np.sqrt(2)*sigmav)
  aesc = vesc/(np.sqrt(2)*sigmav)
  
  vel_integral = 0
  
  N = 1.0/(erf(aesc) - np.sqrt(2.0/np.pi)*(vesc/sigmav)*np.exp(-0.5*(vesc/sigmav)**2))
  
  vel_integral = (0.5/vlag)*(erf(aplus) - erf(aminus))
  vel_integral -= (1.0/(np.sqrt(np.pi)*vlag))*(aplus - aminus)*np.exp(-0.5*(vesc/sigmav)**2)
  
  vel_integral = np.clip(vel_integral, 0, 1e30)

  return N*vel_integral


def calcEta_day(vmin, vlag=230.0, sigmav=156.0,vesc=544.0, t=242.4125):
  T=365.25
  t0 =151.0
  ve0 = 15.0
  ve = vlag + ve0*np.cos(2*np.pi*(t-t0)/T)
  eta = calcEta(vmin, vlag=ve)
  return eta

def vlag(vlag=230.0, t=242.4125):
  T=365.25
  t0 =151.0
  ve0 = 15.0
  ve = vlag + ve0*np.cos(2*np.pi*(t-t0)/T)

  return ve

def plot_eta(t): 
  vmin = np.linspace(0.001, 600, 300)
  plt.plot(vmin, calcEta(vmin), color='k', label=r'time average')
  plt.plot(vmin, calcEta_day(vmin,t=t), label=(r'%i days after Jan 1'%t))
  plt.ylim(ymax=0.004, ymin=1e-4)
  plt.xlim(xmin=0.0, xmax=600)
  plt.xlabel(r'$v_{\rm min}\,[{\rm km} {\rm s}^{-1}]$', size=16)
  plt.ylabel(r'$\eta(v_{\rm min})$', size=16)
  plt.legend(fontsize=14)
  plt.show()

#-----------------------------------------------------------
# Minimum velocity 
def vmin(E, A, m_x):
  m_A = A*0.9315
  mu = (m_A*m_x)/(m_A+m_x)
  v =  3e5*np.sqrt((E/1e6)*(m_A)/(2*mu*mu))
  return v
    

#-----------------------------------------------------------
# Reduced mass - input A as nucleon number and m_x in GeV
def reduced_m(A, m_x):
  m_A = 0.9315*A
  return (m_A * m_x)/(m_A + m_x)
    
# A helper function for calculating the prefactors to dRdE
def rate_prefactor(m_x):
  #mu = 1.78e-27*reduced_m(1.0, m_x)
  #return 1.38413e-12*rho0/(m_x*mu*mu)
  mu = reduced_m(1.0, m_x)
  return 4.34e41*rho0/(2.0*m_x*mu*mu)
    
#0.197 GeV  = 1e13/cm
# -> GeV^-1 = 1.97e-14 cm
    
def coupling_to_xsec(c, m_x):
  return (1.97e-14)**2*c**2*(reduced_m(1.0, m_x)**2)/np.pi

#-----------------------------------------------------------
# Standard Helm Form Factor for SI scattering
def calcSIFormFactor(E, m_N):

  #Define conversion factor from amu-->keV
  amu = 931.5*1e3

  #Convert recoil energy to momentum transfer q in keV
  q1 = np.sqrt(2*m_N*amu*E)

  #Convert q into fm^-1
  q2 = q1*(1e-12/1.97e-7)
  
  #Calculate nuclear parameters
  s = 0.9
  a = 0.52
  c = 1.23*(m_N**(1.0/3.0)) - 0.60
  R1 = np.sqrt(c*c + 7*pi*pi*a*a/3.0 - 5*s*s)
  

  x = q2*R1
  J1 = np.sin(x)/x**2 - np.cos(x)/x
  F = 3*J1/x
  
  formfactor = (F**2)*(np.exp(-(q2*s)**2))
  #formfactor[E < 1e-3] = 1.0
  return formfactor

#----------------------------------------------
#-------- RECOIL RATES ------------------------
#----------------------------------------------
    
#--------------------------------------------------------
def dRdE_standard(E, N_p, N_n, m_x, sig, vlag=232.0, sigmav=156.0, vesc=544.0):
  """
  Standard Spin-Independent recoil rate
  for a particle with (N_p,N_n) protons and neutrons
  ----------
  * `E` [array]:
    Recoil energies.
  * `N_p` [integer]:
    Number of protons in target nucleus.
  * `N_n` [integer]:
    Number of neutrons in target nucleus.
  * `m_x` [float]:
    Dark Matter mass in GeV.
  * `sig` [float]:
    Cross section in cm^{-2}.
  * `vlag` [float]:
    #FIXME: description
  * `sigmav` [float]:
    #FIXME: description
  * `vesc` [float]:
    #FIXME: description
  Returns
  -------
  * `rate` [array like]:
    Recoil rate in units of events/keV/kg/day.
  """
  A = N_p + N_n   
  #print A
  int_factor = sig*calcSIFormFactor(E, A)*(A**2)
  
  return rate_prefactor(m_x)*int_factor*calcEta(vmin(E, A, m_x), vlag, sigmav, vesc)

def rate_xenon(Er, m_x, sigma):
  return dRdE_standard(Er, 74, Avals['Xe132']-74,  m_x, sigma)

def rate_argon(Er, m_x, sigma):
  return dRdE_standard(Er, 18, Avals['Ar40']-18,  m_x, sigma)

def rate_germanium(Er, m_x, sigma):
  return dRdE_standard(Er, 32, Avals['Ge74']-32,  m_x, sigma)

#--------------------------------------------------------
# Total number of events for standard SI DM
@np.vectorize
def Nevents_standard(E_min, E_max, N_p, N_n, m_x, sig, eff=None,vlag=232.0, sigmav=156.0, vesc=544.0):
  if (eff == None):
      integ = lambda x: dRdE_standard(x, N_p, N_n, m_x, sig, vlag=vlag)
      #print(" No efficiency!")
  else:
      integ = lambda x: eff(x)*dRdE_standard(x, N_p, N_n, m_x, sig, vlag=vlag)
  return quad(integ, E_min, E_max)[0]

def Nevents_xenon(Eth, exposure, m_x, sigma, vlag=232.0,):
  Emax=50.0
  Nevents = exposure*Nevents_standard(Eth, Emax, 74, 132-74, m_x, sigma, vlag=vlag)
  return Nevents

def plot_spec(m_x, sigma):
  Er = np.linspace(0.01, 200, 200)
  plt.semilogy(Er, dRdE_standard(Er, 74, Avals['Xe132']-74,  m_x, sigma), label=r'$\rm Xe}$')
  plt.semilogy(Er, dRdE_standard(Er, 18, Avals['Ar40']-18,  m_x, sigma), label=r'${\rm Ar}$')
  plt.semilogy(Er, dRdE_standard(Er, 32, Avals['Ge74']-32,  m_x, sigma), label=r'${\rmGe}$')
  plt.xlim(xmin=0.0,xmax=200)
  plt.legend(fontsize=16)
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=16)
  plt.ylabel(r'${\rm d}R/{\rm d}E_R\,\,\left[{\rm keV}^{-1}{\rm kg}^{-1}{\rm days}^{-1}\right]$', size=16)
  plt.show()

def plot_spec_xe(m_x, sigma):
  Er = np.linspace(0.01, 60, 200)
  plt.semilogy(Er, dRdE_standard(Er, 74, Avals['Xe132']-74,  m_x, sigma), label=r'$\rm Xe}$')
  plt.xlim(xmin=0.0,xmax=60)
  plt.legend(fontsize=16)
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=16)
  plt.ylabel(r'${\rm d}R/{\rm d}E_R\,\,\left[{\rm keV}^{-1}{\rm kg}^{-1}{\rm days}^{-1}\right]$', size=16)
  plt.show()

def plot_for_Eth(Eth, m_x,sigma):
  Er = np.linspace(0.0001, 60, 200)
  drde = dRdE_standard(Er, 74, Avals['Xe132']-74,  m_x, sigma)
  plt.semilogy(Er, drde, label=r'$\rm Xe}$')
  plt.axvline(x=Eth, color='k', linestyle='dashed')
  Emax=50.0
  Nevents = Nevents_standard(Eth, Emax, 74, 132-74, m_x, sigma)
  #print(Nevents)
  if Nevents > 0:
    plt.text(15, drde[0]*0.5, r'$\rm{Counts}=%.2f\times 10^{%i}\,{\rm kg}^{-1}\,\,{\rm days}^{-1}$ '%(Nevents/(10**math.floor(np.log10(Nevents))), math.floor(np.log10(Nevents))) , fontsize=14)
  else: 
    plt.text(15, drde[0]*0.5, r'$\rm{Counts}=0 \,{\rm kg}^{-1}\,\,{\rm days}^{-1}$ ', fontsize=14)

  #plt.bar((Eth+Emax)/2,Nevents, width=Emax-Eth, color="dodgerblue", alpha=0.5, edgecolor="dodgerblue")
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=16)
  plt.ylabel(r'${\rm d}R/{\rm d}E_R\,\,\left[{\rm keV}^{-1}{\rm kg}^{-1}{\rm days}^{-1}\right]$', size=16)
  plt.xlim(xmin=0.0,xmax=60.0)
  #plt.ylim(ymin=0.0)

  plt.show()


def plot_bin_reconstruction(Eth,exposure, m_x, sigma):
  Emax=50.0
  E1 = np.linspace(Eth, Emax, 10)
  Estep = E1[1]-E1[0]
  E2 =np.linspace(E1[0]+Estep, E1[-1]+Estep, 10)
  Data_recon = exposure*Nevents_standard(E1, E2, 74, 132-74, 200, 2e-46)
  Nevents = exposure*Nevents_standard(E1, E2, 74, 132-74, m_x, sigma)
  plt.bar((E1+E2)/2,Nevents, width=E2-E1, color="dodgerblue", alpha=0.5, edgecolor="dodgerblue", label='Theory')
  plt.bar((E1+E2)/2,Data_recon, width=E2-E1, color="None", edgecolor="k", label=r'Observed')
  plt.legend(fontsize=14)
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=18)
  plt.ylabel(r'${\rm Counts }$', size=18)
  plt.xlim(xmin=0.0,xmax=55.0)
  plt.ylim(ymin=0.0, ymax=10.0)

  plt.show()

def plot_modulation(m_x, sigma, t1, t2):
  fig, ax = plt.subplots(1,2, figsize=(12,5))

  Er = np.linspace(0.01, 30, 200)
  vlag1 = vlag(t=t1)
  rate1 = dRdE_standard(Er, 74, Avals['Xe132']-74,  m_x, sigma, vlag=vlag1)
  ax[0].plot(Er, rate1, label=r'${\rm d}R/{\rm d}E_R(t1)$')
  vlag2 = vlag(t=t2)
  rate2 = dRdE_standard(Er, 74, Avals['Xe132']-74,  m_x, sigma, vlag=vlag2)
  ax[0].plot(Er, rate2, label=r'${\rm d}R/{\rm d}E_R(t2)$')
  ax[0].set_xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=16)
  ax[0].set_ylabel(r'${\rm d}R/{\rm d}E_R\,\,\left[{\rm keV}^{-1}{\rm kg}^{-1}{\rm days}^{-1}\right]$', size=16)
  ax[0].set_xlim(xmin=0.0,xmax=30.0)
  ax[0].legend(fontsize=14)

  modulation = 0.5*(rate2-rate1)
  ax[1].plot(Er, modulation)
  ax[1].set_xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=16)
  ax[1].set_ylabel(r'$\Delta(t_1,t_2)\,\,\left[{\rm keV}^{-1}{\rm kg}^{-1}{\rm days}^{-1}\right]$', size=16)
  ax[1].set_xlim(xmin=0.0,xmax=30.0)
  plt.tight_layout()
  plt.show()



def mod_timeseries(exposure,m_x, sigma, E1, E2):

  S0_Data = exposure*Nevents_standard(E1, E2, 74, 132-74, 200, 2e-46)
  #Data_recon = exposure*Nevents_standard(E1, E2, 74, 132-74, 200, 5e-45)
  S0 = exposure*Nevents_standard(E1, E2, 74, 132-74, m_x, sigma )
  tjune = 151
  tdec = 334
  vjune = vlag(t=tjune)
  vdec = vlag(t=tdec)
  CountJune = exposure*Nevents_standard(E1, E2, 74, 132-74, m_x, sigma, vlag=vjune )
  Data_June = exposure*Nevents_standard(E1, E2, 74, 132-74, 200, 2e-46, vlag=vjune)

  CountDec = exposure*Nevents_standard(E1, E2, 74, 132-74, m_x, sigma, vlag=vdec )
  Data_Dec = exposure*Nevents_standard(E1, E2, 74, 132-74, 200, 2e-46, vlag=vdec)

  Sm= 0.5*(CountJune - CountDec)
  Sm_Data = 0.5*(Data_June-Data_Dec)

  #print(Sm, CountJune, CountDec, vdec, vjune)

  phase = 151.0
  T = 365.25
  t1 = np.linspace(0,365.25, 30)
  tstep = t1[1]-t1[0]
  t2 =np.linspace(t1[0]+tstep, t1[-1]+tstep, 30)

  argt1 = 2*np.pi*(t1-phase)/T
  argt2 = 2*np.pi*(t2-phase)/T
  delt = t2-t1
  #print(t1,t2)
  res = Sm*(T/(2*np.pi))*(np.sin(argt2) - np.sin(argt1))
  data_res =  Sm_Data*(T/(2*np.pi))*(np.sin(argt2) - np.sin(argt1))

  return t1, t2, res, data_res


def plot_modulation_timeseries(Eth, exposure, m_x, sigma):
  
  Emax=20.0
  E1 = np.linspace(Eth, Emax, 4)
  Estep = E1[1]-E1[0]
  E2 =np.linspace(E1[0]+Estep, E1[-1]+Estep, 4)

  fig, ax = plt.subplots(2,2, figsize=(12,10))
  counter=0
  for i in range(2):
    for j in range(2):
      t1, t2, res, data_res = mod_timeseries(exposure,m_x, sigma, E1[counter], E2[counter])
      ax[i,j].bar((t1+t2)/2,res, width=t2-t1, color="dodgerblue", alpha=0.5, edgecolor="dodgerblue", label='Theory')
      ax[i,j].bar((t1+t2)/2,data_res, width=t2-t1, color="None", alpha=0.5, edgecolor="k", label='observation')
      
      ax[i,j].set_title(r"$E_R=[%.1f-%.1f]\,{\rm keV}$"%(E1[counter], E2[counter]))
      ax[i,j].legend(fontsize=14)
      ax[i,j].set_xlabel(r'$t\,\,[{\rm days}]$', size=18)
      ax[i,j].set_ylabel(r'$\Delta\,{\rm Counts }$', size=18)
      ax[i,j].set_xlim(xmin=0.0,xmax=365.0)
      ax[i,j].set_ylim(ymin=-2.0, ymax=2.0)
      counter += 1

  plt.tight_layout()

  plt.show()

#def plot_mod_time_series
  


import ipywidgets as widgets

sigma_slide = widgets.FloatLogSlider(value = 1e-45, min = -50, max = -42,step=0.01,
                                      layout={"width" : "400px"}, description=r'$\sigma\,\, (\rm{cm}^2)$')
mdm_slide = widgets.FloatLogSlider(value = 100, min = np.log10(1.0), max=3, step=0.01,
                                    layout={"width":"400px"}, description=r'$m_{\rm DM}\,\, (\rm{GeV})$')
Eth_slide = widgets.FloatSlider(value = 10.0, min = 1.0, max=20.0,step=0.1,
                                layout={"width":"400px"}, description=r'$E_{\rm{th}}\,\,(\rm{keV})$')
expo_slide = widgets.FloatLogSlider(value = 1.0, min = np.log10(1.0), max=5.0,step=0.1,
                                    layout={"width":"400px"}, description=r'$\epsilon\, (\rm{kg}\, \rm{days})$')

g2_expo_slide = widgets.FloatLogSlider(value = 2e4, min = 4.0, max=6.0,step=0.1,
                                      layout={"width":"400px"}, description=r'$\epsilon\, (\rm{kg}\, \rm{days})$')


time_slide = widgets.IntSlider(value = 0, min = 0, max=366,step=1,
                                    layout={"width":"400px"}, description=r'Days since Jan 1')

time_slide1 = widgets.IntSlider(value = 0, min = 0, max=366,step=1,
                                    layout={"width":"400px"}, description=r'time1')
time_slide2 = widgets.IntSlider(value = 0, min = 0, max=366,step=1,
                                    layout={"width":"400px"}, description=r'time 2')

def read_efficiency():
  from scipy.interpolate import interp1d
  
  efficiency = np.genfromtxt('Xenon1t.dat')
  xe1t_eff = interp1d(efficiency[:,0], efficiency[:,1], kind='linear', bounds_error=False, 
                      fill_value=0.0)
  return xe1t_eff

def plot_efficiency():
  eff = read_efficiency()
  Er = np.linspace(0,60,200)
  plt.plot(Er, eff(Er))
  plt.xlim(xmin=0.0,xmax=60.0)
  plt.ylim(ymin=0.0, ymax=1.0)
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=16)
  
  plt.ylabel(r'${\rm Efficiency}$', size=16)
  plt.show()

def plot_xe1t_bin_eff(m_x, sigma):
  Emax=50.0
  Eth = 2.0
  exposure = 0.9*1e3*278.8*0.475
  E1 = np.linspace(Eth, Emax, 12)
  Estep = E1[1]-E1[0]
  E2 =np.linspace(E1[0]+Estep, E1[-1]+Estep, 12)
  efficiency = read_efficiency()
  #Data_recon = exposure*Nevents_standard(E1, E2, 74, 132-74, 200, 5e-45)
  eff_events = exposure*Nevents_standard(E1, E2, 74, 132-74, m_x, sigma, eff = efficiency )
  Nevents = exposure*Nevents_standard(E1, E2, 74, 132-74, m_x, sigma)
  plt.bar((E1+E2)/2,Nevents, width=E2-E1, color="dodgerblue", alpha=0.5, edgecolor="dodgerblue", label=r'$\rm{w/o\,\, eff}$')
  plt.bar((E1+E2)/2,eff_events, width=E2-E1, color="None", edgecolor="k", label=r'$\rm{w\,\,eff}$')
  plt.legend(fontsize=14)
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=18)
  plt.ylabel(r'${\rm Counts }\,\,{\rm kg}^{-1}{\rm days}^{-1}$', size=18)
  plt.xlim(xmin=0.0,xmax=55.0)
  #plt.ylim(ymin=0.0, ymax=10.0)

  plt.show()

def Nevents_xenon_eff(Eth, exposure, m_x, sigma):
  efficiency = read_efficiency()
  Emax=50.0
  Nevents = exposure*Nevents_standard(Eth, Emax, 74, 132-74, m_x, sigma, eff = efficiency)
  return Nevents

def find_sigma90(Eth, expo, m_x):

  ref_sig = 1e-46
  Nevents_ref = Nevents_xenon(Eth = Eth, exposure=expo, m_x = m_x, sigma=ref_sig)
  sigma90 = (ref_sig*2.303)/Nevents_ref
  

  return sigma90

def plot_exclusion(Eth, exposure):
  mass_space = np.geomspace(6,1e3)
  plt.loglog(mass_space,find_sigma90(Eth, exposure,mass_space))
  plt.xlim(xmin=6.0,xmax=1e3)
  plt.ylim(ymin=1e-47, ymax=2e-43)
  plt.xlabel(r'$m_{\rm DM}\,\,\left[{\rm GeV}\right]$', size=18)
  plt.ylabel(r'$\sigma\,\,\left[{\rm cm}^2\right]$', size=18)
  plt.show()


from IPython.display import HTML, display

def set_background(color):    
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)
    
    display(HTML('<img src onerror="{}" style="display:none">'.format(script)))