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
      integ = lambda x: dRdE_standard(x, N_p, N_n, m_x, sig)
      #print(" No efficiency!")
  else:
      integ = lambda x: eff(x)*dRdE_standard(x, N_p, N_n, m_x, sig)
  return quad(integ, E_min, E_max)[0]

def Nevents_xenon(Eth, exposure, m_x, sigma):
  Emax=50.0
  Nevents = exposure*Nevents_standard(Eth, Emax, 74, 132-74, m_x, sigma)
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

def plot_bin_for_Eth(Eth, m_x,sigma):
  Emax=50.0
  Nevents = Nevents_standard(Eth, Emax, 74, 132-74, m_x, sigma)
  plt.bar((Eth+Emax)/2,Nevents, width=Emax-Eth, color="dodgerblue", alpha=0.5, edgecolor="dodgerblue")
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=16)
  plt.ylabel(r'${\rm Counts }\,\,{\rm kg}^{-1}{\rm days}^{-1}$', size=16)
  plt.xlim(xmin=0.0,xmax=55.0)
  plt.ylim(ymin=0.0)

  plt.show()

def plot_bin_for_exposure(Eth, m_x, sigma):
  Emax=50.0
  E1 = np.linspace(5.0, Emax, 10)
  Estep = E1[1]-E1[0]
  E2 =np.linspace(E1[0]+Estep, E1[-1]+Estep, 10)
  Nevents = Nevents_standard(Eth, Emax, 74, 132-74, m_x, sigma)
  plt.bar((Eth+Emax)/2,Nevents, width=Emax-Eth, color="dodgerblue", alpha=0.5, edgecolor="dodgerblue")
  plt.xlabel(r'$E_{R}\,\,\left[{\rm keV}\right]$', size=18)
  plt.ylabel(r'${\rm Counts }\,\,{\rm kg}^{-1}{\rm days}^{-1}$', size=18)
  plt.xlim(xmin=0.0,xmax=55.0)
  plt.ylim(ymin=0.0)

  plt.show()

import ipywidgets as widgets

sigma_slide = widgets.FloatLogSlider(value = 1e-45, min = -50, max = -42, description=r'$\sigma\,\, (\rm{cm}^2)$')
mdm_slide = widgets.FloatLogSlider(value = 100, min = np.log10(1.0), max=3, description=r'$m_{\rm DM}\,\, (\rm{GeV})$')
Eth_slide = widgets.FloatSlider(value = 10.0, min = 1.0, max=20.0, description=r'$E_{\rm{th}}\,\,(\rm{keV})$')
