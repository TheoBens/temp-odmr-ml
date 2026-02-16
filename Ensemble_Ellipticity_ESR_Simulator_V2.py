import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def Hint_e(phase, Ox, coef_x, Oy, coef_y):
    Ax = Ox * coef_x[0] / (2 *np.sqrt(2))
    Ay = Ox * coef_x[1] / (2 *np.sqrt(2))
    Bx = Oy * coef_y[0] / (2 *np.sqrt(2))
    By = Oy * coef_y[1] / (2 *np.sqrt(2))

    # Interaction Hamiltonian for the Electrons
    H_int = np.array([[0, (Ax+By*np.exp(-1j*phase)), 0],[(Ax+By*np.exp(1j*phase)), 0, (Ax-By*np.exp(1j*phase))],[0,(Ax-By*np.exp(-1j*phase)), 0]])
    H_int = H_int + np.array([[0, Bx * np.exp(-1j*phase), 0],[Bx * np.exp(1j*phase), 0, Bx * np.exp(1j*phase)],[0, Bx * np.exp(-1j*phase), 0]])
    H_int = H_int + np.array([[0, -1j*Ay, 0],[1j*Ay, 0, -1j*Ay],[0, 1j*Ay,0]])
    # print(H_int)

    return H_int

def H0_e(Bz=0, W_mw = 2.87e3):
    # Ground State Hamiltonian
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    Sz2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    #Ground State parameters (in MHz units):
    D = 2.87e3
    gamma_e = 28.024e3 # MHz/T
    
    H0 = D * Sz2 + gamma_e * Bz * Sz - W_mw * Sz2

    return np.array(H0)

def MeanPop_e(t=2, Bz=1e-3, W_mw=2.87e3, phase=0, Ox = 1, coef_x=None, Oy = 1, coef_y=None):
    # Calculate Minimum population of state 0 ater MW pulse of length t
    tlist = np.linspace(0,t,10)
    H0 = H0_e(Bz, W_mw)
    # print(phase)
    # print(f"Ox:{Ox}, Oy:{Oy}")
    H_mw = Hint_e(phase, Ox, coef_x, Oy, coef_y)
    H = H0 + H_mw
    U = np.array([linalg.expm(-1j*H*t) for t in tlist]) # Evolution operators
    # Evolve State 0 and save evolution in arrays
    states = np.array([np.matmul(np.matmul(Ut,np.array([[0,0,0],[0,1,0],[0,0,0]])), Ut.conj().T) for Ut in U])

    PiPulsePop0 = np.mean(np.real(states[:, 1, 1]))
    return PiPulsePop0

def MeanPop_HF(t=10, Bz=1e-3, W_mw=2.87e3, phase = 0, Ox = 1, coef_x=None, Oy = 1, coef_y=None):
    # Calculate Minimum population of state 0 ater MW pulse of length t
    tlist = np.linspace(0,t,10)
    H0 = H0_e(Bz, W_mw)
    H_mw = Hint_e(phase, Ox, coef_x, Oy, coef_y)
    H_e = H0 + H_mw
    Iz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    Azz = 2.14
    Hhf = Azz * np.kron(Iz, Iz)
    H = np.kron(H_e, np.eye(3,3)) + Hhf 
    
    
    U = np.array([linalg.expm(-1j*H*t) for t in tlist]) # Evolution operators
    # Evolve State 0 and save evolution in arrays
    rhoT = np.kron(np.array([[0,0,0],[0,1,0],[0,0,0]]), (1/3)*np.eye(3,3))
    states = np.array([np.matmul(np.matmul(Ut,rhoT), Ut.conj().T) for Ut in U])
    
    Proj0 = np.kron(np.array([[0,0,0],[0,1,0],[0,0,0]]), np.eye(3,3))

    Pop0 = [np.trace(np.matmul(Proj0, states[tidx, :, :])) for tidx, _ in enumerate(tlist)]
    PiPulsePop0 = np.mean(Pop0)
    return PiPulsePop0

def SingleNV_spectrum_HF(Bz, phase, Ox, coef_x, Oy, coef_y, freqList = None):
    # Generate Spectrum from a single Nv axis. HYPERFINE INCLUDED
    # Takes:
    #   Magnetic field parallel to the axis, Bz
    #   MW pulse parameters for that axis
    #   Start, end frequencies and frequency step Nr.
    
    RawVals = [MeanPop_HF(t=5, Bz=Bz, W_mw=w, phase=phase, Ox = Ox, coef_x=coef_x, Oy = Oy, coef_y=coef_y) for w in freqList]
    #print(RawVals)
    RawVals = np.array(RawVals)
    Contrast = np.array(RawVals/np.max(RawVals))
    return Contrast

def SingleNV_spectrum(Bz, phase_, Ox, coef_x, Oy, coef_y, freqList = None):
    # Generate Spectrum from a single Nv axis. NO HYPERFINE
    # Takes:
    #   Magnetic field parallel to the axis, Bz
    #   MW pulse parameters for that axis
    #   Start, end frequencies and frequency step Nr.
    
    RawVals = [MeanPop_e(t=5, Bz=Bz, W_mw=w, phase=phase_, Ox= Ox, coef_x=coef_x, Oy= Oy, coef_y=coef_y) for w in freqList]
    #print(RawVals)
    RawVals = np.array(RawVals)
    Contrast = np.array(RawVals/np.max(RawVals))
    return Contrast




def rot_x(alpha):
    return np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])

def rot_y(beta):
    return np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])


def extract_numeric_coefficients(alpha, beta):

    TlabNV = [
    np.array([
        [1/np.sqrt(3), 0,  np.sqrt(2/3)],
        [0, 1, 0],
        [-np.sqrt(2/3), 0, 1/np.sqrt(3)]
    ]),
    np.array([
        [1/np.sqrt(3), 0, -np.sqrt(2/3)],
        [0, 1, 0],
        [ np.sqrt(2/3), 0, 1/np.sqrt(3)]
    ]),
    np.array([
        [1, 0, 0],
        [0, 1/np.sqrt(3), -np.sqrt(2/3)],
        [0, np.sqrt(2/3),  1/np.sqrt(3)]
    ]),
    np.array([
        [1, 0, 0],
        [0, 1/np.sqrt(3),  np.sqrt(2/3)],
        [0, -np.sqrt(2/3), 1/np.sqrt(3)]
    ])
    ]

    # Basis vectors
    vx = (1/np.sqrt(2)) * np.array([1, -1, 0])  # multiplies x*cos(a)
    vy = (1/np.sqrt(2)) * np.array([1,  1, 0])  # multiplies y*sin(b)

    Rtot = rot_x(alpha) @ rot_y(beta)
    R_T = Rtot.T

    results = []

    for i, T in enumerate(TlabNV, start=1):
        coeff_x = T @ R_T @ vx
        coeff_y = T @ R_T @ vy

        results.append({
            "NV": i,
            "Ox*cos(a)": coeff_x,
            "Oy*sin(b)": coeff_y
        })

    return results



def transform_field_toNV(B, alpha=0, beta=0):
    # B_column = [[B[0]], [B[1]], [B[2]]]EN
    Bx_NV = np.sqrt(1/2)*(B[0]+B[1])
    By_NV = np.sqrt(1/2)*(-B[0]+B[1])
    B_column = [[Bx_NV], [By_NV], [B[2]]]
    

    # Rotation of the diamond WRST antenna:
    Rtot = np.matmul(rot_x(alpha), rot_y(beta))

    TlabNV1 =  np.array([[1/np.sqrt(3), 0, np.sqrt(2/3)],
                        [0,  1,  0],
                        [-np.sqrt(2/3), 0,  1/np.sqrt(3)]])
    
    TlabNV2 =np.array([[1/np.sqrt(3), 0, -np.sqrt(2/3)],
                        [0,  1,  0],
                        [np.sqrt(2/3), 0,  1/np.sqrt(3)]])


    TlabNV3 = np.array([[1, 0, 0],
                        [0, 1/np.sqrt(3), -np.sqrt(2/3)],
                        [0, np.sqrt(2/3), 1/np.sqrt(3)]])


    TlabNV4 = np.array([[1, 0, 0],
                        [0, 1/np.sqrt(3), np.sqrt(2/3)],
                        [0, -np.sqrt(2/3), 1/np.sqrt(3)]])

    # Calculate Tranformation matrices for tilted diamond:
    TlabNV1 = np.dot(TlabNV1, Rtot.T)
    TlabNV2 = np.dot(TlabNV2, Rtot.T)
    TlabNV3 = np.dot(TlabNV3, Rtot.T)
    TlabNV4 = np.dot(TlabNV4, Rtot.T)

    # Non tilted NV axis
    uNV1 = np.array([-np.sqrt(2/3), 0.0 , 1/np.sqrt(3)])
    uNV2 = np.array([np.sqrt(2 / 3), 0.0, 1 / np.sqrt(3)])
    uNV3 = np.array([0.0, np.sqrt(2 / 3), 1 / np.sqrt(3)])
    uNV4 = np.array([0.0, -np.sqrt(2 / 3), 1 / np.sqrt(3)])

    B_in_NV_basis = {'NV_1': [], 'NV_2': [], 'NV_3': [], 'NV_4': []}
    B_column_NV1 = np.dot(TlabNV1, B_column)
    B_column_NV2 = np.dot(TlabNV2, B_column)
    B_column_NV3 = np.dot(TlabNV3, B_column)
    B_column_NV4 = np.dot(TlabNV4, B_column)
    B_in_NV_basis['NV_1'] = [B_column_NV1[0,0], B_column_NV1[1, 0], B_column_NV1[2, 0]]
    B_in_NV_basis['NV_2'] = [B_column_NV2[0, 0], B_column_NV2[1, 0], B_column_NV2[2, 0]]
    B_in_NV_basis['NV_3'] = [B_column_NV3[0, 0], B_column_NV3[1, 0], B_column_NV3[2, 0]]
    B_in_NV_basis['NV_4'] = [B_column_NV4[0, 0], B_column_NV4[1, 0], B_column_NV4[2, 0]]


    return B_in_NV_basis

def Ensemble_Spectrum(B, MW_field, MW_phase, freqList = None, tilt_x=0, tilt_y = 0):
    # Calculate Ensemble Spectrum taking only:
    #    Magnetic field:   B = [Bx, By, Bz]
    #   Microwave field:   MW_field = [Ox, Oy, 0]
    #   Microwave phase:   MW_phase = float in [0, 2*pi]

    B_inNV = transform_field_toNV(B, tilt_x, tilt_y)
    Omx = MW_field[0]
    Omy = MW_field[1]
    num_coeffs = extract_numeric_coefficients(tilt_x, tilt_y)
    EnsCont = np.zeros(len(freqList))
    idxCount = 0
    for nv in B_inNV:
        Bz = B_inNV[nv][2]
        coef_x = num_coeffs[idxCount]['Ox*cos(a)']
        coef_y = num_coeffs[idxCount]['Oy*sin(b)']
        # print(f"NV axis: {nv}, MWphase:{MW_phase}, coefx: {coef_x}, coefy: {coef_y}")
        Contrast = SingleNV_spectrum(Bz, MW_phase, Omx, coef_x, Omy, coef_y, freqList)
        # plt.plot(freqList,Contrast)
        EnsCont = (EnsCont + Contrast)   
        idxCount += 1      
    

    EnsCont = EnsCont/4 # Divide by number of axis
    # plt.plot(freqList, EnsCont)
    # plt.show()
    return EnsCont

def Ensemble_Spectrum_HF(B, MW_field, MW_phase, freqList = None, tilt_x=0, tilt_y = 0):
    # Calculate Ensemble Spectrum taking only:
    #    Magnetic field:   B = [Bx, By, Bz]
    #   Microwave field:   MW_field = [Ox, Oy, 0]
    #   Microwave phase:   MW_phase = float in [0, 2*pi]

    B_inNV = transform_field_toNV(B, tilt_x, tilt_y)
    Omx = MW_field[0]
    Omy = MW_field[1]
    num_coeffs = extract_numeric_coefficients(tilt_x, tilt_y)
    EnsCont = np.zeros(len(freqList))
    idxCount = 0
    for nv in B_inNV:
        Bz = B_inNV[nv][2]
        coef_x = num_coeffs[idxCount]['Ox*cos(a)']
        coef_y = num_coeffs[idxCount]['Oy*sin(b)']
        # print(f"NV axis: {nv}, MWphase:{MW_phase}, coefx: {coef_x}, coefy: {coef_y}")
        Contrast = SingleNV_spectrum_HF(Bz, MW_phase, Omx, coef_x, Omy, coef_y, freqList)
        # plt.plot(freqList,Contrast)
        EnsCont = (EnsCont + Contrast)   
        idxCount += 1      
    

    EnsCont = EnsCont/4 # Divide by number of axis
    # plt.plot(freqList, EnsCont)
    # plt.show()
    return EnsCont

# Bz = 0.003
# Wstart = 2.7e3
# Wend = 3.05e3
# Wnr = 400
# phase = np.deg2rad(60)
# Omx = 0.1
# Omy = 0.1
# freqList = np.linspace(Wstart, Wend, Wnr)

# # Cont = SingleNV_spectrum(Bz, np.pi, Omx, Omy, freqList)

# # plt.plot(freqList, Cont)
# # plt.show()

# B_vec = [0.005, 0.001, 0.002]
# MW_vec = [0.05, 0.02, 0]
# # EnsembCont = Ensemble_Spectrum(B_vec, MW_vec, phase, freqList, 0, 0)
# EnsembCont = Ensemble_Spectrum(B_vec, MW_vec, 0, freqList, 0, 0)
# EnsembCont_tilted = Ensemble_Spectrum(B_vec, MW_vec, 0, freqList, np.deg2rad(5), np.deg2rad(10))
# EnsembCont_HF = Ensemble_Spectrum_HF(B_vec, MW_vec, 0, freqList, 0, 0)
# EnsembCont_tilted_HF = Ensemble_Spectrum_HF(B_vec, MW_vec, 0, freqList, np.deg2rad(5), np.deg2rad(10))

# plt.plot(freqList, EnsembCont)
# plt.plot(freqList, EnsembCont_tilted)
# plt.plot(freqList, EnsembCont_HF)
# plt.plot(freqList, EnsembCont_tilted_HF)
# plt.show()