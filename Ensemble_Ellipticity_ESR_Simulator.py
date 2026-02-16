import numpy as np
from scipy import linalg


def Hint_e(phase=0, Ox = 1, Oy=1):
    # Interaction Hamiltonian for the Electrons
    H_int = [[0, (Ox+Oy*np.exp(-1j*phase)), 0],[(Ox+Oy*np.exp(1j*phase)), 0, (Ox-Oy*np.exp(1j*phase))],[0,(Ox-Oy*np.exp(-1j*phase)), 0]]

    return np.array(H_int)

def H0_e(Bz=0, W_mw = 2.87e3):
    # Ground State Hamiltonian
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    Sz2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    #Ground State parameters (in MHz units):
    D = 2.87e3
    gamma_e = 28.024e3 # MHz/T
    
    H0 = D * Sz2 + gamma_e * Bz * Sz - W_mw * Sz2

    return np.array(H0)

def MeanPop_e(t=2, Bz=1e-3, W_mw=2.87e3, phase=0, Ox=1, Oy=1):
    # Calculate Minimum population of state 0 ater MW pulse of length t
    tlist = np.linspace(0,t,10)
    H0 = H0_e(Bz, W_mw)
    H_mw = Hint_e(phase, Ox, Oy)
    H = H0 + H_mw
    U = np.array([linalg.expm(-1j*H*t) for t in tlist]) # Evolution operators
    # Evolve State 0 and save evolution in arrays
    states = np.array([np.matmul(np.matmul(Ut,np.array([[0,0,0],[0,1,0],[0,0,0]])), Ut.conj().T) for Ut in U])

    PiPulsePop0 = np.mean(np.real(states[:, 1, 1]))
    return PiPulsePop0

def MeanPop_HF(t=10, Bz=1e-3, W_mw=2.87e3, phase=0, Ox=0.1, Oy=0.1):
    # Calculate Minimum population of state 0 ater MW pulse of length t
    tlist = np.linspace(0,t,10)
    H0 = H0_e(Bz, W_mw)
    H_mw = Hint_e(phase, Ox, Oy)
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

def SingleNV_spectrum_HF(Bz, phase, Ox, Oy, freqList):
    # Generate Spectrum from a single Nv axis. HYPERFINE INCLUDED
    # Takes:
    #   Magnetic field parallel to the axis, Bz
    #   MW pulse parameters for that axis
    #   Start, end frequencies and frequency step Nr.

    # freqList = np.linspace(Wstart, Wend, Wnr)
    
    RawVals = [MeanPop_HF(t=5, Bz=Bz, W_mw=w, phase=phase, Ox=Ox, Oy=Oy) for w in freqList]
    #print(RawVals)
    RawVals = np.array(RawVals)
    Contrast = np.array(RawVals/np.max(RawVals))
    return Contrast

def SingleNV_spectrum(Bz, phase, Ox, Oy, freqList):
    # Generate Spectrum from a single Nv axis. NO HYPERFINE
    # Takes:
    #   Magnetic field parallel to the axis, Bz
    #   MW pulse parameters for that axis
    #   Start, end frequencies and frequency step Nr.

    # freqList = np.linspace(Wstart, Wend, Wnr)
    
    RawVals = [MeanPop_e(t=5, Bz=Bz, W_mw=w, phase=phase, Ox=Ox, Oy=Oy) for w in freqList]
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

def rot_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def transform_field_toNV(B, MW, alpha=0, beta=0):
    # B_column = [[B[0]], [B[1]], [B[2]]]EN
    Bx_NV = np.sqrt(1/2)*(B[0]+B[1])
    By_NV = np.sqrt(1/2)*(-B[0]+B[1])
    B_column = [[Bx_NV], [By_NV], [B[2]]]
    MW_column = [[np.sqrt(1/2)*(MW[0]+MW[1])], [np.sqrt(1/2)*(-MW[0]+MW[1])], [MW[2]]]

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
    TlabNV1 = np.matmul(TlabNV1, Rtot.T)
    TlabNV2 = np.matmul(TlabNV2, Rtot.T)
    TlabNV3 = np.matmul(TlabNV3, Rtot.T)
    TlabNV4 = np.matmul(TlabNV4, Rtot.T)

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

    MW_in_NV_basis = {'NV_1': [], 'NV_2': [], 'NV_3': [], 'NV_4': []}
    MW_column_NV1 = np.dot(TlabNV1, MW_column)
    MW_column_NV2 = np.dot(TlabNV2, MW_column)
    MW_column_NV3 = np.dot(TlabNV3, MW_column)
    MW_column_NV4 = np.dot(TlabNV4, MW_column)
    MW_in_NV_basis['NV_1'] = [MW_column_NV1[0, 0], MW_column_NV1[1, 0], MW_column_NV1[2, 0]]
    MW_in_NV_basis['NV_2'] = [MW_column_NV2[0, 0], MW_column_NV2[1, 0], MW_column_NV2[2, 0]]
    MW_in_NV_basis['NV_3'] = [MW_column_NV3[0, 0], MW_column_NV3[1, 0], MW_column_NV3[2, 0]]
    MW_in_NV_basis['NV_4'] = [MW_column_NV4[0, 0], MW_column_NV4[1, 0], MW_column_NV4[2, 0]]

    return B_in_NV_basis, MW_in_NV_basis

def Ensemble_Spectrum(B, MW_field, MW_phase, freqList = None, tilt_x=0, tilt_y = 0):
    # Calculate Ensemble Spectrum taking only:
    #    Magnetic field:   B = [Bx, By, Bz]
    #   Microwave field:   MW_field = [Ox, Oy, 0]
    #   Microwave phase:   MW_phase = float in [0, 2*pi]

    B_inNV, MW_inNV = transform_field_toNV(B, MW_field, tilt_x, tilt_y)
    EnsCont = np.zeros(len(freqList))
    for nv in B_inNV:
        Bz = B_inNV[nv][2]
        Omx = MW_inNV[nv][0]
        Omy = MW_inNV[nv][1]
        #print(f"B:{Bz}, Ox: {Omx}, Oy: {Omy}")
        Contrast = SingleNV_spectrum(Bz, MW_phase, Omx, Omy, freqList)
        EnsCont = (EnsCont + Contrast)      

    EnsCont = EnsCont/4 # Divide by number of axis
    return EnsCont

def Ensemble_Spectrum_HF(B, MW_field, MW_phase, freqList = None, tilt_x=0, tilt_y = 0):
    # Calculate Ensemble Spectrum taking only:
    #    Magnetic field:   B = [Bx, By, Bz]
    #   Microwave field:   MW_field = [Ox, Oy, 0]
    #   Microwave phase:   MW_phase = float in [0, 2*pi]

    B_inNV, MW_inNV = transform_field_toNV(B, MW_field, tilt_x, tilt_y)
    EnsCont = np.zeros(len(freqList))
    for nv in B_inNV:
        Bz = B_inNV[nv][2]
        Omx = MW_inNV[nv][0]
        Omy = MW_inNV[nv][1]
        #print(f"B:{Bz}, Ox: {Omx}, Oy: {Omy}")
        Contrast = SingleNV_spectrum_HF(Bz, MW_phase, Omx, Omy, freqList)
        EnsCont = (EnsCont + Contrast)         
        
    EnsCont = EnsCont/4 # Divide by number of axis
    return EnsCont