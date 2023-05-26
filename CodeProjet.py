import streamlit as st
import matplotlib.pyplot as plt
import numpy as np 
import scipy.signal as sc
from  matplotlib import patches


#Affichage Streamlit

st.set_page_config(page_title=None, layout= "centered") 

st.title('Générateur de Courbes de Bode pour Circuits Électriques Passifs')

st.write("")  # Ligne vide
st.write("")  # Ligne vide

#Circuit => H(p)

def Echelle(VN,VD,VType):
   
    VN, VD, VType = np.array(VN), np.array(VD), np.array(VType)
    #definition de la matrice initiale
    NAres=[1]
    DAres=[1]
    NBres=[0]
    DBres=[1]

    #calcul du nombre de quadripoles élémentaires
    Nbr=len(VType)

    for i in range(Nbr):
        if VType[i]==0: # si quadripole de type série
            NAi=[1]
            NBi=VN[i,:]
            NCi=[0]
            NDi=[1]
            DAi=[1]
            DBi=VD[i,:]
            DCi=[1]
            DDi=[1]
        
        else: # si quadripole de type parallèle
            NAi=[1]
            NBi=[0]
            NCi=VD[i,:]
            NDi=[1]
            DAi=[1]
            DBi=[1]
            DCi=VN[i,:]
            DDi=[1] 
        
        #faire Ttemp=Tres*Ti
        NAtemp=np.convolve(NAres,np.convolve(NAi,np.convolve(DBres,DCi)))+ np.convolve(NBres,np.convolve(NCi,np.convolve(DAres,DAi)))
        NBtemp=np.convolve(NAres,np.convolve(NBi,np.convolve(DBres,DDi)))+ np.convolve(NBres,np.convolve(NDi,np.convolve(DAres,DBi)))
        DAtemp=np.convolve(DAres,np.convolve(DAi,np.convolve(DBres,DCi)))
        DBtemp=np.convolve(DAres,np.convolve(DBi,np.convolve(DBres,DDi)))  
        
        #mettre Ttemp dans Tres
        NAres=NAtemp
        DAres=DAtemp
        NBres=NBtemp
        DBres=DBtemp

    i=0 
    while NAres[i]==0:
        i=i+1
    D=NAres[i:len(NAres)]

    i=0
    while DAres[i]==0:
        i=i+1
    N=DAres[i:len(DAres)]
    return N, D

    # Saisie des valeurs VN, VD et VType
n = st.number_input("Nombre de quadripoles", min_value=1, step=1, value=2)
VN = []
VD = []
VType = []
for i in range(n):

    types = st.radio(f"Type_element {i+1}", options=["Résistance","Capacité","Inductance"])
    if types == "Résistance":
        R = st.text_input(f"R{i+1}", value="1000")
        vn = [0, float(R)]
        VN.append(vn)
        vd = [0, 1]
        VD.append(vd)
        vtype = st.radio(f"Positionnement {i+1}", options=["Série", "Parallèle"])
        VType.append(0 if vtype == "Série" else 1)
    elif types == "Capacité":
        C = st.text_input(f"C{i+1}", value="0.000001")
        vn = [0, 1]
        VN.append(vn)
        vd = [float(C), 0]
        VD.append(vd)
        vtype = st.radio(f"Positionnement {i+1}", options=["Série", "Parallèle"])
        VType.append(0 if vtype == "Série" else 1)
    elif types == "Inductance":
        L = st.text_input(f"L{i+1}", value="0.001")
        vn = [float(L), 0]
        VN.append(vn)
        vd = [0, 1]
        VD.append(vd)
        vtype = st.radio(f"Positionnement {i+1}", options=["Série", "Parallèle"])
        VType.append(0 if vtype == "Série" else 1)
    st.markdown("---")

#Tracés des courbes de Bode

def AsymbodeGain(N, D, w_min, w_max):
   
    w1 = np.logspace(w_min, w_max, 10000) # range of frequencies
    
    wout, mag, phase = sc.bode(sc.TransferFunction(N, D), w=w1) # calculate magnitudes and phases

    plt.subplot(1, 1, 1) 
    plt.semilogx(wout, mag)
    
    
    KdB = 20 * np.log10(N[0] / D[0])
    N = np.array(N) / N[0] # normalization
    D = np.array(D) / D[0]
    MN = 0
    Ntemp = np.fliplr([N])[0]
    
    while Ntemp[MN] == 0:
        MN += 1
    
    MD = 0
    Dtemp = np.fliplr([D])[0]
    
    while Dtemp[MD] == 0:
        MD += 1
    
    M = MN - MD
    N = N[:len(N) - MN]
    D = D[:len(D) - MD]
    zph = np.roots(N)
    pph = np.roots(D)
    z = np.abs(zph)
    p = np.abs(pph)
    zp = np.sort(np.concatenate([z, p])) # combine zeros and poles and sort them
    w = zp # add pseudo asymptotic pulsations
    fig = plt.figure(1)
    oldaxes = plt.axis()

    if oldaxes[0] < zp[0]:
        w = np.concatenate([[oldaxes[0]], w])
    
    if oldaxes[1] > zp[len(zp) - 1]:
        w = np.concatenate([w, [oldaxes[1]]])
    
    Gain = np.zeros(len(w)) # pre-allocation
    for i in range(len(w)):
        Gain[i] = KdB + np.sum(20 * np.log10(z[z >= w[i]])) - np.sum(20 * np.log10(p[p >= w[i]]))
        Gain[i] += 20 * np.log10(w[i]) * (np.sum(z < w[i]) - np.sum(p < w[i])) + 20 * M * np.log10(w[i])
    
    fig = plt.figure(1)
    plt.semilogx(w, Gain, 'r-.')
    plt.grid(True, which='both')
    plt.title('Courbe de Bode et courbe asymptotique : Gain')
    plt.xlabel('Fréquence (Rad/s)')
    plt.ylabel('Gain (dB)')
    
    if oldaxes[0] < zp[0]:
        Gain = Gain[1:]
    
    if oldaxes[1] > zp[len(zp) - 1]:
        Gain = Gain[:len(Gain) - 1]
    
    
    plt.semilogx(zp, Gain, 'wx')
    plt.grid(True)

def AsymbodePhase(N, D, w_min, w_max):
   
    w1 = np.logspace(w_min, w_max, 10000)
    
    wout, mag, phase = sc.bode(sc.TransferFunction(N, D),w=w1)
    
    plt.subplot(1, 1, 1) 
    
    plt.semilogx(wout, phase)
    plt.grid(True)
    
    argK = np.angle(N[0]/D[0])
    N = np.array(N) / N[0] # normalization
    D = np.array(D) / D[0]
    MN = 0
    Ntemp = np.fliplr([N])[0]
    
    while Ntemp[MN] == 0:
        MN += 1
    
    MD = 0
    Dtemp = np.fliplr([D])[0]
    
    while Dtemp[MD] == 0:
        MD += 1
    
    M = MN - MD
    N = N[:len(N) - MN]
    D = D[:len(D) - MD]
    zph = np.roots(N)
    pph = np.roots(D)
    z = np.abs(zph)
    p = np.abs(pph)
    zp = np.sort(np.concatenate([z, p])) # combine zeros and poles and sort them
    w = zp # add pseudo asymptotic pulsations

    oldaxes = plt.axis()

    if oldaxes[0] < zp[0]:
        w = np.concatenate(([oldaxes[0]], w))

    if oldaxes[1] > zp[-1]:
        w = np.concatenate((w, [oldaxes[1]]))

    Phase = np.zeros(2*len(w)-2)
    wphase = np.zeros(len(Phase))
    # print(w)
    for i in range(len(w)):
        
        if i > 0:
            wphase[2*(i-1)+1] = w[i]
            
        
        if i < len(w)-1:
            wphase[2*(i-1)+1+1] = w[i]
        
        if i > 0:
            Phase[2*(i-1)+1] = Phase[2*(i-1)-1+1]

        if i < len(w)-1:
            Phase[2*(i-1)+1+1] = np.rad2deg(argK) + M*90 + np.sum(np.rad2deg(np.angle(-zph[z > w[i]]))) - np.sum(np.rad2deg(np.angle(-pph[p > w[i]]))) + 90*np.sum(z <= w[i]) - 90*np.sum(p <= w[i])
        
    plt.semilogx(wphase, Phase, 'r-.')
    plt.title('Courbe de Bode et courbe asymptotique : Phase')
    plt.xlabel('Fréquence (Rad/s)')
    plt.ylabel('Phase (Degré)')
    plt.grid(True, which='both')
    
def BodeIndivGain(num,den):
    
    # Compute the zeros and poles of the transfer function
    zeros, poles, _ = sc.tf2zpk(num, den)
    already_processedZ = [False] * len(zeros)
    already_processedP = [False] * len(poles)
    
    # Loop through each zero and plot its contribution to the magnitude response
    for i, z in enumerate(zeros):
        
        if z == 0 : 
            
            wcas = (1)
            G = 20*np.log10(1)
            x = [1/1000, wcas, wcas*1000]
            y = [G-20*np.log10(1000),G,G+20*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Zéro en {z}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
               
            
        if z > 0 and np.isreal(z) and np.imag(z) == 0:  
            
            wcas = (z)
            G = 20*np.log10(z)
            x = [1/1000, wcas, wcas*1000]
            y = [G,G,G+20*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Zéro en {z}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
            
        if z < 0 and np.isreal(z) and np.imag(z) == 0:
            
            wcas = (-z)
            G = 20*np.log10(-z)
            x = [1/1000, wcas, wcas*1000]
            y = [G,G,G+20*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Zéro en {z}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
        
        if np.iscomplex(zeros[i]) and not already_processedZ[i] and np.real(z) == 0:

            conjugate = np.conjugate(zeros[i])
            sigma = np.real(z)
            omega = np.imag(z)
            rho = np.sqrt((sigma**2)+(omega**2))
            G = 20*np.log10(rho)
            x = [1/1000, rho, rho*1000]
            y = [G,G,G+40*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Zéros en {sigma} ± {omega}j')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(zeros)):
                if np.isclose(zeros[j], conjugate):
                    already_processedZ[j] = True
            already_processedZ[i] = True
        
        
        if np.iscomplex(zeros[i]) and not already_processedZ[i] and not np.real(z) == 0:

            conjugate = np.conjugate(zeros[i])
            sigma = np.real(z)
            omega = np.imag(z)
            rho = np.sqrt((sigma**2)+(omega**2))
            G = 20*np.log10(np.sqrt((rho**2-omega**2)**2+(4*sigma**2*omega**2)))
            x = [1/1000, rho, rho*1000]
            y = [G,G,G+40*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Zéros en {sigma} ± {omega}j')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(zeros)):
                if np.isclose(zeros[j], conjugate):
                    already_processedZ[j] = True
            already_processedZ[i] = True
            
      
    # Loop through each pole and plot its contribution to the magnitude response
    for i, p in enumerate(poles):
        
        if p == 0 : 
            
            wcas = (1)
            G = 20*np.log10(1)
            x = [1/1000, wcas, wcas*1000]
            y = [G+20*np.log10(1000),G,G-20*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Pôle en {p}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()   
        
            
        if p > 0 and np.isreal(p) and np.imag(p) == 0: 
            
            wcas = (p)
            G = 20*np.log10(1/p)
            x = [1/1000, wcas, wcas*1000]
            y = [G,G,G-20*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Pôle en {p}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()   
            
            
        if p < 0 and np.isreal(p) and np.imag(p) == 0: 
            
            wcas = (-p)
            G = 20*np.log10(1/(-p))
            x = [1/1000, wcas, wcas*1000]
            y = [G,G,G-20*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Pôle en {p}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()   
      
        
        if np.iscomplex(poles[i]) and not already_processedP[i] and np.real(p) == 0:
            
            conjugate = np.conjugate(poles[i])
            sigma = np.real(p)
            omega = np.imag(p)
            rho = np.sqrt((sigma**2)+(omega**2))
            G = 20*np.log10(rho)
            
            x = [1/1000, rho, rho*1000]
            y = [G,G,G-40*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Pôles en {sigma} ± {omega}j')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()    
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(poles)):
                if np.isclose(poles[j], conjugate):
                    already_processedP[j] = True
            already_processedP[i] = True
        
            
        if np.iscomplex(poles[i]) and not already_processedP[i] and not np.real(p) == 0:
            
            conjugate = np.conjugate(poles[i])
            sigma = np.real(p)
            omega = np.imag(p)
            rho = np.sqrt((sigma**2)+(omega**2))
            G = 20*np.log10(np.sqrt((rho**2-omega**2)**2+(4*sigma**2*omega**2)))
            x = [1/1000, rho, rho*1000]
            y = [G,G,G-40*np.log10(1000)]
            plt.semilogx(x, y, '-o',label=f'Pôles en {sigma} ± {omega}j')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Gain (dB)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Gain')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()    
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(poles)):
                if np.isclose(poles[j], conjugate):
                    already_processedP[j] = True
            already_processedP[i] = True
        

def BodeIndivPhase(num,den):
    
    # Compute the zeros and poles of the transfer function
    zeros, poles, _ = sc.tf2zpk(num, den)
    already_processedZ = [False] * len(zeros)
    already_processedP = [False] * len(poles)
    
    # Loop through each zero and plot its contribution to the magnitude response
    for i, z in enumerate(zeros):
           
        if z == 0 : 

            wcas = (-z)
            x = [1/1000, 1000]
            y = [90,90]
            plt.semilogx(x, y, '-o',label=f'Zéro en {z}')
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
        if z > 0 and np.isreal(z) and np.imag(z) == 0: 
            
            wcas = (z)
            x = [1/1000, wcas,wcas, wcas*1000]
            y = [180,180,90,90]
            plt.semilogx(x, y, '-o',label=f'Zéro en {z}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
        if z < 0 and np.isreal(z) and np.imag(z) == 0: 

            wcas = (-z)            
            x = [1/1000, wcas,wcas, wcas*1000]
            y = [0,0,90,90]
            plt.semilogx(x, y, '-o',label=f'Zéro en {z}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
        
        if np.iscomplex(zeros[i]) and not already_processedZ[i] and np.real(z) == 0:

            conjugate = np.conjugate(zeros[i])
            sigma = np.real(z)
            omega = np.imag(z)
            rho = np.sqrt((sigma**2)+(omega**2))
            
            x = [1/1000, rho,rho, rho*1000]
            y = [0,0,180,180]
            
            plt.semilogx(x, y, '-o',label=f'Zéros en {sigma} ± {omega}j')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(zeros)):
                if np.isclose(zeros[j], conjugate):
                    already_processedZ[j] = True
            already_processedZ[i] = True
        
            
        if np.iscomplex(zeros[i]) and not already_processedZ[i] and np.real(z) > 0:

            conjugate = np.conjugate(zeros[i])
            sigma = np.real(z)
            omega = np.imag(z)
            rho = np.sqrt((sigma**2)+(omega**2))
            
            x = [1/1000, rho,rho, rho*1000]
            y = [0,0,-180,-180]
            
            plt.semilogx(x, y, '-o',label=f'Zéros en {sigma} ± {omega}j')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(zeros)):
                if np.isclose(zeros[j], conjugate):
                    already_processedZ[j] = True
            already_processedZ[i] = True
        
        
        if np.iscomplex(zeros[i]) and not already_processedZ[i] and np.real(z) < 0:

            conjugate = np.conjugate(zeros[i])
            sigma = np.real(z)
            omega = np.imag(z)
            rho = np.sqrt((sigma**2)+(omega**2))
            G = 20*np.log10(np.sqrt((rho**2-omega**2)**2+(4*sigma**2*omega**2)))
            
            x = [1/1000, rho,rho, rho*1000]
            y = [0,0,180,180]
            
            plt.semilogx(x, y, '-o',label=f'Zéros en {sigma} ± {omega}j')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(zeros)):
                if np.isclose(zeros[j], conjugate):
                    already_processedZ[j] = True
            already_processedZ[i] = True
        
    # Loop through each pole and plot its contribution to the magnitude response
    for i, p in enumerate(poles):
       
        if p == 0 : 
            
            wcas = (-p)
            x = [1/1000, 1000]
            y = [-90,-90]
            plt.semilogx(x, y, '-o',label=f'Pôle en {p}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
            
            
        if p > 0 and np.isreal(p) and np.imag(p) == 0: #EXISTE PAS
            
            wcas = (p)
            x = [1/1000, wcas,wcas, wcas*1000]
            y = [0,0,90,90]
            plt.semilogx(x, y, '-o',label=f'Pôle en {p}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
           
            
        if p < 0 and np.isreal(p) and np.imag(p) == 0: 
            
            wcas = (-p)
            x = [1/1000, wcas,wcas, wcas*1000]
            y = [0,0,-90,-90]
            plt.semilogx(x, y, '-o',label=f'Pôle en {p}')
            
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()
        
        
        if np.iscomplex(poles[i]) and not already_processedP[i] and np.real(p) == 0:
            
            conjugate = np.conjugate(poles[i])
            sigma = np.real(p)
            omega = np.imag(p)
            rho = np.sqrt((sigma**2)+(omega**2))
            x = [1/1000, rho,rho, rho*1000]
            y = [0,0,-180,-180]
            plt.semilogx(x, y, '-o',label=f'Pôles en {sigma} ± {omega}j')
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()   
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(poles)):
                if np.isclose(poles[j], conjugate):
                    already_processedP[j] = True
            already_processedP[i] = True
        
        
        if np.iscomplex(poles[i]) and not already_processedP[i] and np.real(p) > 0:
            
            conjugate = np.conjugate(poles[i])
            sigma = np.real(p)
            omega = np.imag(p)
            rho = np.sqrt((sigma**2)+(omega**2))
            x = [1/1000, rho,rho, rho*1000]
            y = [0,0,180,180]
            plt.semilogx(x, y, '-o',label=f'Pôles en {sigma} ± {omega}j')
            plt.xlabel('Fréquence (rad/s)')
            plt.ylabel('Phase (Degré)')
            plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
            plt.grid(True, which='both')
            plt.legend()
            plt.show()   
            
                # Marquer le conjugué correspondant comme déjà traité
            for j in range(i+1, len(poles)):
                if np.isclose(poles[j], conjugate):
                    already_processedP[j] = True
            already_processedP[i] = True
       
        
        if np.iscomplex(poles[i]) and not already_processedP[i] and np.real(p) < 0:
           
           conjugate = np.conjugate(poles[i])
           sigma = np.real(p)
           omega = np.imag(p)
           rho = np.sqrt((sigma**2)+(omega**2))
           x = [1/1000, rho,rho, rho*1000]
           y = [0,0,-180,-180]
           plt.semilogx(x, y, '-o',label=f'Pôles en {sigma} ± {omega}j')
           
           plt.xlabel('Fréquence (rad/s)')
           plt.ylabel('Phase (Degré)')
           plt.title('Courbe(s) individuelle(s) de Bode asymptotique(s) : Phase')
           plt.grid(True, which='both')
           plt.legend()
           plt.show() 
           
               # Marquer le conjugué correspondant comme déjà traité
           for j in range(i+1, len(poles)):
               if np.isclose(poles[j], conjugate):
                   already_processedP[j] = True
           already_processedP[i] = True

#Affichage H(p) et Zplane

def AffichageFonctionTransfert(num,den):
    h_str = "H(p) = \\frac{"
    numerator_str = ""
    denominator_str = ""
    for i, c in enumerate(num):
        if c != 0:
            sign_str = "+" if c > 0 else "-"
            abs_c_str = str(abs(c)) if i != len(num) - 1 else str(abs(c)) + "\\phantom{p^0}"
            if i == len(num) - 1:
                numerator_str += abs_c_str
            elif i == len(num) - 2:
                numerator_str += f"{abs_c_str}p\\phantom{{^1}}"
            else:
                numerator_str += f"{abs_c_str}p^{len(num)-i-1}\\phantom{{^1}}"
            numerator_str += sign_str
    numerator_str = numerator_str[:-1]
    h_str += numerator_str + "}{"
    for i, c in enumerate(den):
        if c != 0:
            sign_str = "+" if c > 0 else "-"
            abs_c_str = str(abs(c)) if i != len(den) - 1 else str(abs(c)) + "\\phantom{p^0}"
            if i == len(den) - 1:
                denominator_str += abs_c_str
            elif i == len(den) - 2:
                denominator_str += f"{abs_c_str}p\\phantom{{^1}}"
            else:
                denominator_str += f"{abs_c_str}p^{len(den)-i-1}\\phantom{{^1}}"
            denominator_str += sign_str
    denominator_str = denominator_str[:-1]
    h_str += denominator_str + "}"
    st.subheader("Fonction de transfert du circuit électrique :")

    st.write("")  # Ligne vide

    st.latex(h_str)
    
def zplane(num, den):
    b = np.array(num)
    a = np.array(den)

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)

    # Calculate the maximum magnitude of poles and zeros
    max_mag = np.max(np.concatenate((np.abs(p), np.abs(z), [1])))

    # Set the radius based on the maximum magnitude
    r = 1.1 * max_mag

    # Get a figure/plot
    fig, ax = plt.subplots()

    # Create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False, color='black', ls='dashed')
    ax.add_patch(uc)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0, markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0, markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set the ticks and axis limits
    ax.axis('scaled')
    ax.axis([-r, r, -r, r])
    ticks = [-r, -r/2, 0, r/2, r]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.grid()
    plt.xlabel('Real')
    plt.title('Zéro(s) et pôle(s) de H(p) dans le plan complexe')


#Masque erreur automatique Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

#w_min et w_max
w_min = st.sidebar.slider('w_min', -30, 30, -10)
w_max = st.sidebar.slider('w_max', -30, 30, 10)

#APPEL DES FONCTIONS
num,den = Echelle(VN,VD,VType)

AffichageFonctionTransfert(num,den)
st.markdown("---")

if  len(den) == 1:
    st.error("Erreur : La fonction de transfert est un simple gain, veuillez entrer un circuit")
else :
    figure1 = AsymbodeGain(num, den, w_min, w_max)
    st.pyplot(figure1)
    st.markdown("---")

    figure2 = BodeIndivGain(num, den)
    st.pyplot(figure2)
    st.markdown("---")

    figure3 = AsymbodePhase(num, den, w_min, w_max)
    st.pyplot(figure3)
    st.markdown("---")

    figure4 = BodeIndivPhase(num, den)
    st.pyplot(figure4)
    st.markdown("---")

    figure5 = zplane(num, den)
    st.pyplot(figure5)
