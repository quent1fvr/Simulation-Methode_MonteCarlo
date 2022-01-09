
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#####Paramètres#####
sig=2
borne=100
v=0.3410849530281127520293850364917529310616602487743391295306503143#Valeur de reférence
#######################


t=np.linspace(-borne/10,borne/10,1000) #grille des points pour le graphe
plt.plot(t,t*t*t*np.exp(-t**2+t/4)) # graphique de la fonction à intégrer 
plt.plot(t,np.exp(-t*t/(2*sig))/np.sqrt(sig*np.pi*2)) #graphique de la densité d'échantillonage
plt.show() # trace les grahpe. Remarquer que les deux fonctions "s'ajuste" mieux qu'avec la densité de la loi uniforme


####### Définition de la fonction que l'on souhaite intégrer######
def funToIntegrate(t):
    return t*t*t*np.exp(-t**2+t/4)
    
##### Définition de la densité d'échantillonage #######
def gaussian(t):
    return np.exp(-t*t/(2*sig))/np.sqrt(sig*np.pi*2)
   
NMC=np.linspace(10,100000,1000)##### Grille du nombre d'évaluation de Monte-Carlo: on commence par un MC avec 10 pts d'échantillonage, on finit avec beaucoup plus, pour voir l'évolution de l'érreur.
res=np.zeros(len(NMC)) # liste (initialement remplie de zéros) destinée à recevoir les résultats pour MC naïf







for i in range(len(NMC)): #à chaque itération de cette boucle, on effectue une estimation par Monte-Carlo avec un échentillon de taille différente
    print(i)#Juste pour voir où on en est
    N=2*borne*np.random.random(int(NMC[i]))-borne #échantillonage suivant la loi uniforme sur [-100,100] #Rque: on échantillonne toutes les VA d'un coup, c'est bien plus efficace en python
    res[i]=sum(2*borne*funToIntegrate(N))/NMC[i] #on ajoute le résultat à la somme
    




res2=np.zeros(len(NMC))# liste (initialement remplie de zéros) déstinée à recevoir les résultats pour éch préférentiel
for i in range(len(NMC)): #idem que précédement. Différence: on échantillone avec une Gaussienne
    print(i)
    N=np.sqrt(sig)*np.random.normal(0,1,int(NMC[i])) #Rque: on échantillonne toutes les VA d'un coup, c'est bien plus efficace en python
    res2[i] = sum((abs(N)<borne).astype(int)*funToIntegrate(N)/gaussian(N))/NMC[i]
print(res2)
plt.plot(NMC,abs(res-v))
plt.plot(NMC,abs(res2-v))
plt.show()




n=50

#La grille de Ising est notée E ou S

def val(E,i,j,n): #Cette fonction évalue la valeur du spin en un point de la grille (ça permet de détecter les bords plus facilement)
    if(i<0 or j<0 or j >= n or i >= n):
        return 0
    else:
        return E[i][j]

def absMagnet(E,n): #calcul la magnétisation moyenne de la grille
    s=0
    for i in range(n):
        for j in range(n):
            s+=E[i][j]
    return abs(s/(n*n))
    
    
def MHStep(E,en):#Cette fonction prend une grille en entrée, la modifie suivant la dynamique de l'exercice 3 et accepte ou non le nouvel état. 
    #Attention: Cette fonction retourne la valeur de l'energie de la nouvelle configuration et pas le nouvelle grille (la grille est juste modifiée par l'appel de cette fonction)
    
    i=int(np.floor(n*np.random.random()))
    j=int(np.floor(n*np.random.random()))
    
    if(np.log(np.random.random()) < -1*beta*np.sign(E[i][j])*(val(E,i-1,j,n)+val(E,i+1,j,n)+val(E,i,j-1,n)+val(E,i,j+1,n))):
        
        E[i][j]=-E[i][j]
        return en+2*E[i][j]/(n*n)
    else:
        return en

def simIsing(b): #simulation du modèle d'ising  à inverse de température b
    global beta #on déclare beta comme état globale afin d'y avoir accès depuis la fonction MHStep
    beta=b #on assigne ensuite beta à la valeur voulue
    ####Création d'une grille aléatoire#########
    S=np.zeros((n,n)) 
    for i in range(n):
        for j in range(n):
            if np.random.random()>0.5:
                S[i][j]=1
            else:
                S[i][j]=-1
   ##############################################
   #Puis on applique Metropolis 
    for i in range(2000):
        MHStep(S,0)
    #plt.imshow(S) #permet d'afficher les grilles si on a envie (à décommenter le cas échéant)
    #plt.show()
    return absMagnet(S,n)


#On trace une estimation de la magnétisation moyenne en fonction de beta (difficile de voir quelque chose à ce niveau de performance ou alors faut attendre longtemps.)
t=np.linspace(0.1,1,20)

NMC=100
res=np.zeros(len(t))
for i in range(len(t)):
    for j in range(NMC):
        res[i]+=simIsing(t[i])
    res[i]=res[i]/NMC
plt.plot(t,res)
plt.show()

###### Animation ######
#à décommenter si on veut une animation
"""
beta = 2
S=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if np.random.random()>0.5:
            S[i][j]=1
        else:
            S[i][j]=-1      
MHStep(S,0)

fig, (a0,a1)=plt.subplots(2,1,gridspec_kw={'height_ratios':[3, 1]})
ln = a0.imshow(S,vmin=-1)


t=np.linspace(0,10,100)
ln2,=a1.plot([0],[absMagnet(S,n)])

xdata, ydata = [], []
print("here")
def update(frame):
    S = ln.get_array()
    a,b=ln2.get_data()
    en=MHStep(S,b[-1])
    xdata.append(frame)
    ydata.append(en)
    ln2.set_data(xdata,ydata)
    #a1.relim()
    #a1.autoscale(enable=True)
    return ln,ln2,
def init():
    a1.set_xlim(0,100000)
    a1.set_ylim(-1, 1)
    return ln,ln2,

ani = FuncAnimation(fig, update,frames=np.linspace(0,100000,100000),init_func=init, blit=True, interval = 1)

plt.show()

"""
