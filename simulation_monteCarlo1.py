import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.setrecursionlimit(10000)

#Exercice 1##########
#####################
#####################
#Fonction qui simule un Bernouilli de paramètre p
def simX(p):
    return int(np.random.random()<p) #int(.) permet de convertir le booléen np.random.random()<p en 0 ou 1



####Version lente avec appel de simX
NMonteCarlo = 1000 #### nombres d'échantillons pour Monte-Carlo
estimation = 0 ### variable déstinée à recevoir le résultat
p = 0.5 #proba choisie (arbitraire)
for i in range(NMonteCarlo):
    estimation += simX(p)
print("Exercice 1: estimation de p: " + str(estimation/NMonteCarlo))#str(.) converti une valeur en string 

#####Version rapide "vectorialisée"


estimation = 0 #on remet estimation à 0
p = 0.5
estimation = sum((np.random.random(NMonteCarlo)<p).astype(int)) 
"""
Rque: np.random.random(NMonteCarlo)<p renvoie un vecteur de booléen true ou false si la composante correspondante dans np.random.random(NMonteCarlo)est plus petite ou plus grande que p
La méthode .astype(int) permet de convertir ce vecteur en vecteur de 0 et de 1.
"""
print("Exercice 1: estimation de p (rapide): "+str(estimation/NMonteCarlo))


#Exercice 2##########
#####################
#####################



def poissonProba(lamb,k):#fonction qui calcul la proba qu'une Poisson de paramètre lamb renvoie la valeur k
    return np.exp(-lamb)*pow(lamb,k)/np.math.factorial(k)


def simPoisson(lamb):#fonction qui simule une loi de Poisson de paramètre lamb
    p=poissonProba(lamb,0) 
    U=np.random.random()
    I = 0
    while(U>p):
        p+=poissonProba(lamb,I)
        I +=1
    return I
    

#Partie Monte-Carlo
NMonteCarlo = 1000
estimationMoyenne = 0 
for i in range(NMonteCarlo):
    estimationMoyenne += simPoisson(2)
estimationMoyenne = estimationMoyenne / NMonteCarlo
print("Exercice 2: estimation de la moyenne: "+str(estimationMoyenne))


estimationVariance = 0
for i in range(NMonteCarlo):
    estimationVariance += pow(simPoisson(2)-estimationMoyenne,2)
estimationVariance = estimationVariance / NMonteCarlo
print("Exercice 2: estimation de la variance: "+str(estimationVariance))

#Exercice 3

def simDisqueUnite(): #Méthode de rejet pour simuler une uniforme sur le disque
    U1=np.random.random()
    U2=np.random.random()
    while (U1**2+U2**2>1):
        U1=np.random.random()
        U2=np.random.random()
    return [U1,U2]
print("Exercice 3: réalisation d'un VA sur le cercle unité:"+str(simDisqueUnite()))

#Exercice 4

def densiteLaplace(x,lamb):
    return np.exp(-lamb*abs(x))*lamb/2
    
def invRepartition(p,lamb): #Fonction de repartition inverse (on ne vérifie pas que l'argument est dans ]0,1[, on fait confiance à l'utilisateur (on devrait pas !) )
    if(p<=0.5):
        return np.log(2*p)/lamb
    else:
        return -np.log(2*(1-p))/lamb
def simLaplace(lamb):
    return invRepartition(np.random.random(),lamb) #On simule une uniforme, on retourne l'image de celle-ci par la fonction de repartition inverse


estimationVariance = 0
for i in range(NMonteCarlo):
    estimationVariance += pow(simLaplace(4),2)
estimationVariance = estimationVariance / NMonteCarlo
print("Exercice 4: estimation de la variance: " +str(estimationVariance))
#Exercice 5

def densiteGaussienne(x):
    return np.exp(-x*x/2)/np.sqrt(2*np.pi)

t=np.linspace(-6,6,100)
plt.plot(t,densiteGaussienne(t))
plt.plot(t,1.4*densiteLaplace(t,1))
plt.show()

def rejetGaussienne():
    i=1
    L=simLaplace(1)
    U=1.4*densiteLaplace(L,1)*np.random.random()
    while(U>densiteGaussienne(L)):
        i=i+1
        L=simLaplace(1)
        U=1.4*densiteLaplace(L,1)*np.random.random()
    return L,i
    
    
NMonteCarlo = 1000
estimationIteration= 0 
for i in range(NMonteCarlo):
    estimationIteration += simPoisson(2)
estimationIteration = estimationIteration / NMonteCarlo
print("Exercice 5: estimation du nombre d'itération: "+str(estimationIteration))


NombreDeSimu = 1000000
timer=time.perf_counter()
for i in range(NombreDeSimu):
    rejetGaussienne()
print("Temps d'execution rejet : "+str(time.perf_counter()-timer)+ "secondes") #str(.) converti une valeur en string 
timer=time.perf_counter()
for i in range(NombreDeSimu):
    np.random.normal()
print("Temps d'execution numpy : "+str(time.perf_counter()-timer)+ "secondes")  
timer=time.perf_counter()
np.random.normal(NombreDeSimu)
print("Temps d'execution numpy vectorialisée : "+str(time.perf_counter()-timer)+ "secondes")  
tempSimu = 0
tempSimuNumpy = 0




#Exercice 6

#Version 1 en largeur: pas très efficace pour estimer la longueur d'une réaction (ie la profondeur de l'arbre)
def simReaction(p):
    maxIt = 100 ##nbre maximum d'itération 
    it = 0 #compteur d'itération
    NNeutron = 1 #nombre de neutron restants dont on doit simuler le nombre d'enfant
    while NNeutron > 0 and it < maxIt: #tant qu'il reste des neutrons et que l'on a pas atteint la durée maximal, on continue
        U =  np.random.random() #simulation du nombre d'enfants des neutrons restants
        NNeutron = NNeutron - 1
        if U < p:
            NNeutron +=2
        it += 1
    return it

#Version 2 en profondeur: on simule recursivement l'arbre de Galton-Watson non pas en largeur mais en profondeur. On s'arrête dès qu'une branche atteint la profondeur maximale fixée.
def recursiveSimReaction(prop,profondeur):
    
    if profondeur < 500: #on vérifie que l'on a pas atteint la profondeur maximum
        U =  np.random.random() #on simule le nombre d'enfant du neutron courant
        if U > prop:
            return 1
        else:
            leftDepth = recursiveSimReaction(prop,profondeur+1) #si on a deux enfants, on ré-appel la fonction pour simuler l'arbre issu du premier enfant 
            rightDepth = 0
            if leftDepth+profondeur < 500: #si l'arbre issu du premier enfant n'atteint pas la profondeur maximal, on simule celui du second enfant
                rightDepth = recursiveSimReaction(prop,profondeur+1)
            return max(leftDepth,rightDepth)+1
    else:
        return 1
            
            
        
            
####estimation de la longueur de la réaction (sur l'événement que cette longueur est plus petite que 500)  en fonction de la proportion de 235
    
t = np.linspace(0,1,100)
reaction = np.zeros(len(t))
for i in range(len(t)):
    print(i)
    mc = 0
    for j in range(1000):
        mc += recursiveSimReaction(t[i],0)
    reaction[i] = mc/1000
plt.plot(t,reaction)
plt.show()

#on remarque une "rupture" en 0.5
