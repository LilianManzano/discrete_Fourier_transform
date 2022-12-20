from PIL import Image
import numpy as np
import cmath
import time

#------------------------------------------------------------------------------------------------------------------
#FONTIONS TRANSFORMEE DIRECT
#------------------------------------------------------------------------------------------------------------------
#fonction qui calcule la transformée de fourier direct sur une ligne
#prend en paramêtre un tableau une dimension
def transformeeDirect1D(m):
    matriceResultat=np.copy(m)
    for i in range(len(m)):
        somme=0
        for k in range(len(m)):
            somme+=m[k]*(cmath.exp(-(2*1j*cmath.pi*i*k)/len(m)))
        matriceResultat[i]=somme
    return matriceResultat
#fonction qui calcule la transformée de fourier direct inverse sur une ligne
#prend en paramêtre un tableau une dimension
def transformeeDirect1Dinverse(m):
    matriceResultat=np.copy(m)
    for i in range(len(m)):
        somme=0
        for k in range(len(m)):
            somme+=m[k]*(cmath.exp((2*1j*cmath.pi*i*k)/len(m)))
        matriceResultat[i]=somme
    return matriceResultat

#fonction qui calcule la transformée de fourier direct sur toute une matrice
#prend en paramêtre la matrice
def transformeeDirect2D(m):
    matriceResultat=np.zeros((m.shape[0],m.shape[1]),dtype=complex)
    for i in range(m.shape[0]):#application de la transformee direct 1D sur les lignes
        matriceResultat[i] = transformeeDirect1D(m[i])
    for i in range(m.shape[1]):#application de la transformee direct 1D sur les colonnes
        matriceResultat[:,i] = transformeeDirect1D(matriceResultat.transpose()[i]).transpose()
    return matriceResultat
#fonction qui calcule la transformée de fourier direct inverse sur toute une matrice
#prend en paramêtre la matrice
def transformeeDirect2Dinverse(m):
    matriceResultat=np.zeros((m.shape[0],m.shape[1]),dtype=complex)
    for i in range(m.shape[0]):#application de la transformee inverse direct 1D sur les lignes
        matriceResultat[i] = transformeeDirect1Dinverse(m[i])
    for i in range(m.shape[1]):#application de la transformee inverse direct 1D sur les colonnes
        matriceResultat[:,i] = transformeeDirect1Dinverse(matriceResultat.transpose()[i]).transpose()
    return matriceResultat


#------------------------------------------------------------------------------------------------------------------
#FONTIONS TRANSFORMEE RAPIDE
#------------------------------------------------------------------------------------------------------------------
#fonction qui calcule la transformée de fourier rapide sur une ligne
#prend en paramêtre un tableau une dimension
def transformeeRapide1D(m):
    N=len(m)            #on recupere la taille du tableau
    if N<=1:            #si la taille est égal à 1 on retourne l'unique valeur
        return m
    else:
        pair=transformeeRapide1D(m[0::2])   #appel récursif sur un tableau contenant que les éléments en position pair
        impair=transformeeRapide1D(m[1::2]) #appel récursif sur un tableau contenant que les éléments en position impair
        matriceResultat=np.zeros(N).astype(np.complex64)
        for i in range(0, N//2):#boucle permettant les calculs et la reconstitution du tableau
            matriceResultat[i] = pair[i]+cmath.exp(-2j*cmath.pi*i/N)*impair[i]          #application de la formule pour la première moitié (voir rapport)
            matriceResultat[i+N//2] = pair[i]-cmath.exp(-2j*cmath.pi*i/N)*impair[i]     #application de la formule pour la seconde moitié (voir rapport)
        return matriceResultat
#fonction qui calcule la transformée de fourier rapide inverse sur une ligne
#prend en paramêtre un tableau une dimension
def transformeeRapide1Dinverse(m):
    N=len(m)            #on recupere la taille du tableau
    if N<=1:            #si la taille est égal à 1 on retourne l'unique valeur
        return m
    else:
        pair=transformeeRapide1Dinverse(m[0::2])    #appel récursif sur un tableau contenant que les éléments en position pair
        impair=transformeeRapide1Dinverse(m[1::2])  #appel récursif sur un tableau contenant que les éléments en position impair
        matriceResultat=np.zeros(N).astype(np.complex64)
        for i in range(0, N//2):#boucle permettant les calculs et la reconstitution du tableau
            matriceResultat[i] = pair[i]+cmath.exp(2j*cmath.pi*i/N)*impair[i]           #application de la formule pour la première moitié (voir rapport)
            matriceResultat[i+N//2] = pair[i]-cmath.exp(2j*cmath.pi*i/N)*impair[i]      #application de la formule pour la seconde moitié (voir rapport)
        return matriceResultat

#fonction qui calcule la transformée de fourier rapide sur toute une matrice
#prend en paramêtre la matrice
def transformeeRapide2D(m):
    matriceResultat=np.zeros((m.shape[0],m.shape[1]),dtype=complex)
    for i in range(m.shape[0]):#application de la transformee rapide 1D sur les lignes
        matriceResultat[i] = transformeeRapide1D(m[i])
    for i in range(m.shape[1]):#application de la transformee rapide 1D sur les colonnes
        matriceResultat[:,i] = transformeeRapide1D(matriceResultat.transpose()[i]).transpose()
    return matriceResultat
#fonction qui calcule la transformée de fourier rapide inverse sur toute une matrice
#prend en paramêtre la matrice
def transformeeRapide2Dinverse(m):
    matriceResultat=np.zeros((m.shape[0],m.shape[1]),dtype=complex)
    for i in range(m.shape[0]):#application de la transformee inverse rapide 1D sur les lignes
        matriceResultat[i] = transformeeRapide1Dinverse(m[i])
    for i in range(m.shape[1]):#application de la transformee inverse rapide 1D sur les colonnes
        matriceResultat[:,i] = transformeeRapide1Dinverse(matriceResultat.transpose()[i]).transpose()
    return matriceResultat


#------------------------------------------------------------------------------------------------------------------
#FONTIONS AUTRES
#------------------------------------------------------------------------------------------------------------------
#fonction qui retourne la matrice avec les modules calculés pour chaque nombre
#prend en paramêtre la matrice
def moduleMatrice(m):
    matriceResultat = np.zeros((m.shape[0], m.shape[1]), dtype=complex)
    for i in range(m.shape[0]):
        for k in range(m.shape[1]):
            matriceResultat[i][k]=cmath.sqrt(m[i][k].real*m[i][k].real+m[i][k].imag*m[i][k].imag)
    return matriceResultat
#fonction qui recadre une matrice entre 2 nombres donnés
#prend en paramêtre la matrice, le minimum puis le maximum que l'on veut pour le recadrage
def recadrage(m,a,b):
    matriceResultat = np.zeros((m.shape[0], m.shape[1]), dtype=complex)
    imin=np.amin(m)
    imax=np.amax(m)
    for i in range(m.shape[0]):
        for k in range(m.shape[1]):
            matriceResultat[i][k]=((m[i][k]-imin)*((b-a)/(imax-imin)))+a
    return matriceResultat






#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#MAIN
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #choix création/importation de matrice
    choix=int(input("Entrer 1 pour renseigner le lien d'une image, "
                    "\nentrer 2 pour renseigner le lien d'un fichier numpy, "
                    "\nentrer 3 pour créer votre matrice : "))

    if choix == 1 : #code pour importer une image, la transformer en niveu de gris, puis en mettre les valeurs dans un tableau
        lien=input("Entrer le lien absolue de l'image : ")
        imgImport = Image.open(r""+lien+"")
        imgGray = imgImport.convert('L')
        matrice = np.array(imgGray)

    elif choix == 2 : #code pour importer un tableau d'un fichier numpy (.npy), on considère que le tableau est 2D (déjà en niveau de gris)
        lien = input("Entrer le lien absolue du fichier texte : ")
        matrice = np.load(lien)

    elif choix == 3 : #code pour créer sa propre matrice, on créer un tableau 2D (déjà en niveau de gris)
        nbLigne=int(input("Entrer le nombre de ligne de votre matrice : "))
        nbColonne=int(input("Entrer le nombre de colonne de votre matrice : "))
        matrice = np.zeros((nbLigne,nbColonne),dtype=complex)
        for i in range(nbLigne):
            ligne=input(f"Entrer la ligne {i+1}, les nombres séparés par des espaces :")
            for j in range(nbColonne):
                matrice[i][j]=ligne.split()[j]

    #affichage de la matrice créée/importée
    print("\nVous avez désormais votre matrice :")
    print(matrice)

    #choix sur le calcul à effectuer sur la matrice
    choix=int(input("\nEntrer 1 pour appliquer la transformée de fourier directe discrète, "
                    "\n2 pour appliquer la transformée de fourier direct inverse, "
                    "\n3 pour appliquer la transformée de fourier rapide discrète,"
                    "\net 4 pour appliquer la transformée de fourier inverse rapide : "))


    #verification si la matrice est de dimension 1 ou de dimension 2
    nbLigne=matrice.shape[0]
    nbColonne=matrice.shape[1]
    dimension=0
    if nbLigne == 1 or nbColonne == 1 :
        dimension=1
    else :
        dimension=2
    matriceResultat = np.zeros((nbLigne, nbColonne), dtype=complex)

    # début du chrono
    tempsDepart = time.time()




    if choix == 1 :#transformée de fourier direct
        matrice = recadrage(matrice, 0, 1)
        if dimension == 1:
            if nbLigne == 1:
                matriceResultat[0] = transformeeDirect1D(matrice[0])
            else:
                matriceResultat[:,0] = transformeeDirect1D(matrice.transpose()[0]).transpose()
        if dimension == 2:
            matriceResultat = transformeeDirect2D(matrice)

    elif choix == 2 :#transformée de fourier direct inverse
        if dimension == 1:
            if nbLigne == 1:
                matriceResultat[0] = transformeeDirect1Dinverse(matrice[0])
            else:
                matriceResultat[:,0] = transformeeDirect1Dinverse(matrice.transpose()[0]).transpose()
        if dimension == 2:
            matriceResultat = transformeeDirect2Dinverse(matrice)

    elif choix == 3 :#transformée de fourier rapide
        matrice = recadrage(matrice, 0, 1)
        if dimension == 1:
            if nbLigne == 1:
                matriceResultat[0] = transformeeRapide1D(matrice[0])
            else:
                matriceResultat[:,0] = transformeeRapide1D(matrice.transpose()[0]).transpose()
        if dimension == 2:
            matriceResultat = transformeeRapide2D(matrice)

    elif choix == 4:#transformée de fourier rapide inverse
        if dimension == 1:
            if nbLigne == 1:
                matriceResultat[0] = transformeeRapide1Dinverse(matrice[0])
            else:
                matriceResultat[:,0] = transformeeRapide1Dinverse(matrice.transpose()[0]).transpose()
        if dimension == 2:
            matriceResultat = transformeeRapide2Dinverse(matrice)




    # fin du chrono + affichage du temps
    tempsFin = time.time()
    temps = tempsFin-tempsDepart
    print("\nLe calcul a été fait en "+str(temps)+" secondes.")

    #on affiche la matrice résultat
    print("La matrice résultat a été sauvegardée en tant qu'image et fichier texte, en voici le résultat : \n")
    print(matriceResultat)

    #sauvegarde en fichier numpy (utilisable pour faire la transformée inverse)
    np.save("fileResult", matriceResultat)

    #sauvegarde dans un fichier texte (juste pour voir les valeurs)
    np.savetxt("fileResult.txt", matriceResultat)

    #opérations nécessaires pour obtenir une bonne image
    matriceResultat=moduleMatrice(matriceResultat)
    matriceResultat=recadrage(matriceResultat,1,255)
    matriceResultat=matriceResultat.real.astype('uint8')
    #sauvegarde du module dans un fichier texte
    np.savetxt("fileResultModulus.txt", matriceResultat)

    #sauvegarde en tant qu'image
    imageFinale=Image.fromarray(matriceResultat)
    imageFinale.save("imageResult.png")
