import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from util import util
from BD.sqlite import BD

dirResultado = './Resultados/'
archivoResumenFitness = open(f'{dirResultado}resumen_fitness_BEN.csv', 'w')
archivoResumenTimes = open(f'{dirResultado}resumen_times_BEN.csv', 'w')
archivoResumenPercentage = open(f'{dirResultado}resumen_percentage_BEN.csv', 'w')

archivoResumenFitness.write("instance")
archivoResumenTimes.write("instance")
archivoResumenPercentage.write("instance")


graficos = False

incluye_gwo = False
incluye_psa = False
incluye_woa = False
incluye_sca = False
incluye_go = False


bd = BD()

instancias = bd.obtenerInstancias(f'''
                                  "F2","F3","F4","F5","F6","F7","F8","F9","F10","F11"
                                  ''')
print(instancias)
for instancia in instancias:
    print(instancia)
    print(instancia[1])

    blob = bd.obtenerArchivos(instancia[1])
    corrida = 1
    
    
    archivoFitness = open(f'{dirResultado}fitness_'+instancia[1]+'.csv', 'w')
    archivoFitness.write('MH,FITNESS\n')
    
    divSCA = [] 
    divGWO = [] 
    divWOA = [] 
    divPSA = []
    divGO = []

    fitnessSCA = [] 
    fitnessGWO = [] 
    fitnessWOA = [] 
    fitnessPSA = []
    fitnessGO = []

    timeSCA = []
    timeGWO = []
    timeWOA = []
    timePSA = []
    timeGO = []

    xplSCA = [] 
    xplGWO = [] 
    xplWOA = [] 
    xplPSA = []
    xplGO = []

    xptSCA = []
    xptGWO = []
    xptWOA = []
    xptPSA = []
    xptGO = []
    
    bestFitnessSCA = []
    bestFitnessGWO = []
    bestFitnessWOA = []
    bestFitnessPSA = []
    bestFitnessGO = []

    bestTimeSCA = []
    bestTimeGWO = []
    bestTimeWOA = []
    bestTimePSA = []
    bestTimeGO = []
    
    for d in blob:
        
        nombreArchivo = d[0]
        print(nombreArchivo)
        archivo = d[1]

        direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+'.csv'
        print("-------------------------------------------------------------------------------")
        util.writeTofile(archivo,direccionDestiono)
        
        data = pd.read_csv(direccionDestiono)
        
        mh = nombreArchivo.split('_')[0]
        
        if mh == "GWO" and incluye_gwo == False:
            archivoResumenFitness.write(",best,avg. fitness, std fitness")
            archivoResumenTimes.write(", min time (s), avg. time (s), std time (s)")
            archivoResumenPercentage.write(", avg. XPL%, avg. XPT%")
            incluye_gwo = True

        if mh == "GO" and incluye_go == False:
            archivoResumenFitness.write(",best,avg. fitness, std fitness")
            archivoResumenTimes.write(", min time (s), avg. time (s), std time (s)")
            archivoResumenPercentage.write(", avg. XPL%, avg. XPT%")
            incluye_go = True
            
        if mh == "PSA" and incluye_psa == False:
            archivoResumenFitness.write(",best,avg. fitness, std fitness")
            archivoResumenTimes.write(", min time (s), avg. time (s), std time (s)")
            archivoResumenPercentage.write(", avg. XPL%, avg. XPT%")
            incluye_psa = True
            
        if mh == "WOA" and incluye_woa == False:
            archivoResumenFitness.write(",best,avg. fitness, std fitness")
            archivoResumenTimes.write(", min time (s), avg. time (s), std time (s)")
            archivoResumenPercentage.write(", avg. XPL%, avg. XPT%")
            incluye_woa = True
            
        if mh == "SCA" and incluye_sca == False:
            archivoResumenFitness.write(",best,avg. fitness, std fitness")
            archivoResumenTimes.write(", min time (s), avg. time (s), std time (s)")
            archivoResumenPercentage.write(", avg. XPL%, avg. XPT%")
            incluye_sca = True
            
        problem = nombreArchivo.split('_')[1]

        iteraciones = data['iter']
        fitness     = data['fitness']
        time        = data['time']
        xpl         = data['XPL']
        xpt         = data['XPT']
        div         = data['DIV']
        
        if mh == 'PSA':
            fitnessPSA.append(np.min(fitness))
            timePSA.append(np.round(np.sum(time),3))
            xplPSA.append(np.round(np.mean(xpl), decimals=2))
            xptPSA.append(np.round(np.mean(xpt), decimals=2))
            archivoFitness.write(f'PSA,{str(np.min(fitness))}\n')
        if mh == 'GO':
            fitnessGO.append(np.min(fitness))
            timeGO.append(np.round(np.sum(time),3))
            xplGO.append(np.round(np.mean(xpl), decimals=2))
            xptGO.append(np.round(np.mean(xpt), decimals=2))
            archivoFitness.write(f'GO,{str(np.min(fitness))}\n')
        if mh == 'SCA':
            fitnessSCA.append(np.min(fitness))
            timeSCA.append(np.round(np.sum(time),3))
            xplSCA.append(np.round(np.mean(xpl), decimals=2))
            xptSCA.append(np.round(np.mean(xpt), decimals=2))
            archivoFitness.write(f'SCA,{str(np.min(fitness))}\n')
        if mh == 'GWO':
            fitnessGWO.append(np.min(fitness))
            timeGWO.append(np.round(np.sum(time),3))
            xplGWO.append(np.round(np.mean(xpl), decimals=2))
            xptGWO.append(np.round(np.mean(xpt), decimals=2))
            archivoFitness.write(f'GWO,{str(np.min(fitness))}\n')
        if mh == 'WOA':
            fitnessWOA.append(np.min(fitness))
            timeWOA.append(np.round(np.sum(time),3))
            xplWOA.append(np.round(np.mean(xpl), decimals=2))
            xptWOA.append(np.round(np.mean(xpt), decimals=2))
            archivoFitness.write(f'WOA,{str(np.min(fitness))}\n')
            
        if graficos:

            fig , ax = plt.subplots()
            ax.plot(iteraciones,div)
            ax.set_title(f'Diversity {mh} \n {problem} run {corrida}')
            ax.set_ylabel("Diversity")
            ax.set_xlabel("Iteration")
            plt.savefig(f'{dirResultado}/Graficos/Diversity_{mh}_{problem}_{corrida}.pdf')
            plt.close('all')
            print(f'Grafico de diversidad realizado {mh} {problem} ')
            
            figPER, axPER = plt.subplots()
            axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(xpl), decimals=2))+"%")
            axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(xpt), decimals=2))+"%")
            axPER.set_title(f'XPL% - XPT% {mh} \n {problem} run {corrida}')
            axPER.set_ylabel("Percentage")
            axPER.set_xlabel("Iteration")
            axPER.legend(loc = 'upper right')
            plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problem}_{corrida}.pdf')
            plt.close('all')
            print(f'Grafico de exploracion y explotacion realizado para {mh}, problema: {problem}, corrida: {corrida} ')
            print(problem)

        corrida +=1
        
        if corrida == 32:
            corrida = 1
        
        #os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')
    
    
    archivoResumenFitness.write(f'''\n''')
    archivoResumenTimes.write(f'''\n''')
    archivoResumenPercentage.write(f'''\n''')
    
    print(problem)
    archivoResumenFitness.write(f'''{problem}''')
    archivoResumenTimes.write(f'''{problem}''')
    archivoResumenPercentage.write(f'''{problem}''')
    
    if incluye_gwo:
        archivoResumenFitness.write(f''',{np.min(fitnessGWO)},{np.round(np.average(fitnessGWO),3)},{np.round(np.std(fitnessGWO),3)}''')
        archivoResumenTimes.write(f''',{np.min(timeGWO)},{np.round(np.average(timeGWO),3)},{np.round(np.std(timeGWO),3)}''')
        archivoResumenPercentage.write(f''',{np.round(np.average(xplGWO),3)},{np.round(np.average(xptGWO),3)}''')

    if incluye_go:
        archivoResumenFitness.write(f''',{np.min(fitnessGO)},{np.round(np.average(fitnessGO),3)},{np.round(np.std(fitnessGO),3)}''')
        archivoResumenTimes.write(f''',{np.min(timeGO)},{np.round(np.average(timeGO),3)},{np.round(np.std(timeGO),3)}''')
        archivoResumenPercentage.write(f''',{np.round(np.average(xplGO),3)},{np.round(np.average(xptGO),3)}''')

    if incluye_sca:
        archivoResumenFitness.write(f''',{np.min(fitnessSCA)},{np.round(np.average(fitnessSCA),3)},{np.round(np.std(fitnessSCA),3)}''')
        archivoResumenTimes.write(f''',{np.min(timeSCA)},{np.round(np.average(timeSCA),3)},{np.round(np.std(timeSCA),3)}''')
        archivoResumenPercentage.write(f''',{np.round(np.average(xplSCA),3)},{np.round(np.average(xptSCA),3)}''')
        
    if incluye_psa:
        archivoResumenFitness.write(f''',{np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)}''')
        archivoResumenTimes.write(f''',{np.min(timePSA)},{np.round(np.average(timePSA),3)},{np.round(np.std(timePSA),3)}''')
        archivoResumenPercentage.write(f''',{np.round(np.average(xplPSA),3)},{np.round(np.average(xplPSA),3)}''')
        
    if incluye_woa:
        archivoResumenFitness.write(f''',{np.min(fitnessWOA)},{np.round(np.average(fitnessWOA),3)},{np.round(np.std(fitnessWOA),3)}''')
        archivoResumenTimes.write(f''',{np.min(timeWOA)},{np.round(np.average(timeWOA),3)},{np.round(np.std(timeWOA),3)}''')
        archivoResumenPercentage.write(f''',{np.round(np.average(xplWOA),3)},{np.round(np.average(xplWOA),3)}''')
    

    # archivoResumenFitness.write(f'''\n''')
    # archivoResumenTimes.write(f'''\n''')
    # archivoResumenPercentage.write(f'''\n''')
    
    blob = bd.obtenerMejoresArchivos(instancia[1], "")
    
    
    for d in blob:

        nombreArchivo = d[4]
        archivo = d[5]

        direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+'.csv'
        util.writeTofile(archivo,direccionDestiono)
        
        data = pd.read_csv(direccionDestiono)
        
        mh = nombreArchivo.split('_')[0]
        problem = nombreArchivo.split('_')[1]

        iteraciones = data['iter']
        fitness     = data['fitness']
        time        = data['time']
        xpl         = data['XPL']
        xpt         = data['XPT']
        
        if mh == 'PSA':
            bestFitnessPSA      = fitness
            bestTimePSA         = time
        if mh == 'SCA':
            bestFitnessSCA      = fitness
            bestTimeSCA         = time
        if mh == 'GWO':
            bestFitnessGWO      = fitness
            bestTimeGWO         = time
        if mh == 'WOA':
            bestFitnessWOA      = fitness
            bestTimeWOA         = time
        if mh == 'GO':
            bestFitnessGO      = fitness
            bestTimeGO         = time
        
        os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')

    print("------------------------------------------------------------------------------------------------------------")
    #figPER, axPER = plt.subplots()
    #if incluye_gwo:
    #    axPER.plot(iteraciones, bestFitnessGWO, color="r", label="GWO")
    #if incluye_sca:
    #    axPER.plot(iteraciones, bestFitnessSCA, color="b", label="SCA")
    #if incluye_psa:
    #    axPER.plot(iteraciones, bestFitnessPSA, color="g", label="PSA")
    #if incluye_woa:
    #    axPER.plot(iteraciones, bestFitnessWOA, color="y", label="WOA")
    #if incluye_go:
    #    axPER.plot(iteraciones, bestFitnessGO, color="y", label="GO")
    #axPER.set_title(f'Coverage \n {problem}')
    #axPER.set_ylabel("Fitness")
    #axPER.set_xlabel("Iteration")
    #axPER.legend(loc = 'upper right')
    #plt.savefig(f'{dirResultado}/Best/fitness_{problem}.pdf')
    #plt.close('all')
    #print(f'Grafico de fitness realizado {problem} ')
    
    #figPER, axPER = plt.subplots()
    #if incluye_gwo:
    #    axPER.plot(iteraciones, bestTimeGWO, color="r", label="GWO")
    #if incluye_sca:
    #    axPER.plot(iteraciones, bestTimeSCA, color="b", label="SCA")
    #if incluye_psa:
    #    axPER.plot(iteraciones, bestTimePSA, color="g", label="PSA")
    #if incluye_woa:
    #    axPER.plot(iteraciones, bestTimeWOA, color="y", label="WOA")
    #if incluye_go:
    #    axPER.plot(iteraciones, bestTimeGO, color="k", label="GO" )
    #axPER.set_title(f'Time (s) \n {problem}')
    #axPER.set_ylabel("Time (s)")
    #axPER.set_xlabel("Iteration")
    #axPER.legend(loc = 'lower right')
    #plt.savefig(f'{dirResultado}/Best/time_{problem}.pdf')
    #plt.close('all')
    #print(f'Grafico de time realizado {problem} ')
    
    
    archivoFitness.close()
    
    print("------------------------------------------------------------------------------------------------------------")
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #datos = pd.read_csv(dirResultado+"/fitness_"+instancia[1]+'.csv')
    #figFitness, axFitness = plt.subplots()
    #axFitness = sns.boxplot(x='MH', y='FITNESS', data=datos)
    #axFitness.set_title(f'Fitness \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    #axFitness.set_ylabel("Fitness")
    #axFitness.set_xlabel("Metaheuristics")
    #figFitness.savefig(dirResultado+"/boxplot/boxplot_fitness_"+instancia[1]+'.pdf')
    #plt.close('all')
    #print(f'Grafico de boxplot con fitness para la instancia {instancia[1]} realizado con exito')

    #figFitness, axFitness = plt.subplots()
    #axFitness = sns.violinplot(x='MH', y='FITNESS', data=datos, gridsize=50)
    #axFitness.set_title(f'Fitness \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    #axFitness.set_ylabel("Fitness")
    #axFitness.set_xlabel("Metaheuristics")
    #figFitness.savefig(dirResultado+"/violinplot/violinplot_fitness_"+instancia[1]+'.pdf')
    #plt.close('all')
    #print(f'Grafico de violines con fitness para la instancia {instancia[1]} realizado con exito')
    
    #os.remove(dirResultado+"/fitness_"+instancia[1]+'.csv')
    
    print("------------------------------------------------------------------------------------------------------------")

archivoResumenFitness.close()
archivoResumenTimes.close()
archivoResumenPercentage.close()
        