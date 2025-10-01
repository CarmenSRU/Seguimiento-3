# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:48:33 2025

@author: fjose
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats 
from statsmodels.sandbox.stats.runs import runstest_1samp # para la prueba Runs 
from statsmodels.formula.api import ols #modelo lineal del ANOVA
import statsmodels.api as sm #para generar la tabla anova
from tabulate import tabulate # para tabular los dataframe
from statsmodels.stats.multicomp import pairwise_tukeyhsd# pruba de comparaciones multiples 
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
#%% Funciones necesarias. 

def test_normalityKS(data, variable): # Pruaba de Normalidad Kolmogorov-Smirnof 
    """
    data: arreglo de datos a evaluar la normalidad
    variable: string con el nombre de la variable 
    """  
    print(f"\n Análisis de normalidad por Kolmogorov-Smirnov para '{variable}'")

    # Kolmogorov-Smirnov (KS) test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f" Estadístico = {ks_stat:.4f}, p-valor = {ks_p:.4f}")

def test_normalitySW(data, variable): # Prueba de Normalizas Shapiro-Wilks 
    """
    data: arreglo de datos a evaluar la normalidad
    variable: string con el nombre de la variable 
    """
    print(f"\n Análisis de normalidad por Shapiro-Wilk para '{variable}'")
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Estadístico = {shapiro_stat:.4f}, p-valor = {shapiro_p:.4f}")
    
def random_test(residuos):
    """
    Parameters
    ----------
    residuos : Array
        DESCRIPTION: Residuos del ANOVA 

    Returns
    -------
    None.

    """
    _, p_runs = runstest_1samp(residuos, correction=False)

    print(f"Prueba de Runs: p-valor={p_runs}")
    
def test_homogeneityL(var1, var2, name1, name2): # Prueba de levene
    """
    var1 y var2: variables a las que se corroborará homocedasticidad 
    name1 y name2: strings con el nnombre de las variables
    """
    print(f"\n Análisis de homocedasticidad entre '{name1}' y '{name2}'")

    # Prueba de Levene (no asume normalidad)
    levene_stat, levene_p = stats.levene(var1, var2)
    print(f"Levene test: Estadístico = {levene_stat:.4f}, p-valor = {levene_p:.4f}")

def t_test_one(data,mu,variable): #Prueba T para una muestra
    """
    data: arreglo de datos a comparar
    mu: media poblacional o valor de referencia 
    variable: string con el nombre de la variable que se está comparando
    """
    print(f"Prueba T para una sola muestra para {variable}")
    t_stat, p_value = stats.ttest_1samp(data, mu)
    print(f"Estadístico = {t_stat:.4f}, valor_p = {p_value:.4f}")
    
def box_cox(data): #transformación depotencia   

    transformed_data, lambda_opt = stats.boxcox(data)
    return transformed_data, lambda_opt

def tukey(respuesta,factor, alfa,n_factor):
    """

    Parameters
    ----------
    respuesta : Array
        DESCRIPTION. Array con los datos de la variable respuesta
    factor : Array
        DESCRIPTION.Array con los niveles del factor 
    alfa : Float
        DESCRIPTION. Valor alfa de comparación 
    n_factor : String
        DESCRIPTION. Nombre del factor

    Returns
    -------
    None.

    """
    
    tukey = pairwise_tukeyhsd(respuesta, factor, alpha=alfa)
    print(f"Prueba Tukey para el factor {n_factor}")
    print(tukey)
    
def kruskal_W(df,Respuesta,Factor):
    """
    
    Parameters
    ----------
    df : Data_Frame
        DESCRIPTION. estructura con los datos del experimento
    Respuesta : String
        DESCRIPTION. nombre de la variable respuesta, key del dataframe
    Factor : String
        DESCRIPTION. nombre del factor, key del dataframe

    Returns
    -------
    None.

    """
    grupos_B = [df[Respuesta][df[Factor] == nivel] for nivel in df[Factor].unique()]
    stat_B, p_B = stats.kruskal(*grupos_B)
    print(f"Kruskal-Wallis para {Factor}: H = {stat_B:.4f}, p = {p_B:.4f}")
    
    
def kruskal_interaccion(df,Respuesta,Factor1,Factor2):
    """
    

    Parameters
    ----------
    df : Data_Frame
        DESCRIPTION. estructura con los datos del experimento
    Respuesta : String
        DESCRIPTION. nombre de la variable respuesta, key del dataframe
    Factor1 : String
        DESCRIPTION. nombre del factor1, key del dataframe
    Factor2 : String
        DESCRIPTION.nombre del factor12, key del dataframe

    Returns
    -------
    None.

    """
    
    df['interaccion'] = df[Factor1].astype(str) + "_" + df[Factor2].astype(str) # se genera una columana con las combinaciones entre factores

    grupos_interaccion = [df[Respuesta][df['interaccion'] == nivel] for nivel in df['interaccion'].unique()]
    stat_int, p_int = stats.kruskal(*grupos_interaccion)
    print(f"Kruskal-Wallis para la interacción {Factor1}x{Factor2} p = {p_int:.4f}")
  
#%% Cargar datos 
df = pd.read_excel("3 factores.xlsx")

#%% Supuesto de normalidad de la variable respuesta

test_normalitySW(df['conductividad'],'Variable Respuesta')

#%%

#como no es normal hacemos tranformación
data_t, lambda_opt = box_cox(df['conductividad'])
#data_t tiene los datos tranformados

test_normalitySW(data_t,'Variable Respuesta transformada')


df['conductividad_t']=data_t #guardamos los datos tranformados en el df

#%% Supuesto de Homocedasticidad para los datos tranformados


# factor 1 acido

nivel1=df[df['acido']==0]['conductividad_t']
nivel2=df[df['acido']==6]['conductividad_t']
nivel3=df[df['acido']==12]['conductividad_t']
nivel4=df[df['acido']==18]['conductividad_t']

_, levene_p = stats.levene(nivel1,nivel2,nivel3,nivel4)

print(f"Levene test: p-valor = {levene_p:.4f}")

#%% factor 2 sal

nivel1=df[df['sal']==0]['conductividad_t']
nivel2=df[df['sal']==10]['conductividad_t']
nivel3=df[df['sal']==20]['conductividad_t']

_, levene_p = stats.levene(nivel1,nivel2,nivel3)

print(f"Levene test: p-valor = {levene_p:.4f}")
#print(f"Bartlett test: p-valor = {bartlett_p:.4f}")

#%% factor 3 temp

nivel1=df[df['temperatura']==80]['conductividad_t']
nivel2=df[df['temperatura']==100]['conductividad_t']


_, levene_p = stats.levene(nivel1,nivel2)

print(f"Levene test: p-valor = {levene_p:.4f}")


#%% Realizar el ANOVA 


#%%

#df = df.sample(frac=1,ignore_index = True)
# Los ANOVAS son modelos lineales, entonces se debe ajustar el modelo
#se usa una codificación de efectos 
# la siguiente codificación incluye todas las interacciones posibles
#pero si necesitamos descartar interacciones no significativas, no es util 
#modelo = ols('conductividad_t ~ C(acido)*C(sal)*C(temperatura)', data=df).fit()

#entonces escribirmos la codificación de efectos como sigue

modelo = ols('conductividad_t ~ C(acido)+C(sal)+C(temperatura)+C(acido):C(sal)+C(sal):C(temperatura)+C(acido):C(temperatura)+C(acido):C(sal):C(temperatura)', data=df).fit()


#se geneta la tabla ANOVA, aquí es donde están los valores p para analizar
#la significancia de cada interacción y factor

anova_table = sm.stats.anova_lm(modelo, typ=3)


# Mostramos la tabla 

print(tabulate(anova_table,headers='keys',tablefmt='heavy_grid'))

#%% extracción de residuos 

df['Residuos']=modelo.resid


#%% Supuesto normalidad en los residuos 

test_normalitySW(df['Residuos'],'Residuos')

#supuesto de media cero

t_test_one(df['Residuos'],0,"Residuos")


#%% Supuesto de aleatoriedad 
random_test(df['Residuos'])

#%% gráfico de normalidad, para cuando el ANOVa es de efectos fijos 
# y por tanto la normalidad no es tan relevante. 

sm.qqplot(df['Residuos'], line='45', fit=True)
plt.title("QQ-Plot de los Residuos del ANOVA")
plt.show()

"""se comprueba los supuestos, entonces el anova es valido 
y se puede afirmar que solo la concentración del acido y la sal tinene 
efecto sobre la conductividad del metal que es limpiado. 
"""

#%% comparaciones multiples 
"""
Como ya sabemos con certeza que la concentración de acido y sal
tiene efecto sobre la conductividad del metal qeu se limpia
y además los factores de acido y sal tienen 3 o más niveles 
se debe hacer una prueba de comparaciones multiple para determinar
si hay diferencia entre los niveles de cada factor 
"""

# Prueba de Tukey para acido 
tukey(df["conductividad"], df["acido"],0.05,"Ácido")

#prueba Tukey para sal 

tukey(df["conductividad"], df["sal"],0.05,"Sal")

#%% Kruskal Wallis Test 

"""
supongamos que el anterior ejercicio no cumplió con los requisitos para
un ANOVA, entonces se debe hacer una prueba de Kruskal Wallis
Como las pruebas no paramétricas son robustas ante las distribuciones
aunque el ejercicio anterior cumplió en general con todo, el resultado 
de esta prueba debería ser similar a lo ya obtenido.
"""
#para el acido
kruskal_W(df,'conductividad','acido') # notese que se usa el df sin tranformar 

#para la sal
kruskal_W(df,'conductividad','sal')

#para la temperatura
kruskal_W(df,'conductividad','temperatura')

#%%
"""
El resultado es similar, la temperatura no tiene efecto significativo, 
mientras que el acido y la sal sí. Ahora bien, no podemos saber con esta prueba 
si existen interacciones. Podríamos recurrir al gráfico para saberlo, pero se puede
evaluar una a una las interacciones 
"""

kruskal_interaccion(df,'conductividad','acido','sal')

"""
Segun este resultado, la interacción sería significativa, pero no observamos eso en el 
gráfico, y tampoco se observó en el ANOVA, este podría ser un falso positivo. 
la prueba kruskal wallis tiene como condición que haya al menos 5 datos en cada grupo y hay 4

"""