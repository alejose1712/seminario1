# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:23:50 2020

@author: Alejandro
"""
#importar librerias necesarias
#pip install mlxtend <-- utilizar antes de importar apriori
import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules

#importar dataset
data = pd.read_csv('onretail6.csv', encoding = 'unicode_escape')
data.head()

#mostrar columnas de dataset
data.columns

#cantidad de columnas y filas del dataset
data.shape

#mostrar valroes nulos
data.isnull().values.any()

data.isnull().sum()

#quitar espacios al comienzo de cada descripcion de producto
data['DESC_PROD'] = data['DESC_PROD'].str.strip() 
  
#comprobando y quitando las filas que no tenga nro de factura 
data.dropna(axis = 0, subset =['NRO_FACT'], inplace = True) 
data['NRO_FACT'] = data['NRO_FACT'].astype('str') 
  
#comprobando y quitando filas que tengan algo a credito 
data = data[~data['NRO_FACT'].str.contains('C')]

#mostrando distritos 
data.DISTRITO.unique()

#creando nueva tabla solo por distrito y con los campos necesarios
basket_JESUSM = (data[data['DISTRITO'] =="JESUS MARIA"] 
          .groupby(['NRO_FACT', 'DESC_PROD'])['CANTIDAD'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('NRO_FACT'))

#preparando data para el modelo con encode

def hot_encode(x): 
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1
    

basket_encoded = basket_JESUSM.applymap(hot_encode) 
basket_JESUSM = basket_encoded


#mostrar data final para barranco antes de modelo
basket_JESUSM.head()

#MODELO

#aplicando el algoritmo apriori
frq_items = apriori(basket_JESUSM, min_support = 0.1, use_colnames = True) 
  
#extraer association rules del algoritmo apriori
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])

#mostrar tabla de antecedentes con su respectivo consecuente y las estadisticas de cada variable que dan las assotiation rules
print(rules.head())
#rules.to_csv(r'C:\Users\Alejandro\Documents\PYTHON\prueba0\arules.csv')
