# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 22:09:10 2022

@author: Usuario
"""

import pandas as pd
import re

with open('segmentacao_ce_01.txt', 'r+', encoding='utf-8') as file:
    file = file.read()


file_l = [x for x in file.split('//')]

df = pd.DataFrame(file_l, columns= ['utterances'])

df['cl_utterances'] = df['utterances'].apply(lambda x: re.sub(r'\.\.\.|,|\.|\!|\?|–+|--|-|\(trecho.*?\)|\(\(.*?\)\)', '', x))

df['cl_utterances'] = df['cl_utterances'].str.strip()

df['cl_utterances'] = df['cl_utterances'].apply(lambda x: re.sub(r'\s+', ' ', x))

df['cl_utterances'] = df['cl_utterances'].apply(lambda x: re.sub(r'\s+', ' ', x))

df['cl_utterances'] = df['cl_utterances'].apply(lambda x: re.sub(r'$', ' //', x))

df.to_excel('alinhamento_c01.xlsx')
df.to_csv('alinhamento_c01.txt')

