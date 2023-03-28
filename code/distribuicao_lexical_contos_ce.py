# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 23:12:14 2022

@author: Usuario
"""

!pip install nltk
!pip install spacy
!python3 -m spacy download pt

import os
import pandas as pd
import spacy
import nltk 
import numpy as np
from nltk import FreqDist
import re
from spacy.lang.pt.stop_words import STOP_WORDS
from string import punctuation
nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
nlp = spacy.load("pt_core_news_sm")

file = input(str('Enter the filename'))
file_plot = file[:-4]

def count_pos(pos_column, pos_name): 
    pos_column_list = [] 
    for sublist in pos_column:
        for item in sublist:
            pos_column_list.append(item)
            
    pos_flat_list = ' '.join(pos_column_list)
    pos_list_count = FreqDist(pos_flat_list.split())
    
    df_count = pd.DataFrame.from_dict([pos_list_count]).transpose().reset_index()
    df_count.columns = [pos_name, 'frequencia']     
    
    df_count = df_count.sort_values(by='frequencia', ascending = False)
    
    return df_count


df = pd.read_csv(file)

df2 = pd.DataFrame(df['transcricao_pos'])
df2.reset_index(inplace=True)
print('Contando distribuição por classes de palavras')
df2['adjetivos'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='ADJ')", str(x)))
df2['adverbios'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='ADV')", str(x)))
df2['adverbios_cs'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='ADV-KS')", str(x)))
df2['adverbios_cs_rel'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='ADV-KS-REL')", str(x)))
df2['artigos'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='ART')", str(x)))
df2['conjuncoes_coordenadas'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='KC')", str(x)))
df2['conjuncoes_subordinadas'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='KS')", str(x)))
df2['interjeicoes'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='IN')", str(x)))
df2['nomes_proprios'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='NPROP')", str(x)))
df2['numerais'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='NUM')", str(x)))
df2['substantivos'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='N')", str(x)))
df2['participios_passado'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PCP')", str(x)))
df2['palavras_denotativas'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PDEN')", str(x)))
df2['preposicoes'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PREP')", str(x)))
df2['preposicoes_plus'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PREP\\|\\+')", str(x)))
df2['pronomes_adjetivos'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PROADJ')", str(x)))
df2['pronomes_pessoais'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PROPESS')", str(x)))
df2['pronomes_substantivos'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PROSUB')", str(x)))
df2['pronomes_cs'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PRO-KS')", str(x)))
df2['pro_cs_rel'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='PRO-KS-REL')", str(x)))
df2['verbos'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='V')", str(x)))
df2['verbos_aux'] = df2['transcricao_pos'].apply(lambda x: re.findall("'(\w+)',\s(?='VAUX')", str(x)))

df2.to_csv('distribuicao_lexical.csv')

print('Gerando arquivos em csv')

adjetivos = count_pos(df2['adjetivos'].tolist(), 'adjetivos')
# adjetivos.to_csv(f'{file[:-4]}_adjetivos_dist.csv')
adverbios = count_pos(df2['adverbios'].tolist(), 'adverbios')
adverbios.to_csv(f'{file[:-4]}_adverbios_dist.csv')
# adverbios_cs = count_pos(df2['adverbios_cs'].tolist(), 'adverbios_cs')
# adverbios_cs.to_csv(f'{file[:-4]}_adverbios_cs_dist.csv')
# adverbios_cs_rel = count_pos(df2['adverbios_cs_rel'].tolist(), 'adverbios_cs_rel')
# adverbios_cs_rel.to_csv(f'{file[:-4]}_adverbios_cs_rel_dist.csv')
artigos = count_pos(df2['artigos'].tolist(), 'artigos')
artigos.to_csv(f'{file[:-4]}_artigos_dist.csv')
conjuncoes_coordenadas = count_pos(df2['conjuncoes_coordenadas'].tolist(), 'conjuncoes_coordenadas')
conjuncoes_coordenadas.to_csv(f'{file[:-4]}_conjuncoes_coordenadas_dist.csv')
conjuncoes_ subordinadas = count_pos(df2['conjuncoes_subordinadas'].tolist(), 'conjuncoes_subordinadas')
conjuncoes_subordinadas.to_csv(f'{file[:-4]}_conjuncoes_subordinadas_dist.csv')
interjeicoes = count_pos(df2['interjeicoes'].tolist(), 'interjeicoes')
interjeicoes.to_csv(f'{file[:-4]}_interjeicoes_dist.csv')
nomes_proprios = count_pos(df2['nomes_proprios'].tolist(), 'nomes_proprios')
nomes_proprios.to_csv(f'{file[:-4]}_nomes_proprios_dist.csv')
numerais = count_pos(df2['numerais'].tolist(), 'numerais')
numerais.to_csv(f'{file[:-4]}_numerais_dist.csv')
substantivos = count_pos(df2['substantivos'].tolist(), 'substantivos')
# substantivos.to_csv(f'{file[:-4]}_substantivos_dist.csv')
participios_passado = count_pos(df2['participios_passado'].tolist(), 'participios_passado')
participios_passado.to_csv(f'{file[:-4]}_participios_passado_dist.csv')
palavras_denotativas = count_pos(df2['palavras_denotativas'].tolist(), 'palavras_denotativas')
palavras_denotativas.to_csv(f'{file[:-4]}_palavras_denotativas_dist.csv')
preposicoes = count_pos(df2['preposicoes'].tolist(), 'preposicoes')
preposicoes.to_csv(f'{file[:-4]}_preposicoes_dist.csv')
preposicoes_plus = count_pos(df2['preposicoes_plus'].tolist(), 'preposicoes_plus')
preposicoes_plus.to_csv(f'{file[:-4]}_preposicoes_plus_dist.csv')  
pronomes_adjetivos = count_pos(df2['pronomes_adjetivos'].tolist(), 'pronomes_adjetivos')
pronomes_adjetivos.to_csv(f'{file[:-4]}_pronomes_adjetivos_dist.csv')
pronomes_pessoais = count_pos(df2['pronomes_pessoais'].tolist(), 'pronomes_pessoais')
pronomes_pessoais.to_csv(f'{file[:-4]}_pronomes_pessoais_dist.csv')
pronomes_substantivos = count_pos(df2['pronomes_substantivos'].tolist(), 'pronomes_substantivos')
pronomes_substantivos.to_csv(f'{file[:-4]}_pronomes_substantivos_dist.csv')
pronomes_cs = count_pos(df2['pronomes_cs'].tolist(), 'pronomes_cs')
pronomes_cs.to_csv(f'{file[:-4]}_pronomes_cs_dist.csv')
pro_cs_rel = count_pos(df2['pro_cs_rel'].tolist(), 'pro_cs_rel')
pro_cs_rel.to_csv(f'{file[:-4]}_pro_cs_rel_dist.csv')
verbos = count_pos(df2['verbos'].tolist(), 'verbos')
# verbos.to_csv(f'{file[:-4]}_verbos_dist.csv')
verbos_aux = count_pos(df2['verbos_aux'].tolist(), 'verbos_aux')
# verbos_aux.to_csv(f'{file[:-4]}_verbos_aux_dist.csv')

print('Arquivos em csv gerados! Aguarde pela verificação do lema e detalhes \n \
      morfológicos do Nilc')
#lematização e detalhamento morfológico com o Nilc
print('Verificando adjetivos')
adjetivos['adjetivos'] = adjetivos['adjetivos'].apply(lambda x: x.lower())
# adjetivos.columns = ['Unnamed: 0', 'adjetivos', 'frequencia', 'lema_spacy', 'raiz']
adjetivos_b = adjetivos['adjetivos'].tolist()



with open('delaf_pb.txt', 'r', encoding='utf-8') as file2:
    nilc = file2.read()
    nilc = re.sub(r'\w.+\.(?!A:).+', '', nilc)
    nilc = re.sub(r'(?<=:)(D|A|S)(.+)', r'\1 \2', nilc)
    nilc = re.sub(r'(?<=.)A', '', nilc)
    
    
    #nilc = re.sub(r'\w+(?=\.)', '', nilc)
    nilc = re.sub(r'\.', '', nilc)
    nilc = re.sub(r',|:', ' ', nilc)
    
    nilc = [x for x in nilc.splitlines() if len(x)]
    nilc = [x.split() for x in nilc]
  

# adjetivos['adjetivos'] = adjetivos['adjetivos'].apply(lambda x: x.lower())
# adjetivos['lema'] = adjetivos['lema'].apply(lambda x: x.lower())

#verbos_aux_b = verbos_aux['verbos_aux'].tolist()
print('Conferindo lemas e detalhes morfológicos')
lista_adjetivos = []
lista_lemas_adj = []
for x in adjetivos_b:
    #print(palavra_out)
    
    for y in nilc:
        #print(x, y[0])
        if x == y[0]:
            lista_lemas_adj.append(y[1])                  
            lista_adjetivos.append((x, y[2:]))
    
     
df_adjetivos = pd.DataFrame(lista_adjetivos, columns=['adjetivos', 'flexao'])

df_adjetivos['lema_nilc'] = lista_lemas_adj
print('Lematizando com Spacy')
df_adjetivos['lema_spacy'] = df_adjetivos['adjetivos'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))
print('Encontrando raiz morfológica de adjetivos com o NLTK')
df_adjetivos['raiz_nltk'] = df_adjetivos['adjetivos'].apply(lambda x: stemmer.stem(x))
df_adjetivos = adjetivos.merge(df_adjetivos, on='adjetivos', how='outer')

df_adjetivos_f = df_adjetivos[pd.notnull(df_adjetivos['flexao'])]

df_adjetivos_f['femininos'] = df_adjetivos_f['flexao'].apply(lambda x: re.findall(r'.(f.)', str(x)))
df_adjetivos_f['masculinos'] = df_adjetivos_f['flexao'].apply(lambda x: re.findall(r'.(m.)', str(x)))
df_adjetivos_f['fem_cont_sing'] = df_adjetivos_f['femininos'].apply(lambda x: len(re.findall(r'.(fs)', str(x))))

df_adjetivos_f['fem_cont_pl'] = df_adjetivos_f['femininos'].apply(lambda x: len(re.findall(r'.(fp)', str(x))))

df_adjetivos_f['masc_cont_sing'] = df_adjetivos_f['masculinos'].apply(lambda x: len(re.findall(r'.(ms)', str(x))))

df_adjetivos_f['masc_cont_pl'] = df_adjetivos_f['masculinos'].apply(lambda x: len(re.findall(r'.(mp)', str(x))))

df_adjetivos_f['syl_size'] = df_adjetivos_f['adjetivos'].apply(lambda x: len(re.findall(r".?uai.?|.?uão.?|.?ai.?|.?ói.?|.?ua.?|.?uo.?|.?io.?|.?ió.?|.?éi.?|.?ei.?|.?ie.?|.?ói.?|.?oi.?|.?au.?|.?ou.?|.?éu.?|.?ui.?|.?a.?|.?á.?|.?â.?|.?ã.?|.?é.?|.?ê.?|.?e.?|.?o.?|.?ô.?|.?õ.?|.?ó.?|.?ò.?|.?i.?|.?í.?",  str(x), flags = re.IGNORECASE)))

df_adjetivos_f.to_csv('adjetivos_dist.csv')
print('Plotando um lineplot de distribuição de adjetivos mais frequentes')
sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (9, 5))
plt.xticks(rotation=90)
b = sns.lineplot(data = df_adjetivos.sort_values(by='frequencia', ascending = False)[:30], x= 'adjetivos', y = 'frequencia', marker = 's',  lw='3', color = 'orange')
b.set_title(f'Adjetivos mais frequentes em {file_plot}', fontsize = 18)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Adjetivos',fontsize=15)
b.tick_params(labelsize=16)
plt.savefig(f'{file_plot}_adjetivos_lineplot.png', dpi=300)


#substantivos 
print('Verificando os substantivos no Nilc')
substantivos['substantivos'] = substantivos['substantivos'].apply(lambda x: x.lower())
# substantivos.columns = ['Unnamed: 0', 'substantivos', 'frequencia', 'lema_spacy', 'raiz']

with open('delaf_pb.txt', 'r', encoding='utf-8') as file2:
    nilc = file2.read()
    nilc = re.sub(r'\w.+\.(?!N).+', '', nilc)
    nilc = re.sub(r'(?<=:)(D|A)(.+)', r'\1 \2', nilc)
    #nilc = re.sub(r'\w+(?=\.)', '', nilc)
    nilc = re.sub(r'\.', ',', nilc)
    nilc = re.sub(r':', ',', nilc)
    # nilc = re.sub(r'N(?=\s)', '', nilc)
    
    
    nilc = [x for x in nilc.splitlines() if len(x)]
    nilc = [x.split(',') for x in nilc]
   

substantivos_b = substantivos['substantivos'].tolist()

print('Verificando lemas e detalhes morfológicos de substantivos')
lista_nomes = []
lista_lemas_nomes = []
for x in substantivos_b:
    #print(palavra_out)
    
    for y in nilc:
        #print(x, y[1:])
        if x == y[0]:                   
            lista_nomes.append((x, y[2:]))
            lista_lemas_nomes.append(y[1])
    
     
df_nomes = pd.DataFrame(lista_nomes, columns=['substantivos', 'flexao'])
df_nomes['lema_nilc'] = lista_lemas_nomes
print('Lematizando substantivos com Spacy')
df_nomes['lema_spacy'] = df_nomes['substantivos'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))
print('Inserindo raiz morfológica com NLTK')
df_nomes['raiz_nltk'] = df_nomes['substantivos'].apply(lambda x: stemmer.stem(x))
df_nomes = substantivos.merge(df_nomes, on='substantivos', how='outer')

df_substantivos_f = df_nomes[pd.notnull(df_nomes['flexao'])]

df_substantivos_f['femininos'] = df_substantivos_f['flexao'].apply(lambda x: re.findall(r'.(f.)', str(x)))
df_substantivos_f['masculinos'] = df_substantivos_f['flexao'].apply(lambda x: re.findall(r'.(m.)', str(x)))
df_substantivos_f['fem_cont_sing'] = df_substantivos_f['femininos'].apply(lambda x: len(re.findall(r'.(fs)', str(x))))

df_substantivos_f['fem_cont_pl'] = df_substantivos_f['femininos'].apply(lambda x: len(re.findall(r'.(fp)', str(x))))

df_substantivos_f['masc_cont_sing'] = df_substantivos_f['masculinos'].apply(lambda x: len(re.findall(r'.(ms)', str(x))))

df_substantivos_f['masc_cont_pl'] = df_substantivos_f['masculinos'].apply(lambda x: len(re.findall(r'.(mp)', str(x))))

df_substantivos_f['syl_size'] = df_substantivos_f['substantivos'].apply(lambda x: len(re.findall(r".?uai.?|.?uão.?|.?ai.?|.?ói.?|.?ua.?|.?uo.?|.?io.?|.?ió.?|.?éi.?|.?ei.?|.?ie.?|.?ói.?|.?oi.?|.?au.?|.?ou.?|.?éu.?|.?ui.?|.?a.?|.?á.?|.?â.?|.?ã.?|.?é.?|.?ê.?|.?e.?|.?o.?|.?ô.?|.?õ.?|.?ó.?|.?ò.?|.?i.?|.?í.?",  str(x), flags = re.IGNORECASE)))

df_substantivos_f = df_substantivos_f.query('substantivos != "tipo"')

df_substantivos_f.to_csv(f'{file_plot}_substantivos_dist.csv')

#grafico substantivos
# df_substantivos = pd.read_csv('df_substantivos_f.csv')
# df_substantivos.drop(13, axis = 0, inplace = True)
print('Plotando lineplot com distribuição dos substantivos mais frequentes')
sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (9, 5))
plt.xticks(rotation=90)
b = sns.lineplot(data = df_substantivos_f.sort_values(by='frequencia', ascending = False)[:30], x= 'substantivos', y = 'frequencia', marker = 's',  lw='3')
b.set_title(f'Substantivos mais frequentes em {file_plot}', fontsize = 18)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Substantivos',fontsize=15)
b.tick_params(labelsize=16)
plt.savefig('substantivos_lineplot.png', dpi=300)

#verbos aux

print('Verificando as entradas de verbos auxiliares no Nilc')
with open('delaf_pb.txt', 'r', encoding='utf-8') as file2:
    nilc = file2.read()
    nilc = re.sub(r'\w.+\.(?!V:).+', '', nilc)
    nilc = re.sub(r'(?<=:)([A-Z])(\d\w)', r'\1 \2', nilc) #separa tempo da flexão de pessoa
    nilc = re.sub(r'(?<=.)V', '', nilc) # apaga o V da classe
    # nilc = re.sub(r'\w+(?=\.)', '', nilc) #tira o lema
    
    
    nilc = re.sub(r'\.|:|,', ' ', nilc)

    nilc = [x for x in nilc.splitlines() if len(x)]
    nilc = [x.split() for x in nilc]

verbos_aux['verbos_aux'] = verbos_aux['verbos_aux'].apply(lambda x: x.lower())
verbos_aux_b = verbos_aux['verbos_aux'].tolist()

print('Verificando lemas e detalhes morfológicos de verbos auxiliares')
lista_verbos_aux = []
lista_lemas_vaux = []
for x in verbos_aux_b:
    #print(palavra_out)
    
    for y in nilc:
        #print(x, y[0])
        if x == y[0]:
            lista_lemas_vaux.append(y[1])
            
            lista_verbos_aux.append((x, y[2:]))                
                
                
df_vaux = pd.DataFrame(lista_verbos_aux)          
df_vaux.columns = ['verbos_aux', 'classificacao_nilc']
df_vaux['lema_nilc'] = lista_lemas_vaux
print('Lematizando com Spacy')
df_vaux['lema_spacy'] = df_vaux['verbos_aux'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))
print('Inserindo raiz morfológica do verbo auxiliar com NLTK')
df_vaux['raiz_nltk'] = df_vaux['verbos_aux'].apply(lambda x: stemmer.stem(x))
lista_class = df_vaux['classificacao_nilc'].apply(lambda x: str(x)).tolist()

lista_class = '\n'.join(lista_class)
         
print('Inserindo detalhes morfológicos dos verbos auxiliares')
lista_class = re.sub(r"'W'", "'Infinitivo'", lista_class)
lista_class = re.sub("'G'", "'Gerúndio'", lista_class)
lista_class = re.sub(r"'K'", "'Particípio'", lista_class)
lista_class = re.sub(r"'I'", "'Pretérito Imperfeito do Indicativo'", lista_class)
lista_class= re.sub(r"'J'", "'Pretérito Perfeito do Indicativo'", lista_class)
lista_class = re.sub(r"'F'", "'Futuro do Presente do Indicativo'", lista_class)
lista_class = re.sub(r"'Q'", "'Pretérito mais que Perfeito do Indicativo'", lista_class)
lista_class = re.sub(r"'S'", "'Presente do Subjuntivo'", lista_class)
lista_class = re.sub(r"'T'", "'Imperfeito do Subjuntivo'", lista_class)
lista_class = re.sub(r"'U'", "'Futuro do Subjuntivo'", lista_class)
lista_class = re.sub(r"'Y'", "'Imperativo'", lista_class)
lista_class = re.sub(r"'C'", "'Futuro do Pretérito'", lista_class)
lista_class = re.sub(r"'P'", "'Presente do Indicativo'", lista_class)


df_vaux['classificacao_verbal'] = [x for x in lista_class.splitlines()]


df_vaux = verbos_aux.merge(df_vaux, on = 'verbos_aux')

df_vaux = df_vaux.query('lema_nilc != "abraçar" and lema_nilc != "acabar" and lema_nilc != "achar" and lema_nilc != "voltar" and lema_nilc != "seriar" and lema_nilc != "continuar" and lema_nilc != "começar" and lema_nilc != "podar" and lema_nilc != "estivar" and lema_nilc != "estudar" and lema_nilc != "olhar" and lema_nilc != "saber" and lema_nilc != "ver" and lema_nilc != "vir" and lema_nilc != "dizer" and lema_nilc != "fazer"')

df_vaux['syl_size'] = df_vaux['verbos_aux'].apply(lambda x: len(re.findall(r".?uai.?|.?uão.?|.?ai.?|.?ói.?|.?ua.?|.?uo.?|.?io.?|.?ió.?|.?éi.?|.?ei.?|.?ie.?|.?ói.?|.?oi.?|.?au.?|.?ou.?|.?éu.?|.?ui.?|.?a.?|.?á.?|.?â.?|.?ã.?|.?é.?|.?ê.?|.?e.?|.?o.?|.?ô.?|.?õ.?|.?ó.?|.?ò.?|.?i.?|.?í.?",  str(x), flags = re.IGNORECASE)))

df_vaux['classificacao_clean'] = df_vaux['classificacao_verbal'].apply(lambda x: re.sub(r'.\d+\w.', '', x))
df_vaux['classificacao_clean'] = df_vaux['classificacao_clean'].apply(lambda x: re.sub(r"\s,\s(?=')", '', x))
df_vaux['classificacao_clean'] = df_vaux['classificacao_clean'].apply(lambda x: re.sub(r",(?=\s$)", '', x))
df_vaux['infinitivo'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Infinitivo', x)))
df_vaux['gerundio'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Gerúndio', x)))
df_vaux['participio'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Particípio', x)))
df_vaux['pret_perf_ind'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Pretérito Perfeito do Indicativo', x)))
df_vaux['pret_imp_ind'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Pretérito Imperfeito do Indicativo', x)))
df_vaux['futuro_pret_ind'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Futuro do Pretérito', x)))
df_vaux['pres_indicativo'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Presente do Indicativo', x)))
df_vaux['futuro_pres_ind'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Futuro do Presente do Indicativo', x)))
df_vaux['pret_mais_perf_ind'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Pretérito mais que Perfeito do Indicativo', x)))
df_vaux['pres_subj'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Presente do Subjuntivo', x)))
df_vaux['imp_subj'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Imperfeito do Subjuntivo', x)))
df_vaux['imperativo'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Imperativo', x)))
df_vaux['futuro_subj'] = df_vaux['classificacao_clean'].apply(lambda x: len(re.findall(r'Futuro do Subjuntivo', x)))
df_vaux.drop('classificacao_clean', axis = 1, inplace = True)

df_vaux.to_csv(f'{file_plot}_verbos_aux_dist.csv')

sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (9, 5))
plt.xticks(rotation=90)
b = sns.lineplot(data = df_vaux.sort_values(by='frequencia', ascending = False)[:30], x= 'verbos_aux', y = 'frequencia', marker = 's',  lw='3')
b.set_title(f'Verbos auxiliares mais frequentes em {file_plot}', fontsize = 18)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Verbos auxiliares',fontsize=15)
b.tick_params(labelsize=16)
plt.savefig(f'{file_plot}_verbos_aux_lineplot.png', dpi=300)


#verbos 

print('Conferindo as entradas verbais de verbos no Nilc')
with open('delaf_pb.txt', 'r', encoding='utf-8') as file2:
    nilc = file2.read()
    nilc = re.sub(r'\w.+\.(?!V:).+', '', nilc)
    nilc = re.sub(r'(?<=:)([A-Z])(\d\w)', r'\1 \2', nilc) #separa tempo da flexão de pessoa
    nilc = re.sub(r'(?<=.)V', '', nilc) # apaga o V da classe
    # nilc = re.sub(r'\w+(?=\.)', '', nilc) #tira o lema
    
    
    nilc = re.sub(r'\.|:|,', ' ', nilc)

    nilc = [x for x in nilc.splitlines() if len(x)]
    nilc = [x.split() for x in nilc]


verbos_b = verbos['verbos'].tolist()

print('Conferindo lemas e detalhes morfológicos dos verbos no Nilc')
lista_verbos= []
lista_lemas_verbos = []
for x in verbos_b:
    #print(palavra_out)
    
    for y in nilc:
        #print(x, y[0])
        if x == y[0]:
            lista_lemas_verbos.append(y[1])
            
            lista_verbos.append((x, y[2:]))                
                
               
df_verbos = pd.DataFrame(lista_verbos)          
df_verbos.columns = ['verbos', 'classificacao_nilc']
df_verbos['lema_nilc'] = lista_lemas_verbos
print('Lematizando verbos com Spacy')
df_verbos['lema_spacy'] = df_verbos['verbos'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))
print('Inserindo raiz morfológica de verbos com NLTK')
df_verbos['raiz_nltk'] = df_verbos['verbos'].apply(lambda x: stemmer.stem(x))
lista_class = df_verbos['classificacao_nilc'].apply(lambda x: str(x)).tolist()

lista_class = '\n'.join(lista_class)
         
print('Inserindo classificação verbal baseado no Nilc')
lista_class = re.sub(r"'W'", "'Infinitivo'", lista_class)
lista_class = re.sub("'G'", "'Gerúndio'", lista_class)
lista_class = re.sub(r"'K'", "'Particípio'", lista_class)
lista_class = re.sub(r"'I'", "'Pretérito Imperfeito do Indicativo'", lista_class)
lista_class= re.sub(r"'J'", "'Pretérito Perfeito do Indicativo'", lista_class)
lista_class = re.sub(r"'F'", "'Futuro do Presente do Indicativo'", lista_class)
lista_class = re.sub(r"'Q'", "'Pretérito mais que Perfeito do Indicativo'", lista_class)
lista_class = re.sub(r"'S'", "'Presente do Subjuntivo'", lista_class)
lista_class = re.sub(r"'T'", "'Imperfeito do Subjuntivo'", lista_class)
lista_class = re.sub(r"'U'", "'Futuro do Subjuntivo'", lista_class)
lista_class = re.sub(r"'Y'", "'Imperativo'", lista_class)
lista_class = re.sub(r"'C'", "'Futuro do Pretérito'", lista_class)
lista_class = re.sub(r"'P'", "'Presente do Indicativo'", lista_class)


df_verbos['classificacao_verbal'] = [x for x in lista_class.splitlines()]


df_verbos = verbos.merge(df_verbos, on = 'verbos')

df_verbos['syl_size'] = df_verbos['verbos'].apply(lambda x: len(re.findall(r".?uai.?|.?uão.?|.?ai.?|.?ói.?|.?ua.?|.?uo.?|.?io.?|.?ió.?|.?éi.?|.?ei.?|.?ie.?|.?ói.?|.?oi.?|.?au.?|.?ou.?|.?éu.?|.?ui.?|.?a.?|.?á.?|.?â.?|.?ã.?|.?é.?|.?ê.?|.?e.?|.?o.?|.?ô.?|.?õ.?|.?ó.?|.?ò.?|.?i.?|.?í.?",  str(x), flags = re.IGNORECASE)))
df_verbos['classificacao_clean'] = df_verbos['classificacao_verbal'].apply(lambda x: re.sub(r'.\d+\w.', '', x))
df_verbos['classificacao_clean'] = df_verbos['classificacao_clean'].apply(lambda x: re.sub(r"\s,\s(?=')", '', x))
df_verbos['classificacao_clean'] = df_verbos['classificacao_clean'].apply(lambda x: re.sub(r",(?=\s$)", '', x))
df_verbos['infinitivo'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Infinitivo', x)))
df_verbos['gerundio'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Gerúndio', x)))
df_verbos['participio'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Particípio', x)))
df_verbos['pret_perf_ind'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Pretérito Perfeito do Indicativo', x)))

df_verbos['pret_imp_ind'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Pretérito Imperfeito do Indicativo', x)))
df_verbos['futuro_pret_ind'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Futuro do Pretérito', x)))
df_verbos['pres_indicativo'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Presente do Indicativo', x)))
df_verbos['futuro_pres_ind'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Futuro do Presente do Indicativo', x)))
df_verbos['pret_mais_perf_ind'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Pretérito mais que Perfeito do Indicativo', x)))
df_verbos['pres_subj'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Presente do Subjuntivo', x)))
df_verbos['imp_subj'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Imperfeito do Subjuntivo', x)))
df_verbos['imperativo'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Imperativo', x)))
df_verbos['futuro_subj'] = df_verbos['classificacao_clean'].apply(lambda x: len(re.findall(r'Futuro do Subjuntivo', x)))

df_verbos = df_verbos[df_verbos['lema_nilc'].str.endswith('r')]

df_verbos.drop('classificacao_clean', axis = 1, inplace = True)

df_verbos.to_csv(f'{file_plot}_verbos_dist.csv')

print('Plotando gráfico das mais mais frequentes - sem stopwords')

stops_spacy = set(STOP_WORDS)
stop_words_coral = {"es", 'ah', 'hum', 'mim', 'comigo', 'ô', 'se', 'ti', 'outro', "ea", "pa'", "ni", "cum", "nimim", "p'", "pa", "di", "o'", "'", 'nan', 'pode', 'tenho'}
stop_words = stops_spacy.union(stop_words_coral)


lista_palavras = df[pd.notnull(df['normalized_utterances'])]
lista_palavras = lista_palavras['normalized_utterances'].tolist()
lista_palavras = '\n'.join(lista_palavras)

lista_palavras = FreqDist([x for x in lista_palavras.split() if x not in stop_words and x not in punctuation])

lista_palavras_df = pd.DataFrame([lista_palavras]).transpose()
lista_palavras_df.reset_index(inplace=True)
lista_palavras_df.columns = ['Palavras', 'Frequência']

sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (9, 5))
plt.xticks(rotation=90)
b = sns.lineplot(data = lista_palavras_df.sort_values(by='Frequência', ascending = False)[:30], x= 'Palavras', y = 'Frequência', marker = 's',  lw='3', color = 'orange' )
b.set_title(f'Palavras mais frequentes em {file_plot}- sem stopwords', fontsize = 18)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Palavras',fontsize=15)
b.tick_params(labelsize=16)
plt.savefig(f'{file_plot}_palavras_mais_frequentes_sem_stopwords.png', dpi=300)


print('Plotando gráfico das palavras mais frequentes - com stopwords')


stops_spacy = set(STOP_WORDS)
stop_words_coral = {"es", 'ah', 'hum', 'mim', 'comigo', 'ô', 'se', 'ti', 'outro', "ea", "pa'", "ni", "cum", "nimim", "p'", "pa", "di", "o'", "'", 'nan', 'pode', 'tenho'}
stop_words = stops_spacy.union(stop_words_coral)


lista_palavras = df[pd.notnull(df['normalized_utterances'])]
lista_palavras = lista_palavras['normalized_utterances'].tolist()
lista_palavras = '\n'.join(lista_palavras)

lista_palavras = FreqDist([x for x in lista_palavras.split() if x not in punctuation])

lista_palavras_df = pd.DataFrame([lista_palavras]).transpose()
lista_palavras_df.reset_index(inplace=True)
lista_palavras_df.columns = ['Palavras', 'Frequência']

sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (9, 5))
plt.xticks(rotation=90)
b = sns.lineplot(data = lista_palavras_df.sort_values(by='Frequência', ascending = False)[:30], x= 'Palavras', y = 'Frequência', marker = 's',  lw='3', color = 'green' )
b.set_title(f'Palavras mais frequentes em {file_plot}- com stopwords', fontsize = 18)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Palavras',fontsize=15)
b.tick_params(labelsize=16)
plt.savefig('palavras_mais_frequentes_sem_stopwords.png', dpi=300)


print('Plotando gráfico de distribuição dos lemas verbais mais frequentes')
sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (9, 5))
plt.xticks(rotation=90)
b = sns.lineplot(data = df_verbos.sort_values(by='frequencia', ascending = False)[:30], x= 'lema_nilc', ci = False,  y = 'frequencia', marker = 's',  lw='3', color = 'orange')
b.set_title(f'Lemas verbais mais frequentes em {file_plot}', fontsize = 18)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Lemas verbais',fontsize=15)
b.tick_params(labelsize=16)
plt.savefig('verbos_lemas_lineplot.png', dpi=300)


print('Plotando gráfico de distribuição das formas verbais mais frequentes')

sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (9, 5))
plt.xticks(rotation=90)
b = sns.lineplot(data = df_verbos.sort_values(by='frequencia', ascending = False)[:30], x= 'verbos', ci = False,  y = 'frequencia', marker = 's',  lw='3', color = 'orange')
b.set_title(f'Formas verbais mais frequentes em {file_plot}', fontsize = 18)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Verbos',fontsize=15)
b.tick_params(labelsize=16)
plt.savefig('verbos_lineplot.png', dpi=300)


print('Plotando gráfico com tempos e modos verbais mais frequentes')


tempo_verbos = pd.DataFrame(df_verbos.loc[:, 'infinitivo': ].sum())

tempo_verbos.reset_index(inplace=True)
tempo_verbos.columns = ['Classificação_verbal', 'Frequência']


sns.set_style('whitegrid')
plt.figure(dpi = 300, figsize = (6, 4))
plt.xticks(rotation=90)
b = sns.lineplot(data = tempo_verbos.sort_values(by = 'Frequência', ascending = False), x= 'Classificação_verbal', y = 'Frequência', marker = 's',  lw='3', color = 'orange' )
b.set_title(f'Classificação verbal mais frequente em {file_plot}', fontsize = 16)
b.set_ylabel("Frequência", fontsize = 15)
b.set_xlabel('Classificação verbal',fontsize=14)
b.tick_params(labelsize=15)
plt.savefig('classificacao_verbal_lineplot.png', dpi=300)

#plotando Word Cloud
data = df[['audio', 'normalized_utterances']]
data = data[pd.notnull(data['normalized_utterances'])]
data_list = '\n'.join(data['normalized_utterances'].tolist())
data_words = ' '.join([x for x in data_list.split() if x not in stop_words and x not in punctuation])

if os.path.isfile('figura.png'):
  print('Plotando word cloud com a figura que você escolheu')
  figure = np.array(Image.open('figura.png'))
  plt.figure(figsize=(8,6), dpi = 1000)
  wordcloud = WordCloud(mask = figure, collocations=True, background_color= 'black', max_font_size=90, width=500, height=450, colormap = 'BuPu').generate(data_words)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()
  wordcloud.to_file(f'word_cloud_black.png')
                      
  plt.figure(figsize=(8,6), dpi = 1000)
  wordcloud = WordCloud(mask = figure, collocations=True, background_color= 'white', max_font_size=90, width=500, height=450, colormap = 'inferno').generate(data_words)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()
  wordcloud.to_file(f'word_cloud_white.png')
else:
  print('Plotando word cloud')
  plt.figure(figsize=(8,6), dpi = 1000)
  wordcloud = WordCloud(collocations=True, background_color= 'black', max_font_size=90, width=500, height=450).generate(data_words)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()
  wordcloud.to_file(f'word_cloud_black.png')
    
  plt.figure(figsize=(8,6), dpi = 1000)
  wordcloud = WordCloud(collocations=True, background_color= 'white', max_font_size=90, width=500, height=450).generate(data_words)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()
  wordcloud.to_file(f'word_cloud_white.png')


