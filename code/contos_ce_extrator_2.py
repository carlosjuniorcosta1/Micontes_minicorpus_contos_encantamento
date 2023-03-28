# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 00:00:45 2022

@author: Usuario
"""

import pandas as pd
import os
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import FreqDist
from time import sleep

df = '\n'.join([x for x in os.listdir() if x.startswith('df') and x.endswith('.csv')])
df = pd.read_csv(df)

nomes_transcricoes = dict(zip(df['audio'], df['titulo']))
while True:
    file_1 = input(
        f'Digite  o código de um dos contos de encantamento ou aperte enter para todos \n{dict(zip(df["audio"], df["titulo"]))} >>>\n')
    if len(file_1) > 0:
        if file_1 not in nomes_transcricoes.keys():
            print('Código inexistente. \n Por favor, digite um código que conste na lista abaixo:')
            sleep(2)
        if file_1 in nomes_transcricoes.keys():
            print(f'Você escolheu {file_1} - {nomes_transcricoes[file_1]}')
            df = df.query('audio in @file_1')
            break
    if len(file_1) == 0:
        print('Você escolheu todos os contos')
        break

if len(file_1) > 0:

    contagem_palavras = '\n'.join(df['transcricao_limpa'].tolist())
    contagem_palavras = contagem_palavras.lower()
    contagem_palavras = FreqDist(contagem_palavras.split())
    contagem_palavras = pd.DataFrame([contagem_palavras]).transpose()
    contagem_palavras.reset_index(inplace=True)
    contagem_palavras.columns = ['palavra', 'frequência']
    contagem_palavras.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.lineplot(data=contagem_palavras[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='orange')
    plt.xticks(rotation=90)
    a.set_title(f'Palavras mais frequentes em {df["audio"][0]}', fontsize=16)
    # a.set_title(f'Palavras com redução segmental em MICONTES', fontsize = 16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    reducao_contagem = '\n'.join(df['reducao'].tolist())
    reducao_contagem = reducao_contagem.lower()
    reducao_contagem = FreqDist(reducao_contagem.split())
    reducao_contagem = pd.DataFrame([reducao_contagem]).transpose()
    reducao_contagem.reset_index(inplace=True)
    reducao_contagem.columns = ['palavra', 'frequência']
    reducao_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.lineplot(data=reducao_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='purple')
    plt.xticks(rotation=90)
    a.set_title(f'Palavras com redução segmental em {df["audio"][0]}', fontsize=16)
    a.set_title(f'Palavras com redução segmental em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    truncamento_contagem = '\n'.join(df['truncamentos'].tolist())
    truncamento_contagem = truncamento_contagem.lower()
    truncamento_contagem = FreqDist(truncamento_contagem.split())
    truncamento_contagem = pd.DataFrame([truncamento_contagem]).transpose()
    truncamento_contagem.reset_index(inplace=True)
    truncamento_contagem.columns = ['palavra', 'frequência']
    truncamento_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.lineplot(data=truncamento_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='orange')
    plt.xticks(rotation=90)
    a.set_title(f'Truncamentos em {df["audio"][0]}', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    alongamento_voc_contagem = '\n'.join(df['alongamentos_voc'].tolist())
    alongamento_voc_contagem = alongamento_voc_contagem.lower()
    alongamento_voc_contagem = FreqDist(alongamento_voc_contagem.split())
    alongamento_voc_contagem = pd.DataFrame([alongamento_voc_contagem]).transpose()
    alongamento_voc_contagem.reset_index(inplace=True)
    alongamento_voc_contagem.columns = ['palavra', 'frequência']
    alongamento_voc_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.lineplot(data=alongamento_voc_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='green')
    plt.xticks(rotation=90)
    a.set_title(f'Alongamentos vocálicos em {df["audio"][0]} ', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    alongamento_cons_contagem = df[pd.notnull(df['alongamentos_cons'])]
    alongamento_cons_contagem = '\n'.join(alongamento_cons_contagem['alongamentos_cons'].tolist())
    alongamento_cons_contagem = alongamento_cons_contagem.lower()
    alongamento_cons_contagem = FreqDist(alongamento_cons_contagem.split())
    alongamento_cons_contagem = pd.DataFrame([alongamento_cons_contagem]).transpose()
    alongamento_cons_contagem.reset_index(inplace=True)
    alongamento_cons_contagem.columns = ['palavra', 'frequência']
    alongamento_cons_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(6, 4))
    a = sns.lineplot(data=alongamento_cons_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='green')
    plt.xticks(rotation=90)
    a.set_title(f'Alongamentos nasais e consonantais {df["audio"][0]}', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    #

    classes_diante_pausas = df[pd.notnull(df['classes_diante_pausas'])]
    classes_diante_pausas = '\n'.join(classes_diante_pausas['classes_diante_pausas'].tolist())
    # classes_diante_pausas = classes_diante_pausas.lower()
    classes_diante_pausas = FreqDist(classes_diante_pausas.split())
    classes_diante_pausas = pd.DataFrame([classes_diante_pausas]).transpose()
    classes_diante_pausas.reset_index(inplace=True)
    classes_diante_pausas.columns = ['palavra', 'frequência']
    classes_diante_pausas.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(6, 4))
    a = sns.lineplot(data=classes_diante_pausas, x='palavra', y='frequência', \
                     lw='3', marker='^', color='green')
    plt.xticks(rotation=90)
    a.set_title(f'Classes de palavras diante de pausas {df["audio"][0]}', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    #
    palavras_diante_pausas = df[pd.notnull(df['palavras_diante_pausas'])]
    palavras_diante_pausas = '\n'.join(palavras_diante_pausas['palavras_diante_pausas'].tolist())
    # palavras_diante_pausas = palavras_diante_pausas.lower()
    palavras_diante_pausas = FreqDist(palavras_diante_pausas.split())
    palavras_diante_pausas = pd.DataFrame([palavras_diante_pausas]).transpose()
    palavras_diante_pausas.reset_index(inplace=True)
    palavras_diante_pausas.columns = ['palavra', 'frequência']
    palavras_diante_pausas.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.lineplot(data=palavras_diante_pausas[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='deeppink')
    plt.xticks(rotation=90)
    a.set_title(f'Palavras diante de pausas em {df["audio"][0]}', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    #
    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.barplot(data=df, x='audio', y='qt_palavras', palette='inferno')
    plt.xticks(rotation=90)
    a.set_title(f'Quantidade de palavras em {df["audio"][0]}', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

else:

    contagem_palavras = '\n'.join(df['transcricao_limpa'].tolist())
    contagem_palavras = contagem_palavras.lower()
    contagem_palavras = FreqDist(contagem_palavras.split())
    contagem_palavras = pd.DataFrame([contagem_palavras]).transpose()
    contagem_palavras.reset_index(inplace=True)
    contagem_palavras.columns = ['palavra', 'frequência']
    contagem_palavras.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 3))
    a = sns.lineplot(data=contagem_palavras[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='purple')
    plt.xticks(rotation=90)
    a.set_title(f'Palavras mais frequentes em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    reducao_contagem = '\n'.join(df['reducao'].tolist())
    reducao_contagem = reducao_contagem.lower()
    reducao_contagem = FreqDist(reducao_contagem.split())
    reducao_contagem = pd.DataFrame([reducao_contagem]).transpose()
    reducao_contagem.reset_index(inplace=True)
    reducao_contagem.columns = ['palavra', 'frequência']
    reducao_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.lineplot(data=reducao_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='purple')
    plt.xticks(rotation=90)
    a.set_title(f'Palavras com redução segmental em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    truncamento_contagem = '\n'.join(df['truncamentos'].tolist())
    truncamento_contagem = truncamento_contagem.lower()
    truncamento_contagem = FreqDist(truncamento_contagem.split())
    truncamento_contagem = pd.DataFrame([truncamento_contagem]).transpose()
    truncamento_contagem.reset_index(inplace=True)
    truncamento_contagem.columns = ['palavra', 'frequência']
    truncamento_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 3))
    a = sns.lineplot(data=truncamento_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='orange')
    plt.xticks(rotation=90)
    a.set_title('Truncamentos em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    alongamento_voc_contagem = '\n'.join(df['alongamentos_voc'].tolist())
    alongamento_voc_contagem = alongamento_voc_contagem.lower()
    alongamento_voc_contagem = FreqDist(alongamento_voc_contagem.split())
    alongamento_voc_contagem = pd.DataFrame([alongamento_voc_contagem]).transpose()
    alongamento_voc_contagem.reset_index(inplace=True)
    alongamento_voc_contagem.columns = ['palavra', 'frequência']
    alongamento_voc_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.lineplot(data=alongamento_voc_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='green')
    plt.xticks(rotation=90)
    a.set_title(f'Alongamentos vocálicos em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    alongamento_cons_contagem = df[pd.notnull(df['alongamentos_cons'])]
    alongamento_cons_contagem = '\n'.join(alongamento_cons_contagem['alongamentos_cons'].tolist())
    alongamento_cons_contagem = alongamento_cons_contagem.lower()
    alongamento_cons_contagem = FreqDist(alongamento_cons_contagem.split())
    alongamento_cons_contagem = pd.DataFrame([alongamento_cons_contagem]).transpose()
    alongamento_cons_contagem.reset_index(inplace=True)
    alongamento_cons_contagem.columns = ['palavra', 'frequência']
    alongamento_cons_contagem.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(6, 4))
    a = sns.lineplot(data=alongamento_cons_contagem[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='green')
    plt.xticks(rotation=90)
    a.set_title(f'Alongamentos nasais e consonantais em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    #

    classes_diante_pausas = df[pd.notnull(df['classes_diante_pausas'])]
    classes_diante_pausas = '\n'.join(classes_diante_pausas['classes_diante_pausas'].tolist())
    # classes_diante_pausas = classes_diante_pausas.lower()
    classes_diante_pausas = FreqDist(classes_diante_pausas.split())
    classes_diante_pausas = pd.DataFrame([classes_diante_pausas]).transpose()
    classes_diante_pausas.reset_index(inplace=True)
    classes_diante_pausas.columns = ['palavra', 'frequência']
    classes_diante_pausas.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(6, 3))
    a = sns.lineplot(data=classes_diante_pausas, x='palavra', y='frequência', \
                     lw='3', marker='^', color='green')
    plt.xticks(rotation=90)
    a.set_title(f'Classes de palavras diante de pausas em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    #
    palavras_diante_pausas = df[pd.notnull(df['palavras_diante_pausas'])]
    palavras_diante_pausas = '\n'.join(palavras_diante_pausas['palavras_diante_pausas'].tolist())
    # palavras_diante_pausas = palavras_diante_pausas.lower()
    palavras_diante_pausas = FreqDist(palavras_diante_pausas.split())
    palavras_diante_pausas = pd.DataFrame([palavras_diante_pausas]).transpose()
    palavras_diante_pausas.reset_index(inplace=True)
    palavras_diante_pausas.columns = ['palavra', 'frequência']
    palavras_diante_pausas.sort_values(by='frequência', ascending=False, inplace=True)

    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 3))
    a = sns.lineplot(data=palavras_diante_pausas[:30], x='palavra', y='frequência', \
                     lw='3', marker='^', color='deeppink')
    plt.xticks(rotation=90)
    a.set_title('Palavras diante de pausas em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Frequência", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()

    #
    sns.set_style('whitegrid')
    plt.figure(dpi=200, figsize=(8, 5))
    a = sns.barplot(data=df, x='audio', y='qt_palavras', palette='inferno')
    plt.xticks(rotation=90)
    a.set_title(f'Quantidade de palavras por transcrição em MICONTES', fontsize=16)
    a.set_xlabel("", fontsize=14)
    a.set_ylabel("Quantidade de palavras", fontsize=15)
    a.tick_params(labelsize=15)
    plt.show()



df= pd.read_csv('df_ce.csv')

df['duracao'] = [15, 11, 11, 12, 8, 7,4, 9, 10, 7, 8, 4]


sns.set_style('whitegrid')
plt.figure(dpi=200, figsize=(8, 5))
a = sns.barplot(data=df, x='audio', y='duracao', palette='inferno_r')
plt.xticks(rotation=90)
a.set_title(f'Duração dos áudios no minicorpus', fontsize=19)
a.set_xlabel("áudios", fontsize=16)
a.set_ylabel("minutos", fontsize=18)
a.tick_params(labelsize=15)
plt.show()


