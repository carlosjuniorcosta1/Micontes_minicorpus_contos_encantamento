# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:26:27 2022

@author: Usuario
"""
#arrumar V|+

import pandas as pd
import os
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def normaliza_ce(texto):
    dicio = {'abêia': 'abelha', 'adoecê': 'adoecer', 'afroxava': 'afrouxava',
             'ajudr': 'ajudar', 'alembra': 'lembra', 'amarilinha': 'amarelinha',
             'amarru': 'amarro', 'apanhr': 'apanhar', 'apiô/': 'apeou',
             'apiou': 'apeou', 'armoço': 'almoço', 'arquere': 'alqueire',
             'arriou': 'arriou', 'arrumaro': 'arrumaram', 'arta': 'alta',
             'arto': 'alto', 'arvore': 'árvore', 'asfarça': 'alface',
             'avoando': 'voando', 'avuô': 'voou', 'banando': 'abanando',
             'bão': 'bom', 'barreno': 'varrendo', 'baruiada': 'barulhada',
             'bassôra': 'vassoura', 'bassoura': 'vassoura', 'basto': 'bastou',
             'batante': 'bastante', 'bejando': 'beijando', 'braba': 'brava',
             'brabo/': 'bravo', 'bunita': 'bonita', 'burduada': 'bordoada',
             'burracha': 'borracha', 'butão': 'botão', 'c’ocê': 'com você',
             'c’um': 'com um', 'cachchorro': 'cachorro', 'cacunda': 'corcunda',
             'cadin': 'bocadinho', 'cafezim': 'cafezinho', 'caminhano': 'caminhando',
             'carçada': 'calçada', 'carçou': 'calçou', 'casá': 'casar',
             'cascaro': 'cascaram', 'cavacar': 'cavoucar', 'caxetinha': 'caixetinha',
             'caxinha': 'caixinha', 'cê': 'você', 'CÊ': 'você', 'cês': 'vocês',
             'chegu': 'chego', 'co’cê': 'com você', 'companhando': 'acompanhando',
             'consurtou': 'consultou', 'cramando': 'reclamando', 'cum': 'com um',
             'cumbina': 'combina', 'cumê': 'comer', 'cumeu': 'comeu',
             'd’eu': 'de eu', 'd’ocê': 'de você', 'd’outra': 'da outra',
             'dabaixo': 'debaixo', 'daonde': 'da onde', 'debrucou': 'debruçou',
             'demorano': 'demorando', 'denamite': 'dinamite',
             'destraído': 'distraído', 'dexava': 'deixava', 'di': 'de',
             'dibruçado': 'debruçando', 'dipressa': 'depressa', 'disacate': 'desacate',
             'disatar': 'desatar', 'discia': 'descia', 'disincantar': 'desencantar',
             'disincantou': 'desencantou', 'dismanchando': 'desmanchando', 'dispesa': 'despesa',
             'ditardinha': 'de tardinha', 'doeceu': 'adoeceu', 'drobou': 'dobrou',
             'drumindo': 'dormindo',  'dum': 'de um',  'dum/': 'de um', 'duma': 'de uma',
             'embruiar': 'embrulhar', 'encontrouu': 'encontrou', 'entrá': 'entrar',
             'envem': 'vem',  'ênvem': 'está vindo', 'envinha': 'vinha',  'estoraria': 'estouraria',
             'estraçaiar': 'estraçalhar', 'exprementá': 'experimentar', 'exprementar': 'experimentar',
             'exprementou': 'experimentou',
             'falu': 'falo',
             'ferveno': 'fervendo',
             'ficô': 'ficou',
             'fizero': 'fizeram',
             'fógos': 'fogos',
             'foia': 'folha',
             'friage': 'friagem',
             'fugueira': 'fogueira',
             'gaio': 'galho',
             'garrado': 'agarrado',
             'garro': 'agarro',
             'garrou/': 'agarrou',
             'giagante': 'gigante',
             'guentava': 'aguentava',
             'homi': 'homem',
             'incumenda': 'encomenda',
             'infeito': 'enfeite',
             'infezado': 'enfezado',
             'infiando': 'enfiando',
             'interar': 'inteirar',
             'interrou': 'enterrou',
             'intertendo': 'entendendo',
             'intirim': 'inteirinho',
             'joãozinho': 'Joãozinho',
             'JoãozinJoãozinho': 'Joãozinho Joãozinho',
             'jugava/': 'jogava',
             'laçinho': 'lacinho',
             'lâmpra': 'lâmpada',
             'lâmpras': 'lâmpadas',
             'limbuzou': 'lambuzou',
             'madrastra': 'madrasta',
             'marrô': 'amarrou',
             'marrou': 'amarrou',
             'matô': 'matou',
             'memo': 'mesmo',
             'mobíliado': 'mobiliado',
             'moçinha/': 'mocinha',
             'mói': 'molho',
             'mói/': 'molho',
             'moiá': 'molhar',
             'mucadinho': 'bocadinho',
             'muié': 'mulher',
             'muié/': 'mulher',
             'muncadin': 'bocadinho',
             'muntado': 'montado',
             'muntou': 'montou',
             'musga': 'música',
             'n’ocê': 'em você',
             'numa': 'em uma',
             'ocê': 'você',
             'ocê/': 'você',
             'ocês': 'vocês',
             'oiá': 'olhar',
             'óia': 'olha',
             'oiando': 'olhando',
             'oiando/': 'olhando',
             'oiano': 'olhando',
             'oiô': 'olhou',
             'ôio': 'olho',
             'oiô/': 'olhou',
             'ond’é': 'onde é',
             'oreia': 'orelha',
             'ôro': 'ouro',
             'panha': 'apanha',
             'panhá': 'apanhar',
             'panhando': 'apanhando',
             'panhou': 'apanhou',
             'passo': 'passou',
             'passô': 'passou',
             'pegá': 'pegar',
             'pegu': 'pego',
             'pelengrina': 'pelegrina',
             'penerando': 'peneirando',
             'peregina': 'pelegrina',
             'piruca': 'peruca',
             'poquinho': 'pouquinho',
             'pr’ali': 'para ali',
             'pr’ocê': 'para você',
             'pr’um': 'para um',
             'prantar': 'plantar',
             'prantador': 'plantador',
             'pro/': 'para o',
             'pro’cê': 'para você',
             'pro’s': 'para os',
             'pros': 'para os',
             'prum': 'para um',
             'pudê': 'poder',
             'quarqué': 'qualquer',
             'queremdo': 'querendo',
             'quintale': 'quintal',
             'rachá': 'rachar',
             'rancando': 'arrancando',
             'rancou': 'arrancou',
             'raNHADO': 'arranhado',
             'ranhado': 'arranhado',
             'rapazim': 'rapazinho',
             'repôio': 'repolho',
             'robava': 'roubava',
             'rubentar': 'arrebentar',
             'rubentou': 'arrebentou',
             'sabê': 'saber',
             'samiava': 'espalhava',
             'sarta': 'saltar',
             'sartá': 'saltar',
             'sirpultura': 'sepultura',
             'sordado': 'soldado',
             'sordador': 'soldador',
             'sordar': 'soldar',
             'sordou': 'soldou',
             'sortar': 'soltar',
             'sortava': 'soltava',
             'sortei': 'soltei',
             'sorto': 'solto',
             'sortou': 'soltou',
             'sote': 'sótão',
             'tão': 'estão',
             'tava': 'estava',
             'tiquim': 'tiquinho',
             'toaia': 'toalha',
             'trabaiá': 'trabalhar',
             'trabaiá/': 'trabalhar',
             'trabaidor': 'trabalhador',
             'treis': 'três',
             'trêis': 'três',
             'trocaro': 'trocaram',
             'Véia': 'velha',
             'véia': 'velha',
             'véio/': 'velho',
             'vermeio': 'vemelho',
             'vêz': 'vez',
             'viero': 'vieram',
             'vim’bora': 'vim embora',
             'vorta': 'voltar',
             'vortar': 'voltar',
             'vortava': 'voltava',
             'vortou': 'voltou',
             'vuô': 'voou',
             'xô': 'deixa eu',
             'zarôia': 'zarolha',
             'zóio': 'olhos'}

    texto = texto.replace('óia', 'olha').replace('oiando', 'olhando') \
        .replace('oiano', 'olhando').replace('véio', 'velho').replace('sordado', 'soldado')
    texto = texto.replace('quintale', 'quintal').replace('véia', 'velha') \
        .replace('têia', 'telha').replace('foia', 'folha') \
        .replace('oiand', 'olhando').replace('cramando', 'clamando') \
        .replace('repôio', 'repolho') \
        .replace('bassôra', 'vassoura').replace('trêis', 'três') \
        .replace('barreu', 'varreu') \
        .replace('trabaiá', 'trabalhar').replace('toaia', 'toalha') \
        .replace('gaiô', 'galho') \
        .replace('zarôia', 'zarolha').replace('muié', 'mulher') \
        .replace('disapareceu', 'desapareceu').replace('muntou', 'montou') \
        .replace('intirim', 'inteirinho').replace('drumindo', 'dormindo') \
        .replace('pegô', 'pegou').replace('lâmpra', 'lâmpada').replace('sóte', 'sótão') \
        .replace('incoída', 'encolhida').replace('experimentá', 'experimentar') \
        .replace('exprementá', 'experimentar').replace('adoecê', 'adoeceu')

    texto = re.sub(r'(?<=\s|,|\?|\!)vortando(?=\s|,|\?|\!)', 'voltando', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)fazar(?=\s|,|\?|\!)', 'fazer', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)debendo(?=\s|,|\?|\!)', 'bebendo', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)premeiro(?=\s|,|\?|\!)', 'primeiro', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)cachchorro(?=\s|,|\?|\!)', 'cachorro', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)falu(?=\s|,|\?|\!)', 'falou', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)faloo(?=\s|,|\?|\!)', 'falou', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)bença(?=\s|,|\?|\!)', 'benção', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)imbengou(?=\s|,|\?|\!)', 'embengou', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)vorta(?=\s|,|\?|\!)', 'voltar', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)vortava(?=\s|,|\?|\!)', 'voltava', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)piruca(?=\s|,|\?|\!)', 'peruca', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)abêia(?=\s|,|\?|\!)', 'abelha', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)terijum(?=\s|,|\?|\!)', 'tirijum', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)caxa(?=\s|,|\?|\!)', 'caixa', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)caxinha(?=\s|,|\?|\!)', 'caixinha', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)pelengrina(?=\s|,|\?|\!)', 'pelegrina', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)exprementar(?=\s|,|\?|\!)', 'experimentar', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)batante(?=\s|,|\?|\!)', 'bastante', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)arreado(?=\s|,|\?|\!)', 'arriado', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)arreadinho(?=\s|,|\?|\!)', 'arriadinho', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)penerando(?=\s|,|\?|\!)', 'peneirando', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)penera(?=\s|,|\?|\!)', 'peneira', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)véve(?=\s|,|\?|\!)', 'vive', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)vortou(?=\s|,|\?|\!)', 'voltou', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)pra(?=\s|,|\!|\?)', 'para', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)feixo(?=\s|,|\!|\?)', 'feixe', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)benamite(?=\s|,|\!|\?)', 'dinamite', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)envermeiou(?=\s|,|\!|\?)', 'envermelhou', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)pro(?=\s|,|\!|\?)', 'para o', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)pr’ocê(?=\s|,|\?|\!)', 'para você', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)pro’cê(?=\s|,|\?|\!)', 'para você', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)co’cê(?=\s|,|\?|\!)', 'com você', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)arto(?=\s|,|\?|\!)', 'alto', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)arta(?=\s|,|\?|\!)', 'alta', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)cumê(?=\s|,|\?|\!)', 'comer', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)ranhado(?=\s|,|\?|\!)', 'arranhado', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)ranhada(?=\s|,|\?|\!)', 'arranhada', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)banando(?=\s|,|\?|\!)', 'abanando', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)imbira(?=\s|,|\?|\!)', 'embira', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)cavacada(?=\s|,|\?|\!)', 'cavoucada', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)tô(?=\s|,|\?|\!)', 'estou', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)sarta(?=\s|,|\?|\!)', 'salta', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)lindro(?=\s|,|\?|\!)', 'lindo', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)muncado(?=\s|,|\?|\!)', 'bocado', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)muncadinho(?=\s|,|\?|\!)', 'bocadinho', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)muncadin(?=\s|,|\?|\!)', 'bocadinho', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)caas(?=\s|,|\?|\!)', 'casa', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)estalêiro(?=\s|,|\?|\!|-)', 'estaleiro', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)sali(?=\s|,|\?|\!)', 'sal', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)mucadinho(?=\s|,|\?|\!)', 'bocadinho', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)ventá(?=\s|,|\?|\!)', 'ventar', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)lindra(?=\s|,|\?|\!)', 'linda', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)covar(?=\s|,|\?|\!)', 'cavar', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\!|\?)ocê(?=\s|,|\!|\?)', 'você', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\!|\?)ocês(?=\s|,|\!|\?)', 'vocês', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)cês(?=\s|,|\!|\?)', 'vocês', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)memo(?=\s|,|\!|\?)', 'mesmo', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)cê(?=\s|,|\!|\?)', 'você', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)maniquim(?=\s|,|\!|\?)', 'manequim', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)tamo(?=\s|,|\!|\?)', 'estamos', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)tava(?=\s|,|\!|\?)', 'estávamos', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)teve(?=\s|,|\!|\?)', 'esteve', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)cumbina(?=\s|,|\!|\?)', 'combina', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)bagem(?=\s|,|\!|\?)', 'vagem', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)vamo(?=\s|,|\!|\?)', 'vamos', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)tá(?=\s|,|\!|\?)', 'está', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)ta(?=\s|,|\!|\?)', 'está', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)prince(?=\s|,|\!|\?)', 'príncipe', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)cumeu(?=\s|,|\!|\?)', 'comeu', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)matô(?=\s|,|\!|\?)', 'matou', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)cabrintinho(?=\s|,|\!|\?)', 'cabritinho', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)ciloura(?=\s|,|\!|\?)', 'ceroula', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)tardi(?=\s|,|\!|\?)', 'tarde', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)initeligivel(?=\s|,|\!|\?)', 'ininteligível', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)nóh(?=\s|,|\!|\?)', 'nossa', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)ôh(?=\s|,|\!|\?|/)', 'ô', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)óh(?=\s|,|\!|\?|/)', 'oh', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)umndo(?=\s|,|\!|\?)', 'mundo', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)êh(?=\s|,|\!|\?)', 'ê', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)fazedera(?=\s|,|\!|\?)', 'fazedeira', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)toaiá(?=\s|,|\!|\?)', 'toalha', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)toaia(?=\s|,|\!|\?)', 'toalha', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)trabaio(?=\s|,|\!|\?)', 'trabalho', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)trabaiô(?=\s|,|\!|\?)', 'trabalhou', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)trabaiá(?=\s|,|\!|\?)', 'trabalhar', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)véio(?=\s|,|\!|\?|/)', 'velho', texto, flags=re.IGNORECASE)

    texto = re.sub(r'(?<=\s|,|\?|\!)trabaiano(?=\s|,|\!|\?)', 'trabalhando', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)prum(?=\s|,|\!|\?)', 'para um', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)pruns(?=\s|,|\!|\?)', 'para uns', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)pr’um(?=\s|,|\!|\?)', 'para um', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)pr’uns(?=\s|,|\!|\?)', 'para uns', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)prantar(?=\s|,|\!|\?)', 'plantar', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<=\s|,|\?|\!)prantador(?=\s|,|\!|\?)', 'plantador', texto, flags=re.IGNORECASE)



    texto = " ".join([dicio[p] if p in dicio else p for p in texto.split(' ')])
    return texto


files = '\n'.join([x for x in os.listdir() if x.startswith('ce')])

df = pd.DataFrame()

for x in files.splitlines():
    with open(x, 'r+', encoding='utf-8') as source:
        file = source.read()
        df = df.append([file])

df.columns = ['transcricao']
df['audio'] = df['transcricao'].apply(lambda x: ' '.join(re.findall(r'CE-\d+', x)))
df['audio'] = df['audio'].str.lower().str.replace('-', '')
df['titulo'] = ['Maria Caçula Maria do Meio e Maria Mais Véia', 'O gigante', \
                'Os dois menino gêmeo', 'Adão e Diomara na Terra de Era', \
                'Joãozinho e Maria', 'Joãozinho e Diomara', \
                'Joãozinho e Estalêro Estalão', 'O chicotinho', 'O pé de Feijão', \
                'Lampra véia pro lambra nova', 'Joãozinho Borraiero', 'Moça da figuêra']

df['transcricao'] = df['transcricao'].apply(lambda x: re.sub(r'CE-\d+', '', x))

df['transcricao_limpa'] = df['transcricao'].apply(
    lambda x: re.sub(r'\.\.\.|,|\.|\!|\?|–+|--|-|/|\(trecho.*?\)|\(\(.*?\)\)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'::(?=\s)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'\n', ' ', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'\s(?=\s\s\s)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'\s(?=\s\s)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'\s(?=\s)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'(?<=\w)\((?=\w\)\s)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'(?<=\w)\)(?=\s)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'(?<=\w)\(', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'(?<=\w)\)', '', x))
df['transcricao_limpa'] = df['transcricao_limpa'].apply(lambda x: re.sub(r'(?<=\w)::|\(', '', x))
df['transcricao_para_norm'] = df['transcricao'].apply(lambda x: re.sub(r'\w+/|\(\(.*?\)\)|\(', '', x))
df['transcricao_para_norm'] = df['transcricao_para_norm'].apply(lambda x: re.sub(r'\w+/', '', x))
df['transcricao_para_norm'] = df['transcricao_para_norm'].apply(lambda x: re.sub(r'::', '', x))
df['transcricao_para_norm'] = df['transcricao_para_norm'].apply(lambda x: re.sub(r'\)', "", x))
df['transcricao_para_norm'] = df['transcricao_para_norm'].apply(lambda x: re.sub(r'\(', "", x))

df['transcricao_para_norm'] = df['transcricao_para_norm'].apply(lambda x: re.sub(r'(?<!\?|\!)\.\.\.', ",", x))
df['transcricao_para_norm'] = df['transcricao_para_norm'].apply(lambda x: re.sub(r'(?<=\!|\?)\.\.\.', "", x))

df['transcricao_normalizada'] = df['transcricao_para_norm'].apply(normaliza_ce)

#fazer transcrição normalizada para etiquetagem e retirar todos os símbolos
import pickle

with open('brill_00', 'rb') as source:
    tagueador = pickle.load(source)

df['transcricao_pos'] = df['transcricao_limpa'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: tagueador.tag(x))

df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='sio’',\s')\w+|(?<='sio',\s')\w+|(?<='senhora',\s')\w+", 'PROPESS', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='melancia’',\s')\w+", 'N', str(x), flags=re.IGNORECASE))

df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"\(\'Nossa',\s\'\w+\'\),\s\(\'Senhora\',\s\'\w+\'\),", "('Nossa Senhora', 'IN'),", x))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='na',\s')\w+|(?<='no',\s')\w+", 'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='nessa',\s')\w+|(?<='nesse',\s')\w+|(?<='nisso',\s')\w+|(?<='nesses',\s')\w+|(?<='nessas',\s')\w+", 'PREP|+',
    str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='ao',\s')\w+|(?<='aos',\s')\w+", 'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='o',\s')\w+|(?<='os',\s')\w+", 'ART', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='da',\s')\w+|(?<='do',\s')\w+|(?<='dos',\s')\w+|(?<='das',\s')\w+|(?<='duma',\s')\w+|(?<='dum',\s')\w+|(?<='duns',\s')\w+|(?<='dumas',\s')\w+",
    'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='Diomar',\s')\w+", 'NPROP', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='de',\s')\w+", 'PREP', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='pelos',\s')\w+|(?<='pelo',\s')\w+|(?<='pela',\s')\w+|(?<='pelas',\s')\w+", 'PREP|+', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='dele',\s')\w+|(?<='deles',\s')\w+|(?<='dela',\s)\w+|(?<='delas',\s)\w+", 'PROADJ', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='num',\s')\w+", 'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='um',\s')\w+|(?<='uns',\s')\w+|(?<='uma',\s')\w+|(?<='umas',\s')\w+", 'ART', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='comigo',\s')\w+", 'PROPESS', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='por',\s')\w+", 'PREP', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='contigo',\s')\w+", 'PROPESS', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='ahn',\s')\w+|(?<='ham',\s')\w+|(?<='hum',\s')\w+|(?<='uhn',\s')\w+|(?<='êh',\s')\w+|(?<='ôh',\s')\w+", 'IN',
    str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='ah',\s')\w+|(?<='eh',\s')\w+|(?<='ih',\s')\w+|(?<='oh',\s')\w+|(?<='ô',\s')\w+|(?<='uai',\s')\w+|(?<='ué',\s')\w+|(?<='ê',\s')\w+",
    'IN', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='nu',\s')\w+|(?<='pá',\s')\w+|(?<='parará',\s')\w+|(?<='tanãnãnã',\s')\w+|(?<='tchan',\s')\w+|(?<='tum',\s')\w+",
    'IN', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='deixar',\s')\w+|(?<='deixa',\s')\w+|(?<='deixou',\s')\w+|(?<='deixei',\s')\w+|(?<='deixaram',\s')\w+|(?<='deixamos',\s')\w+|(?<='deixaria',\s')\w+|(?<='deixariam',\s')\w+|(?<='deixam',\s')\w+|(?<='deixo',\s')\w+|(?<='deixasse',\s')\w+|(?<='deixassem',\s')\w+|(?<='deixarmos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='começar',\s')\w+|(?<='começa',\s')\w+|(?<='começou',\s')\w+|(?<='comecei',\s')\w+|(?<='começaram',\s')\w+|(?<='começam',\s')\w+|(?<='começaria',\s')\w+|(?<='começariam',\s')\w+|(?<='começamos',\s')\w+|(?<='começo',\s')\w+|(?<='começasse',\s')\w+|(?<='começassem',\s')\w+|(?<='começarmos',\s')\w+|(?<='comece',\s')\w+|(?<='comecemos',\s')\w+",
    'V', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='chegar',\s')\w+|(?<='chega',\s')\w+|(?<='chegou',\s')\w+|(?<='cheguei',\s')\w+|(?<='chegaram',\s')\w+|(?<='chegam',\s')\w+|(?<='chegaria',\s')\w+|(?<='chegariam',\s')\w+|(?<='chegamos',\s')\w+|(?<='chego',\s')\w+|(?<='chegasse',\s')\w+|(?<='chegassem',\s')\w+|(?<='chegarmos',\s')\w+|(?<='chegue',\s')\w+|(?<='cheguemos',\s')\w+|(?<='cheguemos',\s')\w+",
    'V', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='ser',\s')\w+|(?<='é',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='somos',\s')\w+|(?<='seria',\s')\w+|(?<='seriam',\s')\w+|(?<='são',\s')\w+|(?<='sou',\s')\w+|(?<='era',\s')\w+|(?<='eram',\s')\w+|(?<='for',\s')\w+|(?<='formos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='sermos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='poder',\s')\w+|(?<='pode',\s')\w+|(?<='pôde',\s')\w+|(?<='pude',\s')\w+|(?<='puderam',\s')\w+|(?<='podemos',\s')\w+|(?<='poderia',\s')\w+|(?<='poderiam',\s')\w+|(?<='podem',\s')\w+|(?<='posso',\s')\w+|(?<='podia',\s')\w+|(?<='podiam',\s')\w+|(?<='pudesse',\s')\w+|(?<='pudessem',\s')\w+|(?<='podermos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='estar',\s')\w+|(?<='está',\s')\w+|(?<='esteve',\s')\w+|(?<='estive',\s')\w+|(?<='estiveram',\s')\w+|(?<='estamos',\s')\w+|(?<='estaria',\s')\w+|(?<='estariam',\s')\w+|(?<='estão',\s')\w+|(?<='estou',\s')\w+|(?<='estava',\s')\w+|(?<='estivemos',\s')\w+|(?<='estivesse',\s')\w+|(?<='estivéssemos',\s')\w+|(?<='estivessem',\s')\w+|(?<='estarmos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='ir',\s')\w+|(?<='vai',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='vamos',\s')\w+|(?<='iria',\s')\w+|(?<='iriam',\s')\w+|(?<='vão',\s')\w+|(?<='vou',\s')\w+|(?<='ía',\s')\w+|(?<='fomos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='iria',\s')\w+|(?<='vamos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='nũ',\s')\w+|(?<='né',\s')\w+|(?<='aí',\s')\w+", 'ADV', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='Whatsapp',\s')\w+|(?<='Instagram',\s')\w+|(?<='Facebook',\s')\w+|(?<='big',\s')\w+|(?<='brother',\s')\w+|(?<='buffet',\s')\w+|(?<='feedback',\s')\w+|(?<='fair',\s')\w+|(?<='play',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='open',\s')\w+|(?<='over',\s')\w+|(?<='photoshop',\s')\w+|(?<='pop',\s')\w+|(?<='plus',\s')\w+|(?<='réveillon',\s')\w+|(?<='sexy',\s')\w+|(?<='serial',\s')\w+|(?<='killer',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='shopping',\s')\w+|(?<='short',\s')\w+|(?<='show',\s')\w+|(?<='smartphone',\s')\w+|(?<='software',\s')\w+|(?<='telemarketing',\s')\w+|(?<='videogame',\s')\w+|(?<='tablet',\s')\w+|(?<='Windows',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='yes',\s')\w+|(?<='vip',\s')\w+|(?<='web',\s')\w+|(?<='smartphone',\s')\w+|(?<='slide',\s')\w+|(?<='states',\s')\w+|(?<='videogame',\s')\w+|(?<='online',\s')\w+|(?<='office',\s')\w+|(?<='offline',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(lambda x: re.sub(
    r"(?<='ficar',\s')\w+|(?<='fica',\s')\w+|(?<='ficou',\s')\w+|(?<='fiquei',\s')\w+|(?<='ficaram',\s')\w+|(?<='ficamos',\s')\w+|(?<='ficaria',\s')\w+|(?<='ficariam',\s')\w+|(?<='ficam',\s')\w+|(?<='fico',\s')\w+|(?<='ficasse',\s')\w+|(?<='ficassem',\s')\w+|(?<='ficarmos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='né',\s')\w+", 'ADV', str(x), flags=re.IGNORECASE))
df['transcricao_pos'] = df['transcricao_pos'].apply(
    lambda x: re.sub(r"(?<='o',\s')\w+|(?<='os',\s')\w+", 'ART', str(x), flags=re.IGNORECASE))

df['reducoes_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'(?<=\()[a-z-A-Z]{1}(?=\))', x)))
df['marc_disc_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'\?', x)))
df['faticos_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'\!', x)))
df['truncamentos_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'/', x)))
df['pausas_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'\.\.\.', x)))
df['transcricao_sem_par'] = df['transcricao'].apply(lambda x: re.sub(r'\(|\)', '', x))
df['alongamentos_voc_c'] = df['transcricao_sem_par'].apply(lambda x: len(re.findall(
    r'(?<=a)::|(?<=e)::|(?<=i)::|(?<=o)::|(?<=u)::|(?<=â)::|(?<=õ)::|(?<=à)::|(?<=á)::|(?<=â)::|(?<=ô)::|(?<=ó)::|(?<=é)::|(?<=ê)::|(?<=ú)::|(?<=h)::',
    x)))
df['alongamentos_cons_c'] = df['transcricao_sem_par'].apply(
    lambda x: len(re.findall(r'(?<!a|e|i|o|u|ã|õ|à|á|â|ô|ó|é|ê|ú|h)::', x)))
df['discurso_dir_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'–.*?–', x)))
df['descricao_narrador_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'--.*?--', x)))
df['descricao_transcritor_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'\(\(.*?\)\)', x)))

df['faticos_int_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'\!', x)))
df['truncamentos_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'/', x)))
df['pausas_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'\.\.\.', x)))
df['transcricao_sem_par'] = df['transcricao'].apply(lambda x: re.sub(r'\(|\)', '', x))
df['alongamentos_voc_c'] = df['transcricao_sem_par'].apply(lambda x: len(re.findall(
    r'(?<=a)::|(?<=e)::|(?<=i)::|(?<=o)::|(?<=u)::|(?<=â)::|(?<=õ)::|(?<=à)::|(?<=á)::|(?<=â)::|(?<=ô)::|(?<=ó)::|(?<=é)::|(?<=ê)::|(?<=ú)::|(?<=h)::',
    x)))
df['alongamentos_cons_c'] = df['transcricao_sem_par'].apply(
    lambda x: len(re.findall(r'(?<!a|e|i|o|u|ã|õ|à|á|â|ô|ó|é|ê|ú|h)::', x)))
df['discurso_dir_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'–.*?–', x)))
df['descricao_narrador_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'--.*?--', x)))
df['descricao_transcritor_c'] = df['transcricao'].apply(lambda x: len(re.findall(r'\(\(.*?\)\)', x)))

df['transcricao_sem_pausas_along'] = df['transcricao'].apply(lambda x: re.sub(r'\.\.\.|::|\!|\?', '', x))

df['reducao'] = df['transcricao_sem_pausas_along'].apply(
    lambda x: ' '.join(re.findall(r'\(?\w?\)?\(?\w?\)?\w+(?=\(\w\))\(?\w?\)?\w+\(?\w?\)?', x)))
df['transcricao_sem_pausas_along_2'] = df['transcricao'].apply(lambda x: re.sub(r'\.\.\.|::|\!|\(|\)', '', x))
df['marc_disc_int'] = df['transcricao_sem_pausas_along_2'].apply(lambda x: ' '.join(re.findall(r'\w+\?', x)))
df['transcricao_sem_pausas_along_3'] = df['transcricao'].apply(lambda x: re.sub(r'\.\.\.|\?|\(|\)', '', x))

df['faticos_int'] = df['transcricao_sem_pausas_along_3'].apply(lambda x: ' '.join(re.findall(r'\w+\!', x)))
df['truncamentos'] = df['transcricao'].apply(lambda x: ' '.join(re.findall(r'\w+/', x)))
df['transcricao_sem_pausas_along_4'] = df['transcricao'].apply(lambda x: re.sub(r'\.\.\.|\?|\(|\)|\!', '', x))

df['alongamentos_voc'] = df['transcricao_sem_pausas_along_4'].apply(lambda x: ' '.join(re.findall(
    r'\w+(?<=a)::.*?\s|\w+(?<=e)::.*?\s|\w+(?<=i)::.*?\s|\w+(?<=o)::.*?\s|\w+(?<=u)::.*?\s|\w+(?<=â)::.*?\s|\w+(?<=õ)::.*?\s|\w+(?<=à)::.*?\s|\w+(?<=á)::.*?\s|\w+(?<=â)::.*?\s|\w+(?<=ô)::.*?\s|\w+(?<=ó)::.*?\s|\w+(?<=é)::.*?\s|\w+(?<=ê)::.*?\s|\w+(?<=ú)::.*?\s|\w+(?<=h)::',
    x, flags=re.IGNORECASE)))
df['alongamentos_cons'] = df['transcricao_sem_pausas_along_4'].apply(
    lambda x: ' '.join(re.findall(r'\w+(?<!a|e|i|o|u|í|ã|õ|à|á|â|ô|ó|é|ê|ú|h)::', x, flags=re.IGNORECASE)))
df['discurso_dir'] = df['transcricao'].apply(lambda x: ' '.join(re.findall(r'–.*?–', x)))
df['descricao_narrador'] = df['transcricao'].apply(lambda x: ' '.join(re.findall(r'--.*?--', x)))
df['descricao_transcritor'] = df['transcricao'].apply(lambda x: ' '.join(re.findall(r'\(\(.*?\)\)', x)))

df['transcricao_com_pausas'] = df['transcricao'].apply(lambda x: re.sub(r'\?|\(|\)|\!|,|::', '', x))

df['transcricao_pos_pausas'] = df['transcricao_com_pausas'].apply(lambda x: nltk.word_tokenize(x)).apply(
    lambda x: tagueador.tag(x))

# df['pausas_palavras'] = df['transcricao_com_pausas'].apply(lambda x: re.findall(r'(\w+)\.\.\.\s(\w+)', x))

# depuração transcricao pos pausas


df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='sio’',\s')\w+|(?<='sio',\s')\w+|(?<='senhora',\s')\w+", 'PROPESS', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='melancia’',\s')\w+", 'N', str(x), flags=re.IGNORECASE))

df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"\(\'Nossa',\s\'\w+\'\),\s\(\'Senhora\',\s\'\w+\'\),", "('Nossa Senhora', 'IN'),", x))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='na',\s')\w+|(?<='no',\s')\w+", 'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='nessa',\s')\w+|(?<='nesse',\s')\w+|(?<='nisso',\s')\w+|(?<='nesses',\s')\w+|(?<='nessas',\s')\w+", 'PREP|+',
    str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='ao',\s')\w+|(?<='aos',\s')\w+", 'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='o',\s')\w+|(?<='os',\s')\w+", 'ART', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='da',\s')\w+|(?<='do',\s')\w+|(?<='dos',\s')\w+|(?<='das',\s')\w+|(?<='duma',\s')\w+|(?<='dum',\s')\w+|(?<='duns',\s')\w+|(?<='dumas',\s')\w+",
    'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='Diomar',\s')\w+", 'NPROP', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='de',\s')\w+", 'PREP', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='pelos',\s')\w+|(?<='pelo',\s')\w+|(?<='pela',\s')\w+|(?<='pelas',\s')\w+", 'PREP|+', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='dele',\s')\w+|(?<='deles',\s')\w+|(?<='dela',\s)\w+|(?<='delas',\s)\w+", 'PROADJ', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='num',\s')\w+", 'PREP|+', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='um',\s')\w+|(?<='uns',\s')\w+|(?<='uma',\s')\w+|(?<='umas',\s')\w+", 'ART', str(x),
                     flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='comigo',\s')\w+", 'PROPESS', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='por',\s')\w+", 'PREP', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='contigo',\s')\w+", 'PROPESS', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='ahn',\s')\w+|(?<='ham',\s')\w+|(?<='hum',\s')\w+|(?<='uhn',\s')\w+|(?<='êh',\s')\w+|(?<='ôh',\s')\w+", 'IN',
    str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='ah',\s')\w+|(?<='eh',\s')\w+|(?<='ih',\s')\w+|(?<='oh',\s')\w+|(?<='ô',\s')\w+|(?<='uai',\s')\w+|(?<='ué',\s')\w+",
    'IN', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='nu',\s')\w+|(?<='pá',\s')\w+|(?<='parará',\s')\w+|(?<='tanãnãnã',\s')\w+|(?<='tchan',\s')\w+|(?<='tum',\s')\w+",
    'IN', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='deixar',\s')\w+|(?<='deixa',\s')\w+|(?<='deixou',\s')\w+|(?<='deixei',\s')\w+|(?<='deixaram',\s')\w+|(?<='deixamos',\s')\w+|(?<='deixaria',\s')\w+|(?<='deixariam',\s')\w+|(?<='deixam',\s')\w+|(?<='deixo',\s')\w+|(?<='deixasse',\s')\w+|(?<='deixassem',\s')\w+|(?<='deixarmos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='começar',\s')\w+|(?<='começa',\s')\w+|(?<='começou',\s')\w+|(?<='comecei',\s')\w+|(?<='começaram',\s')\w+|(?<='começam',\s')\w+|(?<='começaria',\s')\w+|(?<='começariam',\s')\w+|(?<='começamos',\s')\w+|(?<='começo',\s')\w+|(?<='começasse',\s')\w+|(?<='começassem',\s')\w+|(?<='começarmos',\s')\w+|(?<='comece',\s')\w+|(?<='comecemos',\s')\w+",
    'V', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='chegar',\s')\w+|(?<='chega',\s')\w+|(?<='chegou',\s')\w+|(?<='cheguei',\s')\w+|(?<='chegaram',\s')\w+|(?<='chegam',\s')\w+|(?<='chegaria',\s')\w+|(?<='chegariam',\s')\w+|(?<='chegamos',\s')\w+|(?<='chego',\s')\w+|(?<='chegasse',\s')\w+|(?<='chegassem',\s')\w+|(?<='chegarmos',\s')\w+|(?<='chegue',\s')\w+|(?<='cheguemos',\s')\w+|(?<='cheguemos',\s')\w+",
    'V', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='ser',\s')\w+|(?<='é',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='somos',\s')\w+|(?<='seria',\s')\w+|(?<='seriam',\s')\w+|(?<='são',\s')\w+|(?<='sou',\s')\w+|(?<='era',\s')\w+|(?<='eram',\s')\w+|(?<='for',\s')\w+|(?<='formos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='sermos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='poder',\s')\w+|(?<='pode',\s')\w+|(?<='pôde',\s')\w+|(?<='pude',\s')\w+|(?<='puderam',\s')\w+|(?<='podemos',\s')\w+|(?<='poderia',\s')\w+|(?<='poderiam',\s')\w+|(?<='podem',\s')\w+|(?<='posso',\s')\w+|(?<='podia',\s')\w+|(?<='podiam',\s')\w+|(?<='pudesse',\s')\w+|(?<='pudessem',\s')\w+|(?<='podermos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='estar',\s')\w+|(?<='está',\s')\w+|(?<='esteve',\s')\w+|(?<='estive',\s')\w+|(?<='estiveram',\s')\w+|(?<='estamos',\s')\w+|(?<='estaria',\s')\w+|(?<='estariam',\s')\w+|(?<='estão',\s')\w+|(?<='estou',\s')\w+|(?<='estava',\s')\w+|(?<='estivemos',\s')\w+|(?<='estivesse',\s')\w+|(?<='estivéssemos',\s')\w+|(?<='estivessem',\s')\w+|(?<='estarmos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='ir',\s')\w+|(?<='vai',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='vamos',\s')\w+|(?<='iria',\s')\w+|(?<='iriam',\s')\w+|(?<='vão',\s')\w+|(?<='vou',\s')\w+|(?<='ía',\s')\w+|(?<='fomos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='iria',\s')\w+|(?<='vamos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='nũ',\s')\w+|(?<='né',\s')\w+|(?<='aí',\s')\w+", 'ADV', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='Whatsapp',\s')\w+|(?<='Instagram',\s')\w+|(?<='Facebook',\s')\w+|(?<='big',\s')\w+|(?<='brother',\s')\w+|(?<='buffet',\s')\w+|(?<='feedback',\s')\w+|(?<='fair',\s')\w+|(?<='play',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='open',\s')\w+|(?<='over',\s')\w+|(?<='photoshop',\s')\w+|(?<='pop',\s')\w+|(?<='plus',\s')\w+|(?<='réveillon',\s')\w+|(?<='sexy',\s')\w+|(?<='serial',\s')\w+|(?<='killer',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='shopping',\s')\w+|(?<='short',\s')\w+|(?<='show',\s')\w+|(?<='smartphone',\s')\w+|(?<='software',\s')\w+|(?<='telemarketing',\s')\w+|(?<='videogame',\s')\w+|(?<='tablet',\s')\w+|(?<='Windows',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='yes',\s')\w+|(?<='vip',\s')\w+|(?<='web',\s')\w+|(?<='smartphone',\s')\w+|(?<='slide',\s')\w+|(?<='states',\s')\w+|(?<='videogame',\s')\w+|(?<='online',\s')\w+|(?<='office',\s')\w+|(?<='offline',\s')\w+",
    '|EST', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(lambda x: re.sub(
    r"(?<='ficar',\s')\w+|(?<='fica',\s')\w+|(?<='ficou',\s')\w+|(?<='fiquei',\s')\w+|(?<='ficaram',\s')\w+|(?<='ficamos',\s')\w+|(?<='ficaria',\s')\w+|(?<='ficariam',\s')\w+|(?<='ficam',\s')\w+|(?<='fico',\s')\w+|(?<='ficasse',\s')\w+|(?<='ficassem',\s')\w+|(?<='ficarmos',\s')\w+",
    'VAUX', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='né',\s')\w+", 'ADV', str(x), flags=re.IGNORECASE))
df['transcricao_pos_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: re.sub(r"(?<='o',\s')\w+|(?<='os',\s')\w+", 'ART', str(x), flags=re.IGNORECASE))

df['ocorrencias_diante_pausas'] = df['transcricao_pos_pausas'].apply(
    lambda x: ' '.join(re.findall(r"\('\.\.\.'\,\s'\w+'\),\s(\(\'\w+',\s'\w+'\))", x)))

df['classes_diante_pausas'] = df['ocorrencias_diante_pausas'].apply(
    lambda x: ' '.join(re.findall(r"\w+[A-Z](?='\))", x)))
df['classes_diante_pausas_c'] = df['classes_diante_pausas'].apply(lambda x: Counter(x.split()))
df['palavras_diante_pausas'] = df['ocorrencias_diante_pausas'].apply(lambda x: ' '.join(re.findall(r"(?<=\(')\w+", x)))
df['palavras_diante_pausas_c'] = df['ocorrencias_diante_pausas'].apply(lambda x: Counter(x.split()))

df['qt_palavras'] = df['transcricao_limpa'].apply(lambda x: len(x.split()))

df.drop(['transcricao_sem_par', 'transcricao_sem_pausas_along', \
         'transcricao_sem_pausas_along_2', \
         'transcricao_sem_pausas_along_3', \
         'transcricao_sem_pausas_along_4', 'transcricao_com_pausas', 'transcricao_para_norm'], axis=1, inplace=True)

df.to_csv('df_ce.csv')
df.to_excel('df_ce.xlsx')

import pandas as pd
df = pd.read_csv('MICONTES.csv')
