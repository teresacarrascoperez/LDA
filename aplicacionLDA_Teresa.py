# -*- coding: utf-8 -*-
"""
@author: TERESA CARRASCO PÉREZ
"""
###############################################################################
#      APRENDIZAJE AUTOMATICO: INTRODUCCIÓN AL MODELADO DE TEMAS              #
###############################################################################

# Bibiotecas y paquetes requeridos para procesar los tuits

# Base y limpieza de tuits 
import json
import pandas as pd
import emoji
import re
import contractions
import nltk
nltk.download('stopwords')

# VIsualización de los resultados obtenidos
import matplotlib.pyplot as plt 
import pyLDAvis
import pyLDAvis.gensim_models
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from wordcloud import WordCloud

# Procesamiento del lenguaje natural (NLP) y aplicación Latent Dirichlet 
# Allocation
import spacy
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from collections import Counter



#------------------------------------------------------------------------------

# MÉTODOS PARA CARGAR Y COGER ÚNICAMENTE EL TEXTO DE LOS TUITS PARA SU ANÁLISIS

# Esta función realiza la carga de todos los tuits desde los archivos previamente
# descargados de la API de Twitter   
def cargar_datos(ruta_tuits):
    with open(ruta_tuits) as ifile:
        tuits = [json.loads(line) for line in ifile]
    #print(tuits)
    return tuits

# El siguiente método devuelve una lista de diccionarios donde se realiza un 
# filtro para obtener únicamente tuits en inglés.
def filtrado_de_lenguaje(tuits):
    lista_dicc = []
    long = len(tuits)
    for j in range (long):
        if (tuits[j]['lang']=='en'): #Se filtra por inglés
            # Se añaden a la lista vacía creada los tuits seleccionados
            lista_dicc.append(tuits[j]) 
    #print(list_dict)
    return lista_dicc

# Posteriormente, de todos los tuits, para el análisis del presente sólo 
# sse necesita utilizar el campo texto. Por tanto, se procede a su extracción
# mediante el campo texto.
def extraccion_texto_completo(tuits):
    lista_textos = [] # Se crea lista vacía para añadir los textos
    for tuit in tuits:
        # Se distinguen tres casos para extraer el texto:
        # 1º) Si el tuit es largo, el texto del tuit se encuentra en el campo 'full_text'.
        if 'extended_tweet' in tuit:
            #print(tuit['extended_tweet']['full_text'])
            lista_textos.append(tuit['extended_tweet']['full_text'])
        # 2º) Si el tuit es un retuit, el texto del correpondiente retuit se encuentra dentro de 'full_text' pero accediendo primero a 'retweeted_status'.  
        elif not ('extended_tweet' in tuit) and ('retweeted_status' in tuit) and ('extended_tweet' in tuit['retweeted_status']):
            text_retuit = tuit['retweeted_status']['extended_tweet']['full_text']
            lista_textos.append(text_retuit)
            #print(text_retuit)
        # 3º) Si el tuit es corto, el texto se encuentra directamente en el campo 'text' y no existe campo 'full_text' para estos tuits
        else:
            tuit_corto = tuit['text']
            lista_textos.append(tuit_corto)
            #print(tuit_corto)
    #print(lista_textps)
    return lista_textos

# Esta función crea una lista de diccionarios con la extracción de los textos de los tuits.
# Creo un diccionario para cada tuit con clave 'text_extraction' y valor, el texto extraido de cada tuit.
def lista_dicc_textos(tuits):
    clave_dicc = ['text_extraction'] # Creo la plave del diccionario
    lista_dicc = []
    long = len(tuits)
    for i in range (long):
        valor = [tuits[i]]
        dicc = dict(zip(clave_dicc,valor)) # Creo diccionario por cada tuit
        lista_dicc.append(dicc) # Añado a la lista cada diccionario creado
    #print(lista_dicc)
    return lista_dicc

#------------------------------------------------------------------------------

# MÉTODOS PARA REALIZAR LA LIMPIEZA DEL CONJUNTO DE TUITS 

# Función que dada un tuit, devuelve la misma pero transformandolas
# mayúsculas a minúsculas minúsculas
def tuit_a_minusculas(tuit): 
    return tuit.lower()


# Método que elimina todos los emoticonos de un tuit
def eliminacion_emojis(tuit):
    return emoji.replace_emoji(tuit, replace='') 


# Función que dado un tuit, elimina las URLs que contiene mediante el uso de 
# la expresión regular indicada
def eliminacion_urls(tuit):
    tuit = re.sub(r'http\S+', '', tuit)
    return tuit


# Dada un tuit se eliminan las contracciones de manera que se obtienen
# la forma expandida de las mismas para evitar que una contracción y su palabra
# extendida, sean consideradas distintas en el análisis posterior
def expandir_contracciones(tuit):
    palabras_expandidas = []
    tuit_dividido = tuit.split() # Se divide el tuit en palabras
    for palabra in tuit_dividido:
       palabras_expandidas.append(contractions.fix(palabra)) # Se modifica cada palabra para expandirla
    # Se unen de nuevo la lista de palabras para generar el tuit sin contracciones
    return (' '.join(palabras_expandidas)) 


# Esta función elimina tanto los hastags como las menciones incluidas en un tuit
def eliminacion_hastags_menciones(tuit):
    # Se utilizan las expresiones regulares siguientes para eliminar ambos elementos.
    tuit = re.sub(r'@\S+','', tuit)
    tuit = re.sub(r'#\S+','', tuit)
    return tuit


# Esta función elimina todos los símbolos de puntuación y caracteres especiales del tuit.
def elim_caracteres_especiales(tuit):
    return re.sub(r'[^a-z]', ' ', tuit) 

# La siguiente función engloba todas las anteriores aplicando todas las funciones
# definidas para realizar la limpieza al conjunto de tuits de entrada
def limpieza_tuits(tuits):
    tuits_filtrados = [] # Se crea una lista para incluir todos los tuits listos para aplicar NLP
    for tuit in tuits:
        tuit_minus = tuit_a_minusculas(tuit)
        tuit_sin_emoji = eliminacion_emojis(tuit_minus)
        tuit_sin_urls = eliminacion_urls(tuit_sin_emoji)
        tuit_sin_contrac = expandir_contracciones(tuit_sin_urls)
        tuit_sin_hm = eliminacion_hastags_menciones(tuit_sin_contrac)
        tuit_sin_ce = elim_caracteres_especiales(tuit_sin_hm)
        tuits_filtrados.append(tuit_sin_ce)
    return tuits_filtrados


#------------------------------------------------------------------------------

# MÉTODOS PARA REALIZAR el PREPROCESAMIENTO DE LOS TUITS

nlp = spacy.load('en_core_web_lg') # Canalización de Spacy entrenada en inglés

# A continuación se eliminan las palabras vacías y aquellas cuya longitud es inferior a 3
# Además se incluye una lista adicional de palabras vacías que en este caso como
# ya se sabe que los tuits son de la invasión rusa de Ucrania, no aportan para la
# obtención de temas que posteriormente se obtendrá mediante LDA. 
def eliminacion_stopwords(tuit):
    long_min = 2
    # Palabras vacías customizadas
    stopwords_adicionales = {'kiev','kyiv','russia','vladimir','putin','retweet','ukraine','russian','ukrainian','amp','\n','\n\n','sir','&amp;', 'got', 'want', 'like'}
    # Se añaden las palabras vacías a la lista por defecto que se incluye en "nlp".
    nlp.Defaults.stop_words |= stopwords_adicionales
    tuit_sin_sw =[] 
    tuit_nlp = nlp(tuit)
    for token in tuit_nlp:
        if token.is_stop == False: # En el caso de que el token no sea una stopword
            if len(token.text) > long_min: # Solo tomamos los tokens con longitud > 2
                tuit_sin_sw.append(token.text)
    return (' '.join(tuit_sin_sw)) # Se devuelve un tuit sin stopwordst


# Se generaliza la función anterior para el conjunto de tuits de entrada
def eliminacion_sw_tuits(tuits):
    lista_tuit_sin_sw = []
    for tuit in tuits:
        resultado = eliminacion_stopwords(tuit)
        lista_tuit_sin_sw.append(resultado)
    return lista_tuit_sin_sw


# Este método realiza la lematización que genera la raíz de la palabra, se hace
# uso del vocabulario y del análisis morfológico de las palabras.
# Se utiliza la biblioteca spaCy para realizar la lematización.
def lematizacion(tuit):
    tuit_nlp = nlp(tuit)
    tuit_lematizado = []
    for token in tuit_nlp:
        tuit_lematizado.append(token.lemma_)
    return (' '.join(tuit_lematizado))

# Se genera mediante esta función la lista de tokens eliminando espacios adicionales
# que se han añadido para realizar la limpieza anterior
def tuits_tokens(tuit):
    lista_tokens = []
    tuit_separado = tuit.split(' ')
    for palabra in tuit_separado:
        # Se eliminan los espacios adicionales incluidos en los tuits
        if palabra != '': # En el caso de que la palabra no sea la cadena vacía
            lista_tokens.append(palabra)
    return lista_tokens


#------------------------------------------------------------------------------

# DEFINICIÓN DE LAS MÉTRICAS DE COHERENCIA

# Definición de la métrica de coherencia: C_umass para intentar optimizar el número de temas
def coherencia_c_umass(corpus, diccionario):
    lista_temas = []
    lista_puntuacion = []
    for i in range(1,20,1):
        modelo_lda = LdaMulticore(corpus=corpus, id2word=diccionario, iterations=20, num_topics=i, workers = 4, passes=10, random_state=1)
        cm = CoherenceModel(model=modelo_lda, corpus=corpus, dictionary=diccionario, coherence='u_mass')
        lista_temas.append(i)
        lista_puntuacion.append(cm.get_coherence())
    return (lista_temas, lista_puntuacion)


# Definición de la métrica de coherencia: C_v para intentar optimizar el número de temas
def coherencia_c_v(corpus, diccionario, df):
    tuits = df['tuits_tokenizados'] # Elijo del data frame la columna con los tuits tokenizados
    lista_temas = []
    lista_puntuacion = []
    for i in range(1,20,1):
        modelo_lda = LdaMulticore(corpus=corpus, id2word=diccionario, iterations=20, num_topics=i, workers = 4, passes=10, random_state=1)
        cm = CoherenceModel(model=modelo_lda, texts = tuits, corpus=corpus, dictionary=diccionario, coherence='c_v')
        lista_temas.append(i)
        lista_puntuacion.append(cm.get_coherence())
    return (lista_temas, lista_puntuacion)


# Creación de un nuevo data frame para realizar la representacion que contiene las palabras, id, peso y un contador.
def nuevo_data_frame(lda_model2, df):
    # Lista con los temas que contiene tuplas con las palabras y sus pesos correspondientes.
    temas_optimos = lda_model2.show_topics(formatted=False)
    tuits_procesados = df['tuits_tokenizados']
    lista_tokens = [palabra for lista_palabras in tuits_procesados for palabra in lista_palabras] #lista con todos los tokens de todos los tuits
    contador = Counter(lista_tokens) # Contador con el número de veces que aparece cada token
    resultado = []
    for i, tema in temas_optimos:
        for palabra, peso in tema:
            resultado.append([palabra, i , peso, contador[palabra]])       
    df2 = pd.DataFrame(resultado, columns=['word', 'id_tema', 'peso', 'count'])
    return df2


def temas_por_documento(model, corpus, start=0, end=1):
    corpus_seleccion = corpus[start:end]
    temas_dominantes = []
    porcentaje_temas = []
    for i, corp in enumerate(corpus_seleccion):
        temasporcent = model[corp]
        tema_dominante = sorted(temasporcent, key = lambda x: x[1], reverse=True)[0][0]
        temas_dominantes.append((i, tema_dominante))
        porcentaje_temas.append(temasporcent)
    return(temas_dominantes, porcentaje_temas)




if __name__ == "__main__":
    ruta = '/Users/.........'
    l = cargar_datos(ruta)

    # Se realiza un filtrado del lenguaje, se extrae el texto de los tuits
    tuits_ingles = filtrado_de_lenguaje(l)
    print(len(tuits_ingles)) #34.184 tuits en inglés
    texto_tuits = extraccion_texto_completo(tuits_ingles)
    
    # Se crea una lista de diccionarios para crear un data frame
    diccionario_tuits = lista_dicc_textos(texto_tuits)
    
    #Data frame con construido a partir de la lista de diccionarios
    df = pd.DataFrame(diccionario_tuits)
    
    # A continuación se crean dos columnas para comprobar que las funciones definidas
    # realizan lo correcto. (Son estaas dos lineas de código se pueden omitir, se añadido
    # para la obtención de ejemplos para la memoria)
    # Se crea una columna en la que aparecen los tuits sin emoticonos
    df['tuits_sin_emojis'] = df['text_extraction'].apply(eliminacion_emojis)
    # Se crea una columna que contiene los tuits anteriores sin urlsC
    df['tuits_sin_urls'] = df['tuits_sin_emojis'].apply(eliminacion_urls)
    
    # Se aplica la limpieza a todos los tuits
    limpieza_tuits = limpieza_tuits(texto_tuits)
    df['tuits_analizados'] = limpieza_tuits # Se añade la columna al data frame con todos la limpieza de los tuits
    
    # Se procede a realizar el preprocesamiento de los tuits eliminando stopwords, 
    # aplicando la lematizacion y tokenización
    tuits_sw = eliminacion_sw_tuits(limpieza_tuits)
    df['tuits_sin_sw'] = tuits_sw # Columna en el data frame sin stopwords
    df['lematizacion'] = df['tuits_sin_sw'].apply(lematizacion) # Columna en el data frame con la lematización aplicada
    df['tuits_tokenizados'] = df['lematizacion'].apply(tuits_tokens) # Columna en el data frame con los tokens de los tuits
    
    #--------------------------------------------------------------------------
    # APLICACIÓN LDA (Asignación Latente de Dirichlet)
    #--------------------------------------------------------------------------
    
    # Creación del diccionario mediante el método de Gensim corpora.dictionary
    # Se le asigna a cad token una identificación única: ID. 
    diccionario = Dictionary(df['tuits_tokenizados'])
    #print(diccionario.token2id) #Traza omitible, empleada para la obtendición de ejemplos
    # Se filtra el diccionario para eliminar aquellos elementos tokens que aparecen en menos de 5 tuits
    diccionario.filter_extremes(no_below=5, no_above = 0.7, keep_n = 10000)

    # Se crea el corpus que contiene el ID de cada token y su frecuencia correspondiente
    corpus = [diccionario.doc2bow(doc) for doc in df['tuits_tokenizados']]
    
    # Aplicación del LDA insertando el corpus y diccionario construídos
    lda_model1 = LdaMulticore(corpus=corpus, id2word=diccionario, iterations=50, num_topics=10, workers = 4, passes=10)

    # Se imprimen los 10 temas obtenidos con las palabras más representativas de cada tema
    for id_palabra, tema in lda_model1.print_topics(-1):
        print('Topic: {} \nWords: {}\n'.format(id_palabra , tema))

    # Primer gráfico para la métrica de Coherencia C_UMass
    (lista_temas1, puntuacion1) = coherencia_c_umass(corpus, diccionario)
    plt.plot(lista_temas1, puntuacion1)
    plt.xticks([i for i in range(len(lista_temas1)+1)])
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score C_umass')
    plt.show()
    
    # Segundo gráfico para la métrica de Coherencia C_v
    (lista_temas2, puntuacion2) = coherencia_c_v(corpus, diccionario, df)
    plt.plot(lista_temas2, puntuacion2)
    plt.xticks([i for i in range(len(lista_temas1)+1)])
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score C_v')
    plt.show()

    # Según la coherencia C_v obtenida se decide tomar 4 temas como óptimo. 
    # Se aplica de nuevo LDA para obtener un resultado óptimo a nivel de coherencia
    lda_model2 = LdaMulticore(corpus=corpus, id2word=diccionario, iterations=100, num_topics=4, workers = 4, passes=100)

    # Se imprimen los 4 temas obtenidos con las palabras más representativas de cada tema
    for id_palabra, tema in lda_model2.print_topics(-1):
        print('Topic: {} \nWords: {}\n'.format(id_palabra , tema))


    #--------------------------------------------------------------------------
    # VISUALIZACIÓN DE TEMAS UTILIZANDO pyLDAvis DE GENSIM
    #--------------------------------------------------------------------------
    lda_visualizacion = pyLDAvis.gensim_models.prepare(lda_model2, corpus, diccionario)
    pyLDAvis.display(lda_visualizacion)

    # Se guarda la correspondiente representación en formato HTML
    pyLDAvis.save_html(lda_visualizacion, '/Users/teresa/Desktop/tfg/visualizacion.html')
    
    # Creación de un nuevo data frame para realizar la representacion que contiene las palabras, id, peso y un contador.
    df2 = nuevo_data_frame(lda_model2, df)
       

    # REPRESENTACIÓN GRÁFICA DEL PESO DE LAS PALABRAS POR TEMA
    fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height='peso', data=df2.loc[df2.id_tema==i, :], color=cols[i], width=0.6, alpha=0.6,label='Weights')
        ax.set_ylim(0, 0.065)
        ax.set_title('TOPIC: ' + str(i), color=cols[i], fontsize=18)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df2.loc[df2.id_tema==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper right',fontsize='large')
    fig.tight_layout(w_pad=2)    
    fig.suptitle('Importance of Topic Keywords', fontsize=28, y=1.05)    
    plt.show()
   
    
    # REPRESENTACIÓN GRÁFICA DE LA NUBE DE PALABRAS. Se obtiene una representación
    # más visual de las palabras más relevantes que definen cada tema
    temas = lda_model2.show_topics(formatted=False)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  
    cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)
    # En un mismo gráfico realizamos 4 subgráficos
    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        palabras_temas = dict(temas[i][1])
        cloud.generate_from_frequencies(palabras_temas, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('TOPIC ' + str(i), fontdict=dict(size=18))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    
    
    
    # REPRESENTACIÓN GRÁFICA DE LOS TEMAS DOMINANTES EN CADA COCUMENTO
    temas_dominantes, porcentaje_temas = temas_por_documento(lda_model2, corpus, end=-1)           
    df3 = pd.DataFrame(temas_dominantes, columns=['Document_Id', 'Dominant_Topic'])
    tema_dominante_por_tuit = df3.groupby('Dominant_Topic').size()
    df_tema_dominante_por_tuit = tema_dominante_por_tuit.to_frame(name='count').reset_index()

    # REPRESENTACIÓN GRÁFICA DE LA DISTRIBUCIÓN DE TEMAS POR PESO DE CADA TEMA
    porcentaje_tema_por_tuit = pd.DataFrame([dict(t) for t in porcentaje_temas])
    df_porcentaje_tema_por_tuit = porcentaje_tema_por_tuit.sum().to_frame(name='count').reset_index()
    

    # Se utiliza una gráfica con dos subplots para hacer en la memoria el análisis comparativo
    # más sencillo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

    ax1.bar(x='Dominant_Topic', height='count', data=df_tema_dominante_por_tuit, width=.5,color = 'darkturquoise')
    ax1.set_xticks(range(df_tema_dominante_por_tuit.index.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x))
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')
    ax1.set_ylim(0, 14000)

    ax2.bar(x='index', height='count', data=df_porcentaje_tema_por_tuit, width=.5,color='royalblue')
    ax2.set_xticks(range(df_porcentaje_tema_por_tuit.index.unique().__len__()))
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))
    plt.show()