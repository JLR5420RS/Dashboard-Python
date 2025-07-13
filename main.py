#imports necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import geopandas as gpd
import streamlit as st
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from wordcloud import WordCloud
import json


#padrão do gráfico
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 150

#configuração do streamlit
st.set_page_config(page_title="Dashboard - NAGEM", page_icon="😂", layout="wide",initial_sidebar_state="expanded")
st.title("Dashboard - NAGEM")

#carrega o dataset
df = pd.read_csv('dataset/RECLAMEAQUI_NAGEM.csv')

#criar colunas cidade e estado para separação do local
df['TEMPO'] = pd.to_datetime(df['TEMPO'])

df[['CIDADE', 'ESTADO']] = df['LOCAL'].str.split('-', n=1, expand=True)
df['CIDADE'] = df['CIDADE'].str.strip()
df['ESTADO'] = df['ESTADO'].str.strip()

#para ver no streamlit e retirar valores ausentes ou incorretos

#col1 , col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

#col1.write(df['STATUS'].value_counts())
#col2.write(df['CIDADE'].value_counts())
#col3.write(df['ESTADO'].value_counts())

df_filtrado = df[(df['ESTADO'] != 'naoconsta')&(df['ESTADO'] != '- - MA')]

#col1 , col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
#col1.write(df_filtrado['STATUS'].value_counts())
#col2.write(df_filtrado['CIDADE'].value_counts())
#col3.write(df_filtrado['ESTADO'].value_counts())

#st.write(df_filtrado)

# GRÁFICO 1: RECLAMAÇÕES POR ESTADO E CIDADE

group_level = st.radio("Visualizar por:", ['ESTADO', 'CIDADE'], horizontal=True)

# Filtro de ano
df_filtrado['ANO'] = df_filtrado['ANO'].astype(str)
anos_disponiveis = sorted(df_filtrado['ANO'].dropna().unique())
anos_selecionados = st.multiselect(
    "Filtrar por ANO:",
    options=anos_disponiveis,
    default=anos_disponiveis,
    key="multi_ano_bar"
)

# Botão para mostrar status com a barra dividida por eles
mostrar_por_status = st.toggle("Exibir por STATUS", value=False)

df_filtros = df_filtrado[df_filtrado['ANO'].isin(anos_selecionados)].copy()

# Cores
status_colors = {
    'Respondida': 'royalblue',
    'Resolvido': 'seagreen',
    'Em replica': 'gold',
    'Não respondida': 'orange',
    'Não resolvido': 'crimson',
}

# Visualização por estado
if group_level == 'ESTADO':
    if mostrar_por_status:
        df_grouped = df_filtros.groupby(['ESTADO', 'STATUS']).size().reset_index(name='count')
        fig = px.bar(
            df_grouped,
            x='ESTADO',
            y='count',
            color='STATUS',
            barmode='group',
            title='Reclamações por Estado e Status',
            labels={'ESTADO': 'Estado', 'count': 'Nº Reclamações', 'STATUS': 'Status'},
            text='count',
            color_discrete_map=status_colors
        )
        fig.update_traces(textposition='outside')
    else:
        df_grouped = df_filtros['ESTADO'].value_counts().reset_index()
        df_grouped.columns = ['ESTADO', 'count']
        fig = px.bar(
            df_grouped,
            x='ESTADO',
            y='count',
            title='Reclamações por Estado',
            labels={'ESTADO': 'Estado', 'count': 'Nº Reclamações'},
            text='count'
        )
        fig.update_traces(textposition='outside')

# 📌 Visualização por cidade
else:
    estados = df_filtros['ESTADO'].dropna().unique()
    if len(estados) == 0:
        st.warning("Nenhum estado disponível com os filtros selecionados.")
        st.stop()

    estado_selecionado = st.selectbox("Selecione o Estado:", sorted(estados), key="select_estado_bar")
    cidades = df_filtros[df_filtros['ESTADO'] == estado_selecionado]['CIDADE'].dropna().unique()

    cidades_selecionadas = st.multiselect(
        "Selecione uma ou mais Cidades:",
        sorted(cidades),
        default=sorted(cidades),
        key="multi_cidade_bar"
    )

    df_cidade = df_filtros[
        (df_filtros['ESTADO'] == estado_selecionado) &
        (df_filtros['CIDADE'].isin(cidades_selecionadas))
    ]

    if mostrar_por_status:
        df_grouped = df_cidade.groupby(['CIDADE', 'STATUS']).size().reset_index(name='count')
        fig = px.bar(
            df_grouped,
            x='CIDADE',
            y='count',
            color='STATUS',
            barmode='group',
            title=f'Reclamações por Cidade e Status em {estado_selecionado}',
            labels={'CIDADE': 'Cidade', 'count': 'Nº Reclamações', 'STATUS': 'Status'},
            text='count',
            color_discrete_map=status_colors
        )
        fig.update_traces(textposition='outside')
    else:
        df_grouped = df_cidade['CIDADE'].value_counts().reset_index()
        df_grouped.columns = ['CIDADE', 'count']
        fig = px.bar(
            df_grouped,
            x='CIDADE',
            y='count',
            title=f'Reclamações por Cidade em {estado_selecionado}',
            labels={'CIDADE': 'Cidade', 'count': 'Nº Reclamações'},
            text='count'
        )
        fig.update_traces(textposition='outside')

# Gráfico no streamlit
fig.update_xaxes(categoryorder="total descending")
fig.update_layout(title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

# GRÁFICO 2: FREQUÊNCIA POR TIPO DE STATUS

col1,col2 = st.columns(2)

anos_disponiveis_graf2 = sorted(df_filtrado['ANO'].dropna().unique())

# Seleção de anos e filtro
anos_selecionados_graf2 = col1.multiselect(
    "Selecione um ou mais Anos:",
    options=anos_disponiveis_graf2,
    default=anos_disponiveis_graf2
)

if anos_selecionados_graf2:
    df_status = df_filtrado[df_filtrado['ANO'].isin(anos_selecionados_graf2)]
else:
    df_status = df_filtrado.copy()

# Contador
status_counts = df_status['STATUS'].value_counts().reset_index()
status_counts.columns = ['STATUS', 'count']

# Cores e plot do gráfico
status_colors = {
    'Respondida': 'royalblue',
    'Resolvido': 'seagreen',
    'Em replica': 'yellow',
    'Não respondida': 'orange',
    'Não resolvido': 'crimson',
}

fig = px.bar(
    status_counts,
    x='STATUS',
    y='count',
    title=f'Frequência por Tipo de STATUS ({", ".join(map(str, anos_selecionados_graf2))})',
    labels={'STATUS': 'Status', 'count': 'Quantidade'},
    text_auto=True,
    color='STATUS',
    color_discrete_map=status_colors
)

fig.update_layout(
    title_x=0.5,
    xaxis_tickangle=-45,
    margin=dict(t=50, l=10, r=10, b=10)
)

# Gráfico no streamlit
col1.plotly_chart(fig, use_container_width=True)

# GRÁFICO 3: DISTRIBUIÇÃO DE PALAVRAS

# 🔹 Garantir que 'ANO' é string
df_filtrado['ANO'] = df_filtrado['ANO'].astype(str)

# 🔹 Filtro por ANO
anos_disponiveis = sorted(df_filtrado['ANO'].dropna().unique())
anos_selecionados = col2.multiselect(
    "Filtrar por ANO:",
    options=anos_disponiveis,
    default=anos_disponiveis,
    key="multi_ano_3"
)

# Filtros de estado e status
estados_graf3 = sorted(df_filtrado['ESTADO'].dropna().unique())
status_graf3 = sorted(df_filtrado['STATUS'].dropna().unique())

# Estados (com checkbox)
selecionar_todos_estados = col2.checkbox("Selecionar todos os ESTADOS", value=True, key="check_estados_1")

if selecionar_todos_estados:
    estados_selecionados = estados_graf3
else:
    estados_selecionados = col2.multiselect(
        "Filtrar por ESTADO:",
        options=estados_graf3,
        default=[],
        key="multi_estados_1"
    )

# Status
status_selecionado = col2.multiselect(
    "Filtrar por STATUS:",
    options=status_graf3,
    default=status_graf3,
    key="multi_status_1"
)

# Aplicando filtros
df_dist = df_filtrado[
    df_filtrado['ANO'].isin(anos_selecionados) &
    df_filtrado['ESTADO'].isin(estados_selecionados) &
    df_filtrado['STATUS'].isin(status_selecionado)
].copy()

# Verificar se não tem dados após filtro
if not df_dist.empty:
    # Função de contagem de palavras
    def count_palavras(texto):
        return len(str(texto).split())

    # Aplicar a contagem
    word_counts = df_dist['DESCRICAO'].dropna().apply(count_palavras)

    # Criar gráfico
    fig = ff.create_distplot(
        [word_counts],
        group_labels=["Palavras por descrição"],
        show_hist=True,
        show_rug=False
    )

    fig.update_layout(
        title="Distribuição de palavras por descrição",
        xaxis_title="Número de palavras",
        yaxis_title="Densidade",
        title_x=0.5,
        margin=dict(t=50, l=10, r=10, b=10)
    )

# Gráfico no streamlit
    col2.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

else:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")

#GRÁFICO 4: LINHA TEMPORAL

df_filtrado['ANO'] = df_filtrado['ANO'].astype(str)
df_filtrado['MES'] = df_filtrado['MES'].astype(str).str.zfill(2)
df_filtrado['DIA'] = df_filtrado['DIA'].astype(str).str.zfill(2)

# Filtro de estado com checkbox
estados_graf4 = sorted(df_filtrado['ESTADO'].dropna().unique())

selecionar_todos_estados = st.checkbox("Selecionar todos os ESTADOS", value=True,key="check_estados_2")

if selecionar_todos_estados:
    estados_selecionados = estados_graf4
else:
    estados_selecionados = st.multiselect(
        "Filtrar por ESTADO:",
        options=estados_graf4,
        key="multi_estados_2",
        default=[]
    )

# Filtro de status
status_graf4 = sorted(df_filtrado['STATUS'].dropna().unique())
status_selecionado = st.multiselect(
    "Filtrar por STATUS:",
    options=status_graf4,
    default=status_graf4,
    key="multi_status_2"
)

# Filtro de granularidade por Ano, Ano e Mês; e Ano, Mês e Dia
granularidade = st.selectbox(
    "Agrupar por:",
    ['Ano', 'Ano e Mês', 'Ano, Mês e Dia']
)

# Filtros aplicados
df_ft = df_filtrado[
    df_filtrado['ESTADO'].isin(estados_selecionados) &
    df_filtrado['STATUS'].isin(status_selecionado)
].copy()

# Criação da granularidade no gráfico
if granularidade == 'Ano':
    df_ft['DATA_GRUPO'] = df_ft['ANO']
elif granularidade == 'Ano e Mês':
    df_ft['DATA_GRUPO'] = df_ft['ANO'] + '-' + df_ft['MES']
else:
    df_ft['DATA_GRUPO'] = df_ft['ANO'] + '-' + df_ft['MES'] + '-' + df_ft['DIA']

df_ft['DATA_GRUPO'] = pd.to_datetime(df_ft['DATA_GRUPO'], errors='coerce')
df_ftd = df_ft.dropna(subset=['DATA_GRUPO']).sort_values('DATA_GRUPO')

# Checkar se todos os status estão selecionados para juntá-los em uma só linha
todos_status = set(status_selecionado) == set(status_graf4)

if todos_status:
    # Uma só linha para todos somados
    df_agrupado = df_ftd.groupby('DATA_GRUPO').size().reset_index(name='CONTAGEM')
    fig = px.line(
        df_agrupado,
        x='DATA_GRUPO',
        y='CONTAGEM',
        title=f'Série Temporal Total por {granularidade}',
        labels={'DATA_GRUPO': 'Data', 'CONTAGEM': 'Nº Reclamações'}
    )
else:
    # Cada status uma linha
    df_agrupado = df_ftd.groupby(['DATA_GRUPO', 'STATUS']).size().reset_index(name='CONTAGEM')
    fig = px.line(
        df_agrupado,
        x='DATA_GRUPO',
        y='CONTAGEM',
        color='STATUS',
        title=f'Série Temporal por {granularidade} e STATUS',
        labels={'DATA_GRUPO': 'Data', 'CONTAGEM': 'Nº Reclamações', 'STATUS': 'Status'}
    )

fig.update_layout(
    title_x=0.5,
    margin=dict(t=50, l=10, r=10, b=10)
)

# Gráfico no streamlit
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# GRÁFICO 5: WORDCLOUD

st.subheader("Palavras mais Frequentes nas Descrições")

# Baixear stopwords
nltk.download('stopwords')

# Filtros interativos
df_filtrado['ANO'] = df_filtrado['ANO'].astype(str)

# Ano
anos_disponiveis_graf5 = sorted(df_filtrado['ANO'].dropna().unique())
anos_selecionados_graf5 = st.multiselect(
    "Filtrar por ANO:",
    options=anos_disponiveis_graf5,
    default=anos_disponiveis_graf5,
    key="ano_wc"
)

# Estado
estados_disponiveis_graf5 = sorted(df_filtrado['ESTADO'].dropna().unique())

selecionar_todos_estados = st.checkbox("Selecionar todos os ESTADOS", value=True,key="check_estados_3")
if selecionar_todos_estados:
    estados_selecionados_graf5 = estados_disponiveis_graf5
else:
    estados_selecionados_graf5 = st.multiselect(
    "Filtrar por ESTADO:",
    options=estados_disponiveis_graf5,
    default=estados_disponiveis_graf5,
    key="estado_wc"
)

# Status
status_disponiveis_graf5 = sorted(df_filtrado['STATUS'].dropna().unique())
status_selecionados_graf5 = st.multiselect(
    "Filtrar por STATUS:",
    options=status_disponiveis_graf5,
    default=status_disponiveis_graf5,
    key="status_wc"
)

# Aplicar os filtros
df_wc = df_filtrado[
    df_filtrado['ANO'].isin(anos_selecionados_graf5) &
    df_filtrado['ESTADO'].isin(estados_selecionados_graf5) &
    df_filtrado['STATUS'].isin(status_selecionados_graf5)
].copy()

# Stopwords customizadas
novas_stopwords = [
    "comprei", "comprar", "consertar", "ajeitar", "Nagem", "nagem", "atendido", "atendida", "atendimento", "atendente", "teste",
    "testou", "testaram", "aparelho", "computador", "notebook", "TV", "televisor", "celular", "comprado", "solicitei", "solicitado",
    "reclamação", "produto", "insatisfeito", "insatisfeita", "monitor", "defeito", "falha", "impressora", "reclamações", "pedi", 
    "reclame", "dia", "dias", "voltar", "fazer", "nota", "empresa", "seguro", "garantia", "cliente", "clientes", "constrangido",
    "constrangida", "problema", "loja", "educação", "cartucho", "favor", "muito", "pouco", "voltar", "falei", "falaram", ".", ","
]
stop_pt = set(nltk_stopwords.words("portuguese")) | set(novas_stopwords)

# Gerar o texto para wordcloud
texto = " ".join(df_wc['DESCRICAO'].dropna().astype(str).tolist())

# gerar a wordcloud e gráfico no streamlit
if texto and not texto.isspace():
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stop_pt,
            colormap='viridis',
            max_words=50
        ).generate(texto)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocorreu um erro ao gerar a nuvem de palavras: {e}")
else:
    st.info("Não há dados de texto suficientes para gerar a nuvem de palavras com os filtros selecionados.")

#GRÁFICO 6: MAPA

# Carrega Geojson
with open('dataset/brazil-states.geojson', 'r') as f:
    geojson = json.load(f)

# Lista de UFs do GeoJSON
ufs_geojson = [f['properties']['sigla'] for f in geojson['features']]

# Filtro por ANO
df_filtrado['ANO'] = df_filtrado['ANO'].astype(str)
anos_disponiveis_graf6 = sorted(df_filtrado['ANO'].dropna().unique())
anos_selecionados_graf6 = st.multiselect(
    "Filtrar por ANO:",
    options=anos_disponiveis_graf6,
    default=anos_disponiveis_graf6,
    key="ano_mapa"
)

df_ano = df_filtrado[df_filtrado['ANO'].isin(anos_selecionados_graf6)].copy()

# Contagem por estado
df_mapa = df_ano['ESTADO'].value_counts().reset_index()
df_mapa.columns = ['ESTADO', 'count']

# Garantir todos os estados no mapa
todos_estados = pd.DataFrame({'ESTADO': ufs_geojson})
df_mapa = todos_estados.merge(df_mapa, on='ESTADO', how='left').fillna(0)

# Geração do mapa
fig_mapa = px.choropleth(
    df_mapa,
    geojson=geojson,
    locations='ESTADO',
    featureidkey="properties.sigla",
    color='count',
    hover_name='ESTADO',
    color_continuous_scale="Reds",
    labels={'count': 'Nº Reclamações'}
)

fig_mapa.update_geos(
    fitbounds="locations",
    visible=False,
    projection_type="mercator"
)
fig_mapa.update_layout(
    title="Mapa de Reclamações por Estado",
    title_x=0.5,
    height=650,  # Aumentar o tamanho
    margin={"r":0,"t":50,"l":0,"b":0}
)

# Gráfico no streamlit
st.plotly_chart(fig_mapa, use_container_width=True)


