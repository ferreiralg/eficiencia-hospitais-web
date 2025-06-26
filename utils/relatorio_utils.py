import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import locale
import io
import tempfile
import re
import textwrap
import json
import ast
import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import Paragraph, Frame, Table, Spacer, TableStyle, Image

# Configurações globais
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# ============================
# Funções de formatação
# ============================

def formatar_milhares(x: float, pos: int) -> str:
    """Formata números com separadores de milhar."""
    return '{:,.0f}'.format(x)

def capitalizar_palavras(sentence: str) -> str:
    """Capitaliza as primeiras letras de cada palavra importante."""
    if isinstance(sentence, str):
        words = sentence.split()
        capitalized_words = [word.capitalize() if len(word) > 2 and not word.isdigit() else word.lower() for word in words]
        return ' '.join(capitalized_words)
    return sentence

def formatar_nome(name: str) -> str:
    """Formata nomes para padronização de maiúsculas/minúsculas."""
    def format_word(word):
        if len(word) in [1, 2]:
            return word.lower()
        return word.capitalize()

    # Converter todas as palavras para minúsculas e dividir o nome em palavras
    words = re.split(r'\s+', name.lower())
    # Aplicar a formatação a cada palavra e juntá-las novamente
    formatted_name = ' '.join(format_word(word) for word in words)
    return formatted_name

def formatar_data(data_num: Union[int, str], tipo: int) -> str:
    """Converte data numérica (AAAAMM) para formato de texto.
    
    Args:
        data_num: Data no formato AAAAMM (ex: 202401)
        tipo: Tipo de formatação (1 = nome completo, 2 = abreviado)
        
    Returns:
        String formatada (ex: 'Janeiro-24' ou 'Jan-24')
    """
    # Converte o número em uma string
    data_str = str(data_num)
    
    # Extrai o ano e o mês da string
    ano = data_str[2:4]
    mes = data_str[4:6]
    
    # Converte o mês numérico para o mês abreviado em texto em português
    mes_abrev = datetime.datetime.strptime(mes, '%m').strftime('%b').capitalize()

    # Dicionários para substituir os meses em inglês pelos meses em português
    meses_em_portugues = {
        1: {
            'Jan': 'Janeiro',
            'Fev': 'Fevereiro',
            'Mar': 'Março',
            'Abr': 'Abril',
            'Mai': 'Maio',
            'Jun': 'Junho',
            'Jul': 'Julho',
            'Ago': 'Agosto',
            'Set': 'Setembro',
            'Out': 'Outubro',
            'Nov': 'Novembro',
            'Dez': 'Dezembro'
        },
        2: {
            'Jan': 'Jan',
            'Fev': 'Fev',
            'Mar': 'Mar',
            'Abr': 'Abr',
            'Mai': 'Mai',
            'Jun': 'Jun',
            'Jul': 'Jul',
            'Ago': 'Ago',
            'Set': 'Set',
            'Out': 'Out',
            'Nov': 'Nov',
            'Dez': 'Dez'
        }
    }

    # Seleciona o dicionário correto com base no tipo
    meses_em_portugues_selecionado = meses_em_portugues[tipo]

    # Obtém o mês abreviado em português
    mes_abrev_portugues = meses_em_portugues_selecionado[mes_abrev]
     
    # Combina o mês abreviado e o ano
    data_formatada = f"{mes_abrev_portugues}-{ano}"
     
    return data_formatada

def criar_dataframe_comparacao(dados: List[List[Union[str, float, int]]]) -> pd.DataFrame:
    """Converte dados de comparação para DataFrame formatado.
    
    Args:
        dados: Lista de dados de comparação
        
    Returns:
        DataFrame formatado com as colunas Parâmetro, Original, Alvo e Variação (%)
    """
    # Create DataFrame
    df = pd.DataFrame(dados, columns=['Parâmetro', 'Original', 'Alvo', 'Variação (%)'])
    
    # Convert columns to numeric, removing dots if any
    df[['Original', 'Alvo', 'Variação (%)']] = df[['Original', 'Alvo', 'Variação (%)']].replace('\\.', '', regex=True)
    df[['Original', 'Alvo', 'Variação (%)']] = df[['Original', 'Alvo', 'Variação (%)']].apply(pd.to_numeric)
    
    # Function to format numbers with thousands separator
    def format_with_thousands_separator(x):
        return '{:,.0f}'.format(x)
    
    # Apply the formatting function
    df['Original'] = df['Original'].apply(format_with_thousands_separator)
    df['Alvo'] = df['Alvo'].apply(format_with_thousands_separator)
    df['Variação (%)'] = df['Variação (%)'].apply(format_with_thousands_separator)
    
    return df

def converter_coordenadas(x: float, y: float, unit: float = 1) -> Tuple[float, float]:
    """Converte coordenadas para unidades específicas.
    
    Args:
        x: Coordenada x
        y: Coordenada y
        unit: Unidade de medida (default=1)
        
    Returns:
        Tupla (x, y) convertida
    """
    x, y = x * unit, y * unit
    return x, y

def converter_coordenadas_retangulo(x: float, y: float, largura: float, altura: float, unit: float = 1) -> Tuple[float, float, float, float]:
    """Converte coordenadas retangulares para unidades específicas.
    
    Args:
        x, y: Coordenadas do canto superior esquerdo
        largura, altura: Largura e altura do retângulo
        unit: Unidade de medida (default=1)
        
    Returns:
        Tupla (x, y, largura, altura) convertida
    """
    x, y, largura, altura = x * unit, y * unit, largura * unit, altura * unit
    return x, y, largura, altura

def converter_mm_para_pontos(milimetros: float) -> float:
    """Converte milímetros para pontos (unidade ReportLab)."""
    return milimetros * 2.83465  # 1 milímetro é aproximadamente igual a 2.83465 pontos

# ============================
# Funções para tabelas
# ============================

def criar_tabela_historico(df: pd.DataFrame, dict_nomes: Dict[str, str], num_linhas: int) -> Table:
    """Cria uma tabela de histórico a partir de um DataFrame.
    
    Args:
        df: DataFrame com os dados históricos
        dict_nomes: Dicionário com nomes amigáveis para as colunas
        num_linhas: Número de linhas a mostrar (a partir do final)
        
    Returns:
        Objeto Table do ReportLab
    """
    df = df.tail(num_linhas)
    dados_tabela = []

    # Adiciona a primeira linha com os nomes amigáveis usando o dicionário fornecido
    primeira_linha = [dict_nomes[col] for col in df.columns]
    dados_tabela.append(primeira_linha)

    variacoes_indices = []

    # Adicionar dados com checagem de variação
    for i in range(len(df)):
        row = []
        for col in df.columns:
            valor = df.iloc[i][col]
            valor_str = valor
            if col != 'COMPETEN':
                valor_str = f"{valor:,.0f}".replace(",", ".")
                if i > 0:
                    valor_anterior = df.iloc[i-1][col]
                    variacao = abs((valor - valor_anterior) / valor_anterior)
                    if variacao > 0.2:  # Mais de 20% de variação
                        variacoes_indices.append((i+1, df.columns.get_loc(col)))  # Salvando índice e coluna
            row.append(valor_str)
        dados_tabela.append(row)

   
    # Configurar o estilo da tabela
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),  # Defina a fonte
        ('FONTSIZE', (0, 0), (-1, -1), 7),             # Defina o tamanho da fonte
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),         # Alinhamento central
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),  # Adiciona a grade horizontal abaixo de cada linha
        ('LINEBELOW', (0, 1), (-1, -1), 1, colors.lightgrey),  # Adiciona a grade horizontal abaixo de cada linha
        ('LINEHEIGHT', (0, 0), (-1, -1), 0),  # Define a altura da linha para o mínimo possível,
        ('LEADING', (0, 0), (-1, -1), 9),  # Define a altura da linha para um valor que funcione bem com a fonte de tamanho 6
    ])

    # Adicionar estilo de célula vermelha para variações
    for index, col in variacoes_indices:
        style.add('TEXTCOLOR', (col, index), (col, index), colors.red)

    # Criar a tabela com alturas de linha definidas
    tabela = Table(dados_tabela, style=style)

    return tabela

def criar_tabela_benchmark(df: pd.DataFrame, dict_nomes: Dict[str, str]) -> Table:
    """Cria uma tabela de benchmarking a partir de um DataFrame.
    
    Args:
        df: DataFrame com os dados de benchmark
        dict_nomes: Dicionário com nomes amigáveis para as colunas
        
    Returns:
        Objeto Table do ReportLab
    """
    df['tipo'] = 'Unidade de Referência'
    df['tipo'].iloc[0] = ''

    def processar_lista(lista):
        pl = lista[:60]
        pl = re.sub(r"\(([^)]+), ([^)]+)\)", r"(\1: \2)", pl)
        pl = re.sub(r",", "|", pl)
        pl = pl.replace('[','').replace(']','')
        pl = pl.replace("'", "").replace("(", "").replace(")", "").replace(" ", "")
        return pl

    df['Producao'] = df['Producao'].apply(processar_lista)
    # Cria uma lista para os dados da tabela
    dados_tabela = []
    c =  ['tipo', 'CNES', 'CNES_SALAS', 'CNES_LEITOS_SUS', 'HORAS_MEDICOS', 'HORAS_ENFERMAGEM', 'SIA_SIH_VALOR', 'Lambda', 'Producao']
    df = df[c]

    # Adiciona a primeira linha com os nomes amigáveis usando o dicionário fornecido
    primeira_linha = [dict_nomes[col] for col in df.columns]
    dados_tabela.append(primeira_linha)

    # Adicionar dados com checagem de variação
    for i in range(len(df)):
        row = []
        for col in df.columns:
            valor = df.iloc[i][col]
            valor_str = valor
            if col in ['CNES_SALAS', 'CNES_LEITOS_SUS', 'HORAS_MEDICOS', 'HORAS_ENFERMAGEM', 'SIA_SIH_VALOR']:
                valor_str = f"{valor:,.0f}".replace(",", ".")
            if col in ['Lambda']:
                if valor_str == -1:
                    valor_str = ''
                else:
                    valor_str = f"{valor:,.2f}".replace(".", ",")
            row.append(valor_str)
        dados_tabela.append(row)
    
    # Configurar o estilo da tabela
    style = TableStyle([
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),  # Defina a fonte
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),             # Defina o tamanho da fonte
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),         # Alinhamento central
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),  # Adiciona a grade horizontal abaixo de cada linha
        ('LINEBELOW', (0, 1), (-1, -1), 1, colors.lightgrey),  # Adiciona a grade horizontal abaixo de cada linha
        ('LINEHEIGHT', (0, 0), (-1, -1), 0),  # Define a altura da linha para o mínimo possível,
        ('LEADING', (0, 0), (-1, -1), 9),  # Define a altura da linha para um valor que funcione bem com a fonte de tamanho 6
    ])

    # Criar a tabela
    tabela = Table(dados_tabela, style=style)

    return tabela

def criar_tabela_alvos(df: pd.DataFrame) -> Table:
    """Cria uma tabela de alvos a partir de um DataFrame.
    
    Args:
        df: DataFrame com os dados de alvos
        
    Returns:
        Objeto Table do ReportLab
    """
    df = df.drop(columns=['cluster'])
    # Cria uma lista para os dados da tabela
    dados_tabela = []

    df['Alvo'] = df['Original'] - df['Alvo']
    df['Variação (%)'] = 100 * df['Alvo'] / df['Original']
    
    # Adiciona a primeira linha com os nomes amigáveis usando o dicionário fornecido
    primeira_linha = ['Entradas', 'Original', 'Ociosidade', 'Percentual (%)']
    dados_tabela.append(primeira_linha)

    # Adicionar dados com checagem de variação
    for i in range(len(df)-1):
        row = []
        for col in df.columns:
            valor = df.iloc[i][col]
            valor_str = valor
            if col in ['Original', 'Alvo', 'Variação (%)']:
                valor_str = f"{valor:,.0f}".replace(",", ".")
            row.append(valor_str)
        dados_tabela.append(row)

    # Adiciona a primeira linha com os nomes amigáveis usando o dicionário fornecido
    primeira_linha = ['Saída', 'Original', 'Alvo', 'Percentual (%)']
    dados_tabela.append(primeira_linha)

    df = df[df.Parâmetro=='Produção']
    df['Alvo'] =  df['Original'] - df['Alvo']
    df['Variação (%)'] = 100 * (df['Alvo'] / df['Original'] - 1)
    # Adicionar dados com checagem de variação
    for i in range(len(df)):
        row = []
        for col in df.columns:
            valor = df.iloc[i][col]
            valor_str = valor
            if col in ['Original', 'Alvo', 'Variação (%)']:
                valor_str = f"{valor:,.0f}".replace(",", ".")
            row.append(valor_str)
        dados_tabela.append(row)
    
    # Configurar o estilo da tabela
    style = TableStyle([
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),  # Defina a fonte
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('BACKGROUND', (0, 5), (-1, 5), colors.lightblue),
        ('FONTNAME', (0, 5), (-1, 5), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),             # Defina o tamanho da fonte
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),         # Alinhamento central
        ('LINEBELOW', (0, 1), (-1, -1), 1, colors.lightgrey),  # Adiciona a grade horizontal abaixo de cada linha
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),  # Adiciona a grade horizontal abaixo de cada linha
        ('LINEBELOW', (0, 5), (-1, 5), 1, colors.black),  # Adiciona a grade horizontal abaixo de cada linha
        ('LINEHEIGHT', (0, 0), (-1, -1), 0),  # Define a altura da linha para o mínimo possível,
        ('LEADING', (0, 0), (-1, -1), 9),  # Define a altura da linha para um valor que funcione bem com a fonte de tamanho 6
    ])

    # Criar a tabela
    tabela = Table(dados_tabela, style=style)

    return tabela

# ============================
# Funções para gráficos
# ============================

def criar_grafico_linha(df: pd.DataFrame, x_col: str, y_col: str, titulo: str, ax: plt.Axes, 
                        multiplicador: int = 0) -> None:
    """Cria um gráfico de linha com formatação padrão.
    
    Args:
        df: DataFrame com os dados
        x_col: Nome da coluna para o eixo X
        y_col: Nome da coluna para o eixo Y
        titulo: Título do gráfico
        ax: Objeto de eixos matplotlib
        multiplicador: Se 1, multiplica valores por 100 (para percentuais)
    """
    custom_font = FontProperties()  # Ajuste conforme necessário
    
    sns.lineplot(data=df, x=x_col, y=y_col, marker='o', ax=ax, linewidth=3)
    for x_val, y_val in zip(df[x_col], df[y_col]):
        if multiplicador == 1:
            y_formatted = locale.format_string("%.2f", 100*y_val, grouping=True)
        else:    
            y_formatted = locale.format_string("%.0f", y_val, grouping=True)
        ax.annotate(y_formatted, (x_val, y_val), textcoords="offset points", xytext=(0, 9), 
                   ha='center', va='bottom', fontproperties=custom_font, fontsize=14, rotation=45)

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel('', fontproperties=custom_font)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(custom_font)
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(custom_font)
    ax.set_xlabel('')
    if multiplicador == 1:
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(0, df[y_col].max() * 1.1)
    ax.yaxis.set_ticks([])
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(titulo, fontproperties=custom_font, fontsize=18)
    sns.despine(left=True, right=True, top=True, bottom=False)

def criar_grafico_eficiencia(df: pd.DataFrame, x_col: str, y_col: str, y2_col: str, titulo: str, 
                            ax: plt.Axes, multiplicador: int = 0) -> None:
    """Cria um gráfico de eficiência com barras e linha.
    
    Args:
        df: DataFrame com os dados
        x_col: Nome da coluna para o eixo X
        y_col: Nome da coluna para o eixo Y (linha)
        y2_col: Nome da coluna para o eixo Y secundário (barras)
        titulo: Título do gráfico
        ax: Objeto de eixos matplotlib
        multiplicador: Se 1, multiplica valores por 100 (para percentuais)
    """
    custom_font = FontProperties()  # Ajuste conforme necessário
    
    ax2 = ax.twinx()
    bars = ax2.bar(df[x_col], df[y2_col], color='gray', alpha=0.6, label='Barra')
    bars[-1].set_color('blue')  # Destacar a última barra em azul

    sns.lineplot(data=df, x=x_col, y=y_col, marker='o', ax=ax, linewidth=5)
    
    for i, (x_val, y_val) in enumerate(zip(df[x_col], df[y_col])):
            if multiplicador == 1:
                y_formatted = locale.format_string("%.2f", 100 * y_val, grouping=True)
            else:
                y_formatted = locale.format_string("%.0f", y_val, grouping=True)
            if i == len(df) - 1:  # Último mês
                ax.annotate(y_formatted, (x_val, y_val), textcoords="offset points", xytext=(0, 9), ha='center', va='bottom',
                            fontproperties=custom_font, fontsize=18, weight='bold', bbox=dict(facecolor='blue', alpha=0.25))
            else:
                ax.annotate(y_formatted, (x_val, y_val), textcoords="offset points", xytext=(0, 9), ha='center', va='bottom',
                            fontproperties=custom_font, fontsize=14)
    
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel('', fontproperties=custom_font)
    
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(custom_font)
        tick.set_fontsize(14)
    
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(custom_font)
    
    ax.set_xlabel('')
    ax.set_ylim(0, df[y_col].max() * 1.1)
    
    ax.yaxis.set_ticks([])
    ax.tick_params(axis='x')
    ax.set_title(titulo, fontproperties=custom_font, fontsize=18)
    sns.despine(left=True, right=True, top=True, bottom=False)

    # Ajustes para o eixo secundário (barra)
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax2.set_ylabel('', fontproperties=custom_font)
    ax2.set_ylim(0, df[y2_col].max() * 1.1)
    ax2.yaxis.set_ticks([])

def filtrar_procedimentos_principais(df: pd.DataFrame, colunas_procedimentos: List[str], n_procedimentos: int = 10) -> pd.DataFrame:
    """Seleciona os N maiores procedimentos do DataFrame.
    
    Args:
        df: DataFrame com os dados
        colunas_procedimentos: Lista de colunas de procedimentos
        n_procedimentos: Número de procedimentos a selecionar
        
    Returns:
        DataFrame com os maiores procedimentos
    """
    # Verifica se o DataFrame tem pelo menos um registro
    if df.empty:
        raise ValueError("O DataFrame está vazio")
    
    df2 = df[colunas_procedimentos]
    # Seleciona o primeiro registro
    procedimentos = df2.iloc[0]

    # Identifica os 'n' maiores procedimentos
    maiores_procedimentos = procedimentos.nlargest(n_procedimentos)
    
    # Obtém os nomes das colunas dos maiores procedimentos
    colunas_maiores_procedimentos = maiores_procedimentos.index.tolist()
    
    # Adiciona a coluna de ID na lista de colunas a serem selecionadas
    colunas_maiores_procedimentos.insert(0, 'DESCESTAB')
    colunas_maiores_procedimentos.insert(0, 'CNES')
    
    # Cria um novo DataFrame apenas com as colunas selecionadas
    df_novo = df[colunas_maiores_procedimentos]

    return df_novo

def criar_grafico_procedimentos(df: pd.DataFrame, nome_procedimentos: Dict[str, str], cluster_nome: str) -> None:
    """Cria gráfico de barras dos procedimentos.
    
    Args:
        df: DataFrame com os dados
        nome_procedimentos: Dicionário com nomes amigáveis dos procedimentos
        cluster_nome: Nome do cluster para identificação
    """
    custom_font = FontProperties()  # Ajuste conforme necessário
    
    # Define o tamanho da figura
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Número de IDs e procedimentos
    num_ids = df.shape[0]
    num_procedimentos = df.shape[1] - 2
    
    # Definindo a posição das barras
    indices = np.arange(num_procedimentos)
    largura = 0.15  # Largura das barras
    
    # Cores para as barras
    cores = plt.cm.Blues(np.linspace(0.3, 0.8, num_procedimentos))
    
    # Percorre cada ID para plotar os valores dos procedimentos
    for i in range(num_ids):
        valores = df.iloc[i, 2:]  # Exclui as duas primeiras colunas (ID e DESCESTB)
        
        # Se for o primeiro hospital, destacar com uma cor azul forte
        if i == 0:
            ax.bar(indices + i * largura, valores, largura, label=f'{df.iloc[i, 1]}', color='darkblue')
        else:
            ax.bar(indices + i * largura, valores, largura, label=f'{df.iloc[i, 1]}', color=cores[i])
        
    # Configurações do gráfico
    ax.set_xlabel('', fontproperties=custom_font)
    ax.set_ylabel('', fontproperties=custom_font)
    ax.set_title('', fontproperties=custom_font)
    
    # Definindo os rótulos do eixo x sem quebrar palavras
    max_width = 13  # Defina o tamanho máximo de caracteres por linha para os rótulos do eixo x
    wrapped_labels = [
        '\n'.join(textwrap.wrap(nome_procedimentos.get(coluna, coluna), max_width, break_long_words=False, replace_whitespace=False))
        for coluna in df.columns[2:]
    ]
    
    ax.set_xticks(indices + largura * (num_ids - 1) / 2)
    ax.set_xticklabels(wrapped_labels, ha='center', fontsize=8, fontproperties=custom_font)
    ax.tick_params(axis='y', labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(formatar_milhares))
        
    # Formatação do eixo y
    ax.yaxis.set_major_formatter(FuncFormatter(formatar_milhares))
    ax.tick_params(axis='y', labelsize=9)
    ax.yaxis.set_tick_params(width=0)

    # Alterar o tamanho e a fonte das etiquetas do eixo y
    for label in ax.get_yticklabels():
        label.set_fontproperties(custom_font)
        label.set_fontsize(9)

    # Remover o retângulo externo
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.spines['bottom'].set_visible(True)
    
    # Remover grids verticais e definir grids horizontais finos e cinzas
    ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', color='white', linestyle='-', linewidth=0)

    legend = ax.legend(title="", fontsize=8, prop=custom_font, loc='upper right', frameon=True)
    legend.get_frame().set_facecolor('white')

# ============================
# Funções auxiliares
# ============================

def gerar_texto_ranking(df: pd.DataFrame, cluster: str) -> str:
    """Gera mensagem informativa sobre o ranking do hospital no cluster.
    
    Args:
        df: DataFrame com os dados de eficiência
        cluster: Código do cluster para identificação
        
    Returns:
        String com a mensagem formatada
    """
    # Extrair o CNES dos primeiros sete dígitos do cluster
    cnes = cluster[:7]
    
    # Filtrar o DataFrame pelo valor do cluster
    df_filtrado = df[df['Cluster'] == cluster]
    
    # Verificar o ranking com base na coluna 'Eficiencia'
    df_filtrado['ranking'] = df_filtrado['Eficiencia'].rank(method='min', ascending=False)
    
    # Contar o total de linhas no cluster filtrado
    total_linhas = df_filtrado.shape[0]
    
    # Encontrar a linha específica onde 'CNES' é igual ao valor extraído
    linha_cnes = df_filtrado[df_filtrado['CNES'] == cnes]
    
    if not linha_cnes.empty:
        ranking = (linha_cnes['ranking'].values[0])
        eficiencia = linha_cnes['Eficiencia'].values[0]
        hospital = linha_cnes['DESCESTAB'].values[0]
        mensagem = (f"O estabelecimento de saúde <b>{hospital}</b> obteve o ranking {ranking:.0f} de um total de {total_linhas} hospitais no grupo selecionado para comparação. A sua eficiência no mês foi de <b>{100*eficiencia:.2f}</b>.")
    else:
        mensagem = f"O hospital com CNES {cnes} não foi encontrado no grupo {cluster}."
    
    return mensagem

def gerar_paragrafos_alertas(row: pd.Series, alertas_dict: Dict[str, str]) -> List[str]:
    """Gera parágrafos de alertas a partir dos dados.
    
    Args:
        row: Linha do DataFrame com alertas
        alertas_dict: Dicionário de descrições de alertas
        
    Returns:
        Lista de strings com descrições de alertas ativos
    """
    paragrafos = []
    for alerta, descricao in alertas_dict.items():
        if row.get(alerta) == 1:
            paragrafos.append(f"• {descricao}")
    return paragrafos

# Constantes - dicionários
DATA_DICT = {
    # Tabela SIGTAP (apenas exemplo parcial)
    'SIA-0101': 'Ações coletivas/individuais em saúde (SIA-0101)',
    'SIA-0102': 'Vigilância em saúde (SIA-0102)',
    # ... outros códigos aqui
}

DICT_NOMES = {
    'COMPETEN' : 'Mês',
    'CNES_SALAS': 'Salas',
    'CNES_CENTROS_CIRURGICOS': 'Centros\nCirúrgicos',
    'CNES_LEITOS_SUS': 'Leitos\nSUS',
    'CNES_MEDICOS': 'Médicos',
    'HORAS_MEDICOS': 'Horas\nMédicos',
    'CNES_PROFISSIONAIS_ENFERMAGEM': 'Profissionais\nEnfermagem',
    'HORAS_ENFERMAGEM': 'Horas\nEnfermagem',
    'SIH_VALOR': 'SIH\n(R/$)',
    'SIA_VALOR': 'SIA\n(R/$)',
    'SIA_SIH_VALOR': 'Produção\nTotal (R/$)',
    'tipo': ' '
}

DICT_NOMES_BM = {
    'CNES':'CNES',
    'CNES_SALAS': 'Salas',
    'CNES_LEITOS_SUS': 'Leitos\nSUS',
    'HORAS_MEDICOS': 'Horas\nMédicos',
    'HORAS_ENFERMAGEM': 'Horas\nEnfermagem',
    'SIA_SIH_VALOR': 'Produção\nTotal',
    'Lambda': 'Peso',
    'tipo': ' ',
    'Producao': 'Perfil'
} 