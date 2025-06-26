#!/usr/bin/env python3
"""
Gerador de Relatório Detalhado por CNES

Este script gera um relatório completo para um CNES específico incluindo:
- Dados de média móvel (últimos 12 meses)
- Evolução temporal dos inputs/outputs DEA
- Comparação com benchmarks
- Gráfico spider dos alvos
- Análise dos principais procedimentos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import logging
import sys
import json
import pathlib
import yaml
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import warnings
import tempfile
import base64

# Configurações de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Dicionário padrão para nomes de campos
NOMES_CAMPOS_PADRAO = {
    'COMPETEN': 'Competência',
    'CNES_SALAS': 'Salas Cirúrgicas',
    'CNES_LEITOS_SUS': 'Leitos SUS',
    'HORAS_MEDICOS': 'Horas Médicos',
    'HORAS_ENFERMAGEM': 'Horas Enfermagem',
    'SIA_VALOR': 'Valor SIA',
    'SIH_VALOR': 'Valor SIH',
    'SIA_SIH_VALOR': 'Produção Total',
    'DESCESTAB': 'Nome do Hospital',
    'MUNICIPIO': 'Município',
    'UF': 'UF',
    'REGIAO': 'Região'
}

def obter_nome_padrao(campo: str) -> str:
    """Retorna o nome padronizado de um campo."""
    return NOMES_CAMPOS_PADRAO.get(campo, campo.replace('_', ' ').replace('CNES ', '').title())

# Adiciona o diretório raiz do projeto ao sys.path para garantir que os módulos sejam encontrados
project_root_path = pathlib.Path(__file__).resolve().parent.parent
if str(project_root_path) not in sys.path:
    sys.path.append(str(project_root_path))

# Importações do projeto
try:
    from utils.constantes import DEA_INPUT_COLS, DEA_OUTPUT_COL, cols_sia, cols_sih, alertas_dict, data_dict
except ImportError as e:
    print(f"ERRO: Erro ao importar módulos do projeto: {e}")
    print(f"Certifique-se de que está executando o script a partir do diretório raiz do projeto.")
    print(f"Caminho do projeto: {project_root_path}")
    sys.exit(1)


class ErroRelatorio(Exception):
    """Exceção base para erros no relatório."""
    pass


class CarregadorConfiguracao:
    """Carrega configurações do projeto."""
    def __init__(self, caminho_config: str = 'config.yaml'):
        self.caminho_config = pathlib.Path(caminho_config).resolve()
        self.config = self._carregar()
        self.project_root = self.caminho_config.parent

    def _carregar(self) -> Dict[str, Any]:
        try:
            with open(self.caminho_config, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ErroRelatorio(f"Erro ao carregar configuração: {e}")

    def obter(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


class CarregadorDados:
    """Carrega os dados dos arquivos gerados pelo pipeline."""
    
    def __init__(self, config: CarregadorConfiguracao):
        self.config = config
        self.project_root = config.project_root
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def carregar_media_movel(self) -> pd.DataFrame:
        """Carrega dados de média móvel."""
        arquivo_mm = self.project_root / self.config.obter('moving_average_dir', 'resultados/media_movel') / 'media_movel_consolidada.parquet'
        if not arquivo_mm.exists():
            raise ErroRelatorio(f"Arquivo de média móvel não encontrado: {arquivo_mm}")
        
        df = pd.read_parquet(arquivo_mm)
        df['COMPETEN'] = df['COMPETEN'].astype(str)
        self.logger.info(f"Dados de média móvel carregados: {len(df)} registros")
        return df
    

    
    def carregar_dea(self) -> pd.DataFrame:
        """Carrega dados de eficiência DEA."""
        arquivo_dea = self.project_root / self.config.obter('dea_dir', 'resultados/dea') / 'resultados_dea_final.parquet'
        if not arquivo_dea.exists():
            raise ErroRelatorio(f"Arquivo de DEA não encontrado: {arquivo_dea}")
        
        df = pd.read_parquet(arquivo_dea)
        
        # Extrai COMPETENCIA e CNES do campo id_grupo_original (formato: AAAAMM-CNES)
        df[['COMPETEN', 'CNES']] = df['id_grupo_original'].str.split('-', expand=True)
        df['COMPETEN'] = df['COMPETEN'].astype(str)
        
        # Renomeia colunas para compatibilidade com o código existente (mantém CNES_ALVO)
        colunas_rename = {
            'eficiencia_envoltoria': 'Eficiencia',
            'theta_envoltoria': 'phi',
            'benchmarks_envoltoria': 'benchmarks',
            'metas_envoltoria': 'targets'
        }
        df = df.rename(columns=colunas_rename)
        
        self.logger.info(f"Dados de DEA carregados: {len(df)} registros")
        return df
    
    def carregar_alertas(self) -> pd.DataFrame:
        """Carrega dados de alertas."""
        # Tenta diferentes locais possíveis para o arquivo de alertas
        possiveis_caminhos = [
            self.project_root / 'resultados' / 'alertas' / 'alertas_hospitais_consolidado.parquet',
            self.project_root / 'resultados' / 'alertas' / 'alertas_hospitais_consolidado_latest.parquet',
            self.project_root / self.config.obter('alertas_dir', 'resultados/alertas') / 'alertas_hospitais_consolidado.parquet',
            self.project_root / 'resultados' / 'alertas_hospitais_consolidado.parquet'
        ]
        
        for arquivo_alertas in possiveis_caminhos:
            if arquivo_alertas.exists():
                df = pd.read_parquet(arquivo_alertas)
                df['COMPETEN'] = df['COMPETEN'].astype(str)
                # Garantir que CNES seja string com zero à esquerda se necessário
                df['CNES'] = df['CNES'].astype(str).str.zfill(7)
                self.logger.info(f"Dados de alertas carregados: {len(df)} registros de {arquivo_alertas}")
                return df
        
        self.logger.warning("Arquivo de alertas não encontrado em nenhum local. Relatório será gerado sem dados de alertas.")
        return pd.DataFrame()


class ProcessadorDados:
    """Processa e prepara os dados para o relatório."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def filtrar_dados_cnes(self, df: pd.DataFrame, cnes: str, competencia: str = None) -> pd.DataFrame:
        """Filtra dados para um CNES específico."""
        df_filtrado = df[df['CNES'] == cnes].copy()
        if competencia:
            df_filtrado = df_filtrado[df_filtrado['COMPETEN'] == competencia]
        return df_filtrado.sort_values('COMPETEN')
    
    def obter_ultimos_meses(self, df: pd.DataFrame, cnes: str, competencia_referencia: str, num_meses: int = 12) -> pd.DataFrame:
        """Obtém dados dos últimos N meses para um CNES até uma competência de referência."""
        df_cnes = self.filtrar_dados_cnes(df, cnes)
        if df_cnes.empty:
            return df_cnes
        
        # Filtra dados até a competência de referência
        df_ate_referencia = df_cnes[df_cnes['COMPETEN'] <= competencia_referencia]
        
        if df_ate_referencia.empty:
            return df_ate_referencia
        
        # Pega os últimos N meses até a competência de referência
        competencias_ordenadas = sorted(df_ate_referencia['COMPETEN'].unique(), reverse=True)
        ultimas_competencias = competencias_ordenadas[:num_meses]
        return df_ate_referencia[df_ate_referencia['COMPETEN'].isin(ultimas_competencias)].sort_values('COMPETEN')
    
    def extrair_benchmarks(self, df_dea: pd.DataFrame, cnes: str, competencia: str) -> List[str]:
        """Extrai lista de CNEs benchmark do arquivo DEA para um CNES em uma competência."""
        # Usa os nomes corretos das colunas no arquivo DEA
        df_cnes = df_dea[(df_dea['CNES_ALVO'] == cnes) & (df_dea['id_grupo_original'] == f"{competencia}-{cnes}")]
        if df_cnes.empty:
            return []
        
        try:
            dados = df_cnes.iloc[0]
            # Usa o campo renomeado 'benchmarks' (que era 'benchmarks_envoltoria')
            if 'benchmarks' in dados and pd.notna(dados['benchmarks']):
                benchmarks_data = json.loads(dados['benchmarks'])
                return list(benchmarks_data.keys())
            return []
        except (json.JSONDecodeError, AttributeError):
            return []
    
    def preparar_dados_procedimentos(self, df: pd.DataFrame, cnes: str, competencia: str) -> Tuple[pd.Series, List[str]]:
        """Prepara dados dos principais procedimentos para um CNES."""
        df_cnes = df[(df['CNES'] == cnes) & (df['COMPETEN'] == competencia)]
        if df_cnes.empty:
            return pd.Series(), []
        
        # Colunas de procedimentos SIA e SIH
        cols_procedimentos = [col for col in df.columns if col.startswith(('SIA-', 'SIH-'))]
        
        if not cols_procedimentos:
            return pd.Series(), []
        
        # Valores dos procedimentos para este CNES
        valores_procedimentos = df_cnes[cols_procedimentos].iloc[0]
        valores_procedimentos = valores_procedimentos[valores_procedimentos > 0]
        
        # Top 5 procedimentos
        top_procedimentos = valores_procedimentos.nlargest(5)
        return top_procedimentos, top_procedimentos.index.tolist()
    
    def obter_informacoes_hospital(self, df: pd.DataFrame, cnes: str, competencia: str) -> Dict[str, str]:
        """Obtém informações básicas do hospital (nome, município, UF)."""
        df_cnes = df[(df['CNES'] == cnes) & (df['COMPETEN'] == competencia)]
        if df_cnes.empty:
            # Se não encontrar na competência específica, pega qualquer registro do CNES
            df_cnes = df[df['CNES'] == cnes]
            if df_cnes.empty:
                return {}
        
        dados = df_cnes.iloc[0]
        info = {}
        
        # Campos de interesse para o cabeçalho
        campos_info = ['DESCESTAB', 'MUNICIPIO', 'UF', 'REGIAO']
        for campo in campos_info:
            if campo in dados:
                info[campo] = str(dados[campo]) if pd.notna(dados[campo]) else 'Não informado'
            else:
                info[campo] = 'Não disponível'
        
        return info
    
    def obter_eficiencia_com_variacao(self, df_dea: pd.DataFrame, cnes: str, competencia: str) -> Dict[str, Any]:
        """Obtém eficiência atual e variação em relação ao mês anterior."""
        # Busca eficiência da competência atual
        df_atual = df_dea[(df_dea['CNES'] == cnes) & (df_dea['COMPETEN'] == competencia)]
        
        if df_atual.empty:
            return {'eficiencia_atual': None, 'variacao_pct': None, 'tem_melhora': None}
        
        eficiencia_atual = df_atual.iloc[0]['Eficiencia']
        
        # Calcula competência anterior (mês anterior)
        try:
            ano = int(competencia[:4])
            mes = int(competencia[4:])
            if mes == 1:
                comp_anterior = f"{ano-1}12"
            else:
                comp_anterior = f"{ano}{mes-1:02d}"
        except:
            return {'eficiencia_atual': eficiencia_atual, 'variacao_pct': None, 'tem_melhora': None}
        
        # Busca eficiência do mês anterior
        df_anterior = df_dea[(df_dea['CNES'] == cnes) & (df_dea['COMPETEN'] == comp_anterior)]
        
        if df_anterior.empty:
            return {'eficiencia_atual': eficiencia_atual, 'variacao_pct': None, 'tem_melhora': None}
        
        eficiencia_anterior = df_anterior.iloc[0]['Eficiencia']
        
        # Calcula variação percentual
        if eficiencia_anterior > 0:
            variacao_pct = ((eficiencia_atual - eficiencia_anterior) / eficiencia_anterior) * 100
            tem_melhora = variacao_pct > 0
        else:
            variacao_pct = None
            tem_melhora = None
        
        return {
            'eficiencia_atual': eficiencia_atual,
            'variacao_pct': variacao_pct,
            'tem_melhora': tem_melhora,
            'competencia_anterior': comp_anterior
        }
    
    def obter_alvos_com_diferencas(self, df_dea: pd.DataFrame, cnes: str, competencia: str) -> Dict[str, Dict[str, float]]:
        """Obtém alvos DEA e calcula diferenças percentuais em relação aos valores atuais."""
        df_cnes = df_dea[(df_dea['CNES_ALVO'] == cnes) & (df_dea['id_grupo_original'] == f"{competencia}-{cnes}")]
        
        if df_cnes.empty:
            return {}
        
        try:
            dados = df_cnes.iloc[0]
            
            # Carrega dados originais e metas
            valores_originais = {}
            metas = {}
            
            if 'valores_originais' in dados and pd.notna(dados['valores_originais']):
                valores_originais = json.loads(dados['valores_originais'])
            
            if 'targets' in dados and pd.notna(dados['targets']):
                metas = json.loads(dados['targets'])
            
            if not valores_originais or not metas:
                return {}
            
            # Calcula diferenças para cada input/output
            colunas_dea = DEA_INPUT_COLS + [DEA_OUTPUT_COL]
            alvos_info = {}
            
            for col in colunas_dea:
                if col in valores_originais:
                    valor_atual = float(valores_originais[col])
                    
                    # Busca meta
                    meta_key = f"target_{col}" if f"target_{col}" in metas else col
                    if meta_key in metas:
                        valor_meta = float(metas[meta_key])
                        
                        # Calcula diferença percentual
                        if valor_atual > 0:
                            if col in DEA_INPUT_COLS:
                                # Para inputs: negativo = precisa reduzir, positivo = pode aumentar
                                diff_pct = ((valor_meta - valor_atual) / valor_atual) * 100
                            else:
                                # Para output: positivo = precisa aumentar, negativo = pode diminuir  
                                diff_pct = ((valor_meta - valor_atual) / valor_atual) * 100
                            
                            alvos_info[col] = {
                                'valor_atual': valor_atual,
                                'valor_meta': valor_meta,
                                'diferenca_pct': diff_pct,
                                'nome_campo': obter_nome_padrao(col)
                            }
            
            return alvos_info
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Erro ao processar alvos DEA: {e}")
            return {}


class GeradorVisualiza:
    """Gera visualizações para o relatório."""
    
    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configurações de estilo
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def criar_tabela_media_movel(self, df: pd.DataFrame, cnes: str) -> str:
        """Cria tabela HTML com dados de média móvel dos últimos 12 meses com coloração baseada em variação percentual."""
        if df.empty:
            return "<p>Nenhum dado de média móvel encontrado.</p>"
        
        # Seleciona colunas relevantes
        colunas_interesse = ['COMPETEN'] + DEA_INPUT_COLS + ['SIA_VALOR', 'SIH_VALOR', 'SIA_SIH_VALOR']
        colunas_existentes = [col for col in colunas_interesse if col in df.columns]
        
        df_tabela = df[colunas_existentes].copy()
        df_original = df_tabela.copy()  # Mantém valores originais para cálculo de variação
        
        # Calcula variações percentuais entre meses consecutivos para cada campo
        variacoes = {}
        colunas_numericas = [col for col in df_tabela.columns if col != 'COMPETEN']
        
        for col in colunas_numericas:
            if col in df_original.columns and df_original[col].dtype in ['float64', 'int64']:
                # Calcula variação percentual em relação ao mês anterior (ordenado por competência)
                df_sorted = df_original.sort_values('COMPETEN')
                valores = df_sorted[col].values
                variacoes_col = []
                
                for i in range(len(valores)):
                    if i == 0:
                        variacoes_col.append(0)  # Primeiro mês não tem variação
                    else:
                        if valores[i-1] != 0 and pd.notna(valores[i-1]) and pd.notna(valores[i]):
                            variacao = ((valores[i] - valores[i-1]) / valores[i-1]) * 100
                            variacoes_col.append(variacao)
                        else:
                            variacoes_col.append(0)
                
                # Mapeia de volta para o DataFrame original (mesmo índice)
                variacoes[col] = pd.Series(variacoes_col, index=df_sorted.index)
                variacoes[col] = variacoes[col].reindex(df_tabela.index).fillna(0)
        
        # Cria DataFrame para HTML com formatação e classes CSS
        df_html = df_tabela.copy()
        
        # Renomeia colunas para nomes padronizados
        mapeamento_colunas = NOMES_CAMPOS_PADRAO.copy()
        
        # Cria HTML manualmente para ter controle total sobre as classes CSS
        html_rows = []
        
        # Cabeçalho
        headers = []
        for col in df_tabela.columns:
            col_display = mapeamento_colunas.get(col, col)
            headers.append(f'<th>{col_display}</th>')
        
        html_rows.append(f'<tr>{"".join(headers)}</tr>')
        
        # Linhas de dados
        for idx, row in df_tabela.iterrows():
            cells = []
            
            for col_idx, col in enumerate(df_tabela.columns):
                if col == 'COMPETEN':
                    # Primeira coluna (competência) sem formatação especial
                    cells.append(f'<td>{row[col]}</td>')
                else:
                    # Colunas numéricas com possível coloração
                    valor_original = row[col]
                    valor_formatado = f"{valor_original:,.0f}" if pd.notna(valor_original) else "0"
                    
                    # Verifica se há variação significativa para esta célula
                    classe_css = ""
                    tooltip = ""
                    
                    if col in variacoes:
                        variacao = variacoes[col].loc[idx]
                        
                        if abs(variacao) > 20:  # Variação acima de 20%
                            if variacao > 20:
                                classe_css = ' class="variacao-alta-positiva"'
                                tooltip = f' title="↗ Aumento de {variacao:.1f}% em relação ao mês anterior"'
                            elif variacao < -20:
                                classe_css = ' class="variacao-alta-negativa"'
                                tooltip = f' title="↘ Redução de {abs(variacao):.1f}% em relação ao mês anterior"'
                    
                    cells.append(f'<td{classe_css}{tooltip}>{valor_formatado}</td>')
            
            html_rows.append(f'<tr>{"".join(cells)}</tr>')
        
        # Monta HTML completo da tabela
        html_table = f'''
        <table id="media-movel-table" class="table table-striped table-hover">
            <thead>
                {html_rows[0]}
            </thead>
            <tbody>
                {"".join(html_rows[1:])}
            </tbody>
        </table>
        '''
        
        # CSS para formatação e cores
        style_css = """
        <style>
        #media-movel-table th:not(:first-child) { text-align: right !important; }
        #media-movel-table td:not(:first-child) { text-align: right !important; }
        .variacao-alta-positiva { 
            background-color: #d4edda !important; 
            color: #155724 !important; 
            font-weight: bold !important;
        }
        .variacao-alta-negativa { 
            background-color: #f8d7da !important; 
            color: #721c24 !important; 
            font-weight: bold !important;
        }
        #media-movel-table td[title] {
            cursor: help;
        }
        </style>
        """
        
        return style_css + html_table
    
    def criar_grafico_evolucao_temporal(self, df: pd.DataFrame, df_dea: pd.DataFrame, cnes: str, competencia: str, nome_arquivo: str) -> str:
        """Cria gráfico de evolução temporal dos inputs/outputs DEA com eficiência."""
        if df.empty:
            return ""
        
        # Prepara dados
        df_plot = df.copy()
        df_plot['COMPETEN_DATE'] = pd.to_datetime(df_plot['COMPETEN'], format='%Y%m')
        
        # Adiciona dados de eficiência do DEA
        df_dea_cnes = df_dea[df_dea['CNES'] == cnes].copy()
        if not df_dea_cnes.empty:
            df_plot = df_plot.merge(df_dea_cnes[['COMPETEN', 'Eficiencia']], on='COMPETEN', how='left')
        
        # Cria subplot com 3x2 (6 gráficos)
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                NOMES_CAMPOS_PADRAO['CNES_SALAS'], NOMES_CAMPOS_PADRAO['CNES_LEITOS_SUS'], 
                NOMES_CAMPOS_PADRAO['HORAS_MEDICOS'], NOMES_CAMPOS_PADRAO['HORAS_ENFERMAGEM'],
                'Produção Total (Output)', 'Eficiência DEA'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Inputs individuais
        input_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for i, col in enumerate(DEA_INPUT_COLS):
            if col in df_plot.columns and i < len(input_positions):
                row, col_pos = input_positions[i]
                fig.add_trace(
                    go.Scatter(
                        x=df_plot['COMPETEN_DATE'], 
                        y=df_plot[col], 
                        name=obter_nome_padrao(col),
                        mode='lines+markers',
                        line=dict(width=3)
                    ),
                    row=row, col=col_pos
                )
        
        # Output
        if DEA_OUTPUT_COL in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot['COMPETEN_DATE'], 
                    y=df_plot[DEA_OUTPUT_COL], 
                    name='Produção Total',
                    mode='lines+markers', 
                    line=dict(color='red', width=3)
                ),
                row=3, col=1
            )
        
        # Eficiência
        if 'Eficiencia' in df_plot.columns:
            # Remove valores nulos para plotar
            df_eff = df_plot.dropna(subset=['Eficiencia'])
            if not df_eff.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_eff['COMPETEN_DATE'], 
                        y=df_eff['Eficiencia'], 
                        name='Eficiência DEA',
                        mode='lines+markers', 
                        line=dict(color='green', width=3)
                    ),
                    row=3, col=2
                )
                
                # Adiciona linha de referência em 1.0 (eficiência máxima)
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=3, col=2)
        
        fig.update_layout(
            title_text=f"Evolução Temporal Detalhada - CNES {cnes}",
            showlegend=False,  # Remove legenda para não poluir
            height=900,
            margin=dict(t=100)
        )
        
        # Atualiza eixos Y para melhor visualização
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_yaxes(title_text="Valor", row=i, col=j)
        
        # Salva gráfico
        caminho_arquivo = self.output_dir / nome_arquivo
        fig.write_html(str(caminho_arquivo))
        self.logger.info(f"Gráfico de evolução temporal salvo: {caminho_arquivo}")
        
        return str(caminho_arquivo)
    
    def criar_tabela_benchmarks(self, df_dea: pd.DataFrame, df_mm: pd.DataFrame, 
                               cnes: str, competencia: str, benchmarks: List[str]) -> str:
        """Cria tabela comparativa com benchmarks usando dados DEA."""
        # Usa os nomes corretos das colunas no arquivo DEA
        df_dea_alvo = df_dea[(df_dea['CNES_ALVO'] == cnes) & (df_dea['id_grupo_original'] == f"{competencia}-{cnes}")]
        
        if df_dea_alvo.empty:
            return "<p>Dados de DEA não encontrados para este CNES.</p>"
        
        try:
            dados_alvo = df_dea_alvo.iloc[0]
            
            # Carrega dados originais e benchmarks do DEA
            valores_originais = {}
            benchmarks_dea = {}
            metas = {}
            
            if 'valores_originais' in dados_alvo and pd.notna(dados_alvo['valores_originais']):
                valores_originais = json.loads(dados_alvo['valores_originais'])
            
            # Usa os campos renomeados
            if 'benchmarks' in dados_alvo and pd.notna(dados_alvo['benchmarks']):
                benchmarks_dea = json.loads(dados_alvo['benchmarks'])
            
            if 'targets' in dados_alvo and pd.notna(dados_alvo['targets']):
                metas = json.loads(dados_alvo['targets'])
            
            if not valores_originais or not benchmarks_dea:
                return "<p>Dados insuficientes para criar tabela de benchmarks.</p>"
            
            colunas_comparacao = DEA_INPUT_COLS + [DEA_OUTPUT_COL]
            dados_tabela = []
            
            # 1ª Linha: CNES alvo
            linha_alvo = {'CNES': cnes, 'Tipo': 'Hospital Analisado', 'Lambda': 1.0, '_destaque': True}
            for col in colunas_comparacao:
                if col in valores_originais:
                    linha_alvo[col] = float(valores_originais[col])
                else:
                    linha_alvo[col] = 0
            dados_tabela.append(linha_alvo)
            
            # 2ª Linha: Metas DEA (logo após o hospital analisado)
            if metas:
                linha_meta = {'CNES': cnes, 'Tipo': 'Metas de Eficiência', 'Lambda': '-', '_destaque': True}
                for col in colunas_comparacao:
                    meta_key = f"target_{col}" if f"target_{col}" in metas else col
                    if meta_key in metas:
                        linha_meta[col] = float(metas[meta_key])
                    else:
                        linha_meta[col] = linha_alvo.get(col, 0)
                dados_tabela.append(linha_meta)
            
            # Linhas dos benchmarks com lambdas (a partir da 3ª linha)
            for benchmark_cnes, lambda_val in list(benchmarks_dea.items())[:10]:
                df_bench = df_mm[
                    (df_mm['CNES'] == benchmark_cnes) & 
                    (df_mm['COMPETEN'] == competencia)
                ]
                
                if not df_bench.empty:
                    linha_bench = {
                        'CNES': benchmark_cnes, 
                        'Tipo': 'Benchmark',
                        'Lambda': float(lambda_val),
                        '_destaque': False
                    }
                    for col in colunas_comparacao:
                        if col in df_bench.columns:
                            linha_bench[col] = float(df_bench.iloc[0][col])
                        else:
                            linha_bench[col] = 0
                    dados_tabela.append(linha_bench)
            
            df_tabela = pd.DataFrame(dados_tabela)
            
            # Cria HTML manualmente para controle de formatação e destaque
            mapeamento_colunas = NOMES_CAMPOS_PADRAO.copy()
            html_rows = []
            
            # Cabeçalho
            headers = []
            for col in ['CNES', 'Tipo', 'Lambda'] + colunas_comparacao:
                if col in ['CNES', 'Tipo', 'Lambda']:
                    col_display = col
                else:
                    col_display = mapeamento_colunas.get(col, col)
                headers.append(f'<th>{col_display}</th>')
            
            html_rows.append(f'<tr>{"".join(headers)}</tr>')
            
            # Linhas de dados
            for idx, row in df_tabela.iterrows():
                cells = []
                row_class = ' class="linha-destaque"' if row.get('_destaque', False) else ''
                
                # CNES
                cells.append(f'<td>{row["CNES"]}</td>')
                
                # Tipo
                cells.append(f'<td>{row["Tipo"]}</td>')
                
                # Lambda
                lambda_val = row['Lambda']
                lambda_formatado = f"{lambda_val:.4f}" if isinstance(lambda_val, (int, float)) else str(lambda_val)
                cells.append(f'<td>{lambda_formatado}</td>')
                
                # Colunas numéricas (inputs/outputs) com coloração especial para metas
                for col in colunas_comparacao:
                    valor = row[col]
                    valor_formatado = f"{valor:,.0f}" if pd.notna(valor) else "0"
                    
                    # Se for linha de metas, aplica coloração e setas
                    if row.get('Tipo') == 'Metas de Eficiência' and idx > 0:
                        # Compara com valor atual (primeira linha)
                        valor_atual = dados_tabela[0][col]  # Primeira linha é sempre o hospital analisado
                        
                        # Verifica se valores são próximos ou se valor atual é muito baixo
                        if valor_atual > 0 and abs(valor - valor_atual) > 0.01:
                            # Calcula diferença percentual
                            diff_pct = ((valor - valor_atual) / valor_atual) * 100
                            
                            # Só aplica coloração se a diferença for significativa (>1%)
                            if abs(diff_pct) > 1.0:
                                # Lógica de cores CORRIGIDA baseada no tipo de variável DEA
                                if col in DEA_INPUT_COLS:
                                    # Para INPUTS: redução = PROBLEMA (vermelho), valor igual/próximo = BOM (verde)
                                    if valor < valor_atual:  # Meta menor que atual = PROBLEMA (usando recursos demais)
                                        classe_meta = "meta-atencao"
                                        seta = "↘"
                                        tooltip = f"⚠️ Reduzir {abs(diff_pct):.1f}% - Recursos em excesso"
                                    else:  # Meta maior que atual = neutro (pode precisar de mais recursos)
                                        classe_meta = "meta-neutra"
                                        seta = "↗"
                                        tooltip = f"ℹ️ Aumentar {diff_pct:.1f}% - Mais recursos necessários"
                                else:
                                    # Para OUTPUT: aumento = PROBLEMA (vermelho), valor igual/próximo = BOM (verde)
                                    if valor > valor_atual:  # Meta maior que atual = PROBLEMA (produzindo pouco)
                                        classe_meta = "meta-atencao"
                                        seta = "↗"
                                        tooltip = f"⚠️ Aumentar {diff_pct:.1f}% - Produção insuficiente"
                                    else:  # Meta menor que atual = bom (produzindo adequadamente)
                                        classe_meta = "meta-boa"
                                        seta = "↘"
                                        tooltip = f"✅ Produção adequada ({abs(diff_pct):.1f}% acima da meta)"
                                
                                cells.append(f'<td class="{classe_meta}" title="{tooltip}">{seta} {valor_formatado}</td>')
                            else:
                                # Diferença menor que 1% = eficiente
                                cells.append(f'<td class="meta-boa" title="✅ Meta praticamente atingida (diferença < 1%)">≈ {valor_formatado}</td>')
                        else:
                            # Valores iguais, muito próximos ou valor atual = 0 = eficiente
                            cells.append(f'<td class="meta-boa" title="✅ Meta atingida - Operação eficiente">≈ {valor_formatado}</td>')
                    else:
                        # Linhas normais (hospital analisado e benchmarks)
                        cells.append(f'<td>{valor_formatado}</td>')
                
                html_rows.append(f'<tr{row_class}>{"".join(cells)}</tr>')
            
            # Monta HTML completo da tabela
            html_table = f'''
            <table id="benchmarks-table" class="table table-striped table-hover">
                <thead>
                    {html_rows[0]}
                </thead>
                <tbody>
                    {"".join(html_rows[1:])}
                </tbody>
            </table>
            '''
            
            # CSS para formatação, alinhamento e destaque
            style_css = """
            <style>
            #benchmarks-table th:nth-child(n+3) { text-align: right !important; }
            #benchmarks-table td:nth-child(n+3) { text-align: right !important; }
            .linha-destaque { 
                background-color: #e3f2fd !important; 
                font-weight: bold !important;
                border-left: 4px solid #1976d2 !important;
            }
            .linha-destaque td {
                color: #0d47a1 !important;
            }
            .meta-boa {
                background-color: #c8e6c9 !important;
                color: #2e7d32 !important;
                font-weight: bold !important;
                border-radius: 4px !important;
                padding: 4px 8px !important;
            }
            .meta-atencao {
                background-color: #ffcdd2 !important;
                color: #c62828 !important;
                font-weight: bold !important;
                border-radius: 4px !important;
                padding: 4px 8px !important;
            }
            .meta-neutra {
                background-color: #fff3e0 !important;
                color: #ef6c00 !important;
                font-weight: bold !important;
                border-radius: 4px !important;
                padding: 4px 8px !important;
            }
            #benchmarks-table td[title] {
                cursor: help;
            }
            </style>
            """
            
            return style_css + html_table
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Erro ao processar dados DEA para tabela benchmarks: {e}")
            return "<p>Erro ao processar dados de benchmarks.</p>"
    
    def criar_grafico_spider(self, df_dea: pd.DataFrame, cnes: str, competencia: str, nome_arquivo: str) -> str:
        """Cria gráfico spider comparando inputs, outputs e alvos usando EXATAMENTE a mesma lógica da tabela de benchmarks."""
        # Usa os nomes corretos das colunas no arquivo DEA
        df_cnes = df_dea[(df_dea['CNES_ALVO'] == cnes) & (df_dea['id_grupo_original'] == f"{competencia}-{cnes}")]
        if df_cnes.empty:
            self.logger.warning(f"Nenhum dado DEA encontrado para CNES {cnes}, competência {competencia}")
            return ""
        
        try:
            dados_alvo = df_cnes.iloc[0]
            
            # Usa EXATAMENTE a mesma lógica da tabela de benchmarks
            valores_originais = {}
            metas = {}
            
            if 'valores_originais' in dados_alvo and pd.notna(dados_alvo['valores_originais']):
                valores_originais = json.loads(dados_alvo['valores_originais'])
            
            # Usa os campos renomeados (igual tabela benchmarks)
            if 'targets' in dados_alvo and pd.notna(dados_alvo['targets']):
                metas = json.loads(dados_alvo['targets'])
            
            if not valores_originais or not metas:
                self.logger.warning("Dados insuficientes para criar spider chart")
                return ""
            
            # Usa as mesmas colunas da tabela benchmarks
            colunas_comparacao = DEA_INPUT_COLS + [DEA_OUTPUT_COL]
            
            categorias = []
            valores_atuais = []
            valores_alvo = []
            
            # Processa cada coluna EXATAMENTE como na tabela benchmarks
            for col in colunas_comparacao:
                if col in valores_originais:
                    nome_padrao = obter_nome_padrao(col)
                    categorias.append(nome_padrao)
                    
                    # Valor atual (igual tabela benchmarks)
                    valores_atuais.append(float(valores_originais[col]))
                    
                    # Meta (usa mesma lógica da tabela benchmarks)
                    meta_key = f"target_{col}" if f"target_{col}" in metas else col
                    if meta_key in metas:
                        valores_alvo.append(float(metas[meta_key]))
                        self.logger.info(f"Spider - {col}: Atual={valores_originais[col]:.2f}, Meta={metas[meta_key]:.2f}")
                    else:
                        # Fallback igual tabela (usa valor atual)
                        valores_alvo.append(float(valores_originais[col]))
                        self.logger.info(f"Spider - {col}: Atual={valores_originais[col]:.2f}, Meta=Atual (fallback)")
            
            # Verifica se temos dados para plotar
            if len(valores_atuais) == 0 or len(valores_alvo) == 0:
                self.logger.warning("Nenhum dado válido para o spider chart")
                return ""
            
            # NORMALIZAÇÃO PARA ESCALA 0-1
            # Calcula o máximo valor para cada categoria (atual vs alvo)
            valores_max_por_categoria = []
            for i in range(len(valores_atuais)):
                max_val = max(valores_atuais[i], valores_alvo[i])
                valores_max_por_categoria.append(max_val if max_val > 0 else 1)
            
            # Normaliza os valores (divide cada valor pelo seu máximo)
            valores_atuais_norm = [valores_atuais[i] / valores_max_por_categoria[i] for i in range(len(valores_atuais))]
            valores_alvo_norm = [valores_alvo[i] / valores_max_por_categoria[i] for i in range(len(valores_alvo))]
            
            self.logger.info(f"Spider chart: {len(categorias)} categorias processadas e normalizadas para escala 0-1")
            for i, cat in enumerate(categorias):
                self.logger.info(f"  {cat}: Atual={valores_atuais_norm[i]:.3f}, Meta={valores_alvo_norm[i]:.3f}")
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Erro ao processar dados DEA para spider chart: {e}")
            return ""
        
        # Cria o gráfico spider
        fig = go.Figure()
        
        # Valores atuais (normalizados)
        fig.add_trace(go.Scatterpolar(
            r=valores_atuais_norm,
            theta=categorias,
            fill='toself',
            name='Valores Atuais',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='<b>%{theta}</b><br>Normalizado: %{r:.3f}<extra></extra>'
        ))
        
        # Valores alvo (metas DEA, normalizados)
        fig.add_trace(go.Scatterpolar(
            r=valores_alvo_norm,
            theta=categorias,
            fill='toself',
            name='Metas DEA',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            opacity=0.7,
            hovertemplate='<b>%{theta}</b><br>Normalizado: %{r:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],  # Força escala 0-1
                    gridcolor='lightgray',
                    tickfont=dict(size=10),
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",  # Legenda horizontal
                yanchor="top",
                y=-0.1,  # Posiciona abaixo do gráfico
                xanchor="center",
                x=0.5  # Centraliza horizontalmente
            ),
            title=dict(
                text=f"Spider Chart: Atual vs Metas DEA - CNES {cnes}<br><sub>Valores normalizados (0-1 por categoria)</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            height=650,  # Aumenta altura para acomodar legenda
            width=700,
            margin=dict(t=100, b=100, l=50, r=50),  # Margem ajustada
            annotations=[
                dict(
                    text="Valores normalizados: 1.0 = máximo da categoria",
                    showarrow=False,
                    x=0.5,
                    y=-0.2,  # Move mais para baixo devido à legenda
                    xref="paper",
                    yref="paper",
                    font=dict(size=10, color="gray")
                )
            ]
        )
        
        # Salva gráfico
        caminho_arquivo = self.output_dir / nome_arquivo
        fig.write_html(str(caminho_arquivo))
        self.logger.info(f"Gráfico spider salvo: {caminho_arquivo}")
        
        return str(caminho_arquivo)
    
    def criar_grafico_procedimentos(self, df_mm: pd.DataFrame, cnes: str, competencia: str, 
                                   benchmarks: List[str], nome_arquivo: str) -> str:
        """Cria gráfico de barras dos principais procedimentos replicando o modelo do relatório original."""
        # Dados do CNES alvo
        df_cnes = df_mm[(df_mm['CNES'] == cnes) & (df_mm['COMPETEN'] == competencia)]
        if df_cnes.empty:
            return ""
        
        # Colunas de procedimentos
        cols_procedimentos = [col for col in df_mm.columns if col.startswith(('SIA-', 'SIH-'))]
        if not cols_procedimentos:
            return ""
        
        # Top 10 procedimentos do CNES (como no modelo original)
        valores_cnes = df_cnes[cols_procedimentos].iloc[0]
        valores_cnes = valores_cnes[valores_cnes > 0]
        top_procedimentos = valores_cnes.nlargest(10)
        
        if top_procedimentos.empty:
            return ""
        
        # Preparar dados no formato do modelo original
        procedimentos_selecionados = top_procedimentos.index.tolist()
        
        # Dados dos benchmarks para os mesmos procedimentos
        df_benchmarks = df_mm[
            (df_mm['CNES'].isin(benchmarks)) & 
            (df_mm['COMPETEN'] == competencia)
        ]
        
        # Criar DataFrame similar ao modelo original
        dados_comparacao = []
        
        # Adicionar CNES alvo (primeiro na lista)
        nome_hospital_alvo = df_cnes['DESCESTAB'].iloc[0] if 'DESCESTAB' in df_cnes.columns else f'CNES {cnes}'
        linha_alvo = {'CNES': cnes, 'DESCESTAB': nome_hospital_alvo}
        for proc in procedimentos_selecionados:
            linha_alvo[proc] = top_procedimentos[proc]
        dados_comparacao.append(linha_alvo)
        
        # Adicionar benchmarks (limitando a 3 para melhor visualização)
        for i, bench_cnes in enumerate(benchmarks[:3]):
            df_bench = df_benchmarks[df_benchmarks['CNES'] == bench_cnes]
            if not df_bench.empty:
                nome_hospital_bench = df_bench['DESCESTAB'].iloc[0] if 'DESCESTAB' in df_bench.columns else f'CNES {bench_cnes}'
                linha_bench = {'CNES': bench_cnes, 'DESCESTAB': nome_hospital_bench}
                for proc in procedimentos_selecionados:
                    if proc in df_bench.columns:
                        linha_bench[proc] = df_bench[proc].iloc[0]
                    else:
                        linha_bench[proc] = 0
                dados_comparacao.append(linha_bench)
        
        df_plot = pd.DataFrame(dados_comparacao)
        
        if df_plot.empty:
            return ""
        
        # REPLICAR EXATAMENTE O MODELO ORIGINAL usando matplotlib
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Número de hospitais e procedimentos
        num_hospitais = df_plot.shape[0]
        num_procedimentos = len(procedimentos_selecionados)
        
        # Definindo a posição das barras
        indices = np.arange(num_procedimentos)
        largura = 0.15  # Largura das barras (igual ao modelo)
        
        # Cores para as barras (azuis como no modelo)
        cores_azuis = ['darkblue', 'steelblue', 'lightblue', 'cornflowerblue']
        
        # Plotar barras para cada hospital
        for i in range(num_hospitais):
            valores = [df_plot.iloc[i][proc] for proc in procedimentos_selecionados]
            nome_hospital = df_plot.iloc[i]['DESCESTAB']
            
            # Primeira barra (CNES alvo) em azul escuro, outras em azuis mais claros
            cor = cores_azuis[min(i, len(cores_azuis)-1)]
            
            ax.bar(indices + i * largura, valores, largura, 
                  label=nome_hospital, color=cor)
        
        # Formatação IGUAL ao modelo original
        ax.set_xlabel('', fontsize=8)
        ax.set_ylabel('', fontsize=8)
        ax.set_title(f'Figura 7: Procedimentos mais relevantes no mês (em R$) para o hospital em análise comparados com os de referência', 
                    fontsize=10, pad=20)
        
        # Rótulos do eixo X usando dicionário de constantes
        max_width = 13  # Igual ao modelo original
        wrapped_labels = []
        for proc in procedimentos_selecionados:
            descricao = data_dict.get(proc, proc)
            # Incluir código entre parênteses após a descrição
            texto_completo = f"{descricao} ({proc})"
            # Quebrar em linhas para melhor visualização
            import textwrap
            wrapped_label = '\n'.join(textwrap.wrap(texto_completo, max_width, break_long_words=False, replace_whitespace=False))
            wrapped_labels.append(wrapped_label)
        
        ax.set_xticks(indices + largura * (num_hospitais - 1) / 2)
        ax.set_xticklabels(wrapped_labels, ha='center', fontsize=8)
        
        # Formatação do eixo Y (igual ao modelo)
        from matplotlib.ticker import FuncFormatter
        def format_thousands(x, pos):
            return '{:,.0f}'.format(x)
        
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
        ax.tick_params(axis='y', labelsize=9, width=0)
        
        # Remover bordas (igual ao modelo)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        
        # Grid horizontal fino e cinza (igual ao modelo)
        ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', color='white', linestyle='-', linewidth=0)
        
        # Legenda (igual ao modelo)
        legend = ax.legend(fontsize=8, loc='upper right', frameon=True)
        legend.get_frame().set_facecolor('white')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar gráfico
        caminho_arquivo = self.output_dir / nome_arquivo
        plt.savefig(str(caminho_arquivo.with_suffix('.png')), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Gráfico de procedimentos salvo: {caminho_arquivo}")
        
        return str(caminho_arquivo)
    
    def criar_grafico_procedimentos_plotly(self, df_mm: pd.DataFrame, cnes: str, competencia: str, 
                                          benchmarks: List[str], nome_arquivo: str) -> str:
        """Cria gráfico de barras dos principais procedimentos usando Plotly com quebras de texto."""
        # Dados do CNES alvo
        df_cnes = df_mm[(df_mm['CNES'] == cnes) & (df_mm['COMPETEN'] == competencia)]
        if df_cnes.empty:
            return ""
        
        # Colunas de procedimentos
        cols_procedimentos = [col for col in df_mm.columns if col.startswith(('SIA-', 'SIH-'))]
        if not cols_procedimentos:
            return ""
        
        # Top 10 procedimentos do CNES
        valores_cnes = df_cnes[cols_procedimentos].iloc[0]
        valores_cnes = valores_cnes[valores_cnes > 0]
        top_procedimentos = valores_cnes.nlargest(10)
        
        if top_procedimentos.empty:
            return ""
        
        # Função para formatar nome do procedimento com código
        def formatar_nome_procedimento_plotly(codigo_proc: str) -> str:
            """Converte código para formato 'DESCRIÇÃO (CODIGO)' com quebras de linha."""
            descricao = data_dict.get(codigo_proc, codigo_proc)
            texto_completo = f"{descricao} ({codigo_proc})"
            # Simular quebra de texto para Plotly
            import textwrap
            linhas = textwrap.wrap(texto_completo, width=20, break_long_words=False)
            return '<br>'.join(linhas)  # Plotly usa <br> para quebra de linha
        
        # Dados dos benchmarks para os mesmos procedimentos
        df_benchmarks = df_mm[
            (df_mm['CNES'].isin(benchmarks)) & 
            (df_mm['COMPETEN'] == competencia)
        ]
        
        # Prepara dados para o gráfico Plotly
        procedimentos = top_procedimentos.index.tolist()
        dados_plot = []
        
        # CNES alvo
        for proc in procedimentos:
            nome_formatado = formatar_nome_procedimento_plotly(proc)
            dados_plot.append({
                'Procedimento': nome_formatado,
                'CNES': f'CNES {cnes} (Hospital Analisado)',
                'Valor': top_procedimentos[proc],
                'Codigo': proc
            })
        
        # Benchmarks - limitando a 3 para melhor visualização
        for i, bench_cnes in enumerate(benchmarks[:3]):
            df_bench = df_benchmarks[df_benchmarks['CNES'] == bench_cnes]
            if not df_bench.empty:
                nome_hospital = df_bench['DESCESTAB'].iloc[0] if 'DESCESTAB' in df_bench.columns else f'CNES {bench_cnes}'
                for proc in procedimentos:
                    nome_formatado = formatar_nome_procedimento_plotly(proc)
                    valor = 0
                    if proc in df_bench.columns:
                        valor = df_bench[proc].iloc[0]
                    
                    if valor > 0:  # Só adiciona se tiver valor
                        dados_plot.append({
                            'Procedimento': nome_formatado,
                            'CNES': nome_hospital,
                            'Valor': valor,
                            'Codigo': proc
                        })
        
        df_plot = pd.DataFrame(dados_plot)
        
        if df_plot.empty:
            return ""
        
        # Ordenar procedimentos por valor médio para controlar ordem do eixo Y
        ordem_procedimentos = df_plot.groupby('Procedimento')['Valor'].mean().sort_values(ascending=False)
        df_plot['Procedimento'] = pd.Categorical(df_plot['Procedimento'], 
                                                categories=ordem_procedimentos.index, 
                                                ordered=True)
        
        # Criar gráfico Plotly com cores azuis personalizadas
        cores_azuis = ['#000080', '#4682B4', '#87CEEB', '#6495ED']  # darkblue, steelblue, lightblue, cornflowerblue
        
        fig = go.Figure()
        
        # Obter lista única de CNES para aplicar cores consistentes
        cnes_unicos = df_plot['CNES'].unique()
        
        for i, cnes_nome in enumerate(cnes_unicos):
            df_cnes_plot = df_plot[df_plot['CNES'] == cnes_nome]
            cor = cores_azuis[min(i, len(cores_azuis)-1)]
            
            fig.add_trace(go.Bar(
                y=df_cnes_plot['Procedimento'],
                x=df_cnes_plot['Valor'],
                name=cnes_nome,
                marker_color=cor,
                orientation='h',  # Barras horizontais
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Procedimento: %{y}<br>' +
                             'Valor: R$ %{x:,.0f}<br>' +
                             '<extra></extra>'
            ))
        
        # Layout do gráfico
        fig.update_layout(
            title=dict(
                text='Top 10 Procedimentos - Visualização Interativa<br><sub>Valores em R$ com códigos de procedimento</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Valor (R$)",
            yaxis_title="Procedimentos",
            barmode='group',  # Barras lado a lado
            height=800,  # Aumentar altura para barras horizontais
            showlegend=True,
            legend=dict(
                orientation="h",  # Legenda horizontal
                yanchor="top",
                y=-0.1,  # Posicionar abaixo do gráfico
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=300, r=50, t=80, b=120),  # Margem maior embaixo para legenda e à esquerda para nomes
            font=dict(size=10)
        )
        
        # Formatação do eixo X (agora valores)
        fig.update_xaxes(
            tickformat=',.0f',  # Formato de milhares
            title_font_size=12
        )
        
        # Formatação do eixo Y (agora procedimentos)
        fig.update_yaxes(
            title_font_size=12,
            tickfont=dict(size=9)
        )
        
        # Salva gráfico
        caminho_arquivo = self.output_dir / nome_arquivo
        fig.write_html(str(caminho_arquivo))
        self.logger.info(f"Gráfico Plotly de procedimentos salvo: {caminho_arquivo}")
        
        return str(caminho_arquivo)


class GeradorRelatorio:
    """Classe principal que orquestra a geração do relatório."""
    
    def __init__(self, config: CarregadorConfiguracao):
        self.config = config
        self.carregador = CarregadorDados(config)
        self.processador = ProcessadorDados()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def gerar_relatorio_completo(self, cnes: str, competencia: str, output_dir: str = None) -> str:
        """Gera relatório completo para um CNES em uma competência."""
        self.logger.info(f"Iniciando geração de relatório para CNES {cnes}, competência {competencia}")
        
        # Define diretório de saída
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.project_root / 'resultados' / 'relatorios' / f'relatorio_{cnes}_{competencia}_{timestamp}'
        else:
            output_dir = pathlib.Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        visualizador = GeradorVisualiza(output_dir)
        
        # Carrega dados
        self.logger.info("Carregando dados...")
        df_mm = self.carregador.carregar_media_movel()
        df_dea = self.carregador.carregar_dea()
        df_alertas = self.carregador.carregar_alertas()
        
        # Verifica se CNES existe
        if cnes not in df_mm['CNES'].values:
            raise ErroRelatorio(f"CNES {cnes} não encontrado nos dados.")
        
        # Processa dados
        self.logger.info("Processando dados...")
        df_mm_12_meses = self.processador.obter_ultimos_meses(df_mm, cnes, competencia, 12)
        df_mm_cnes_historico = self.processador.filtrar_dados_cnes(df_mm, cnes)
        # Filtra histórico até a competência solicitada
        df_mm_cnes_historico = df_mm_cnes_historico[df_mm_cnes_historico['COMPETEN'] <= competencia]
        benchmarks = self.processador.extrair_benchmarks(df_dea, cnes, competencia)
        
        # Obtém informações do hospital para o cabeçalho
        info_hospital = self.processador.obter_informacoes_hospital(df_mm, cnes, competencia)
        
        # Obtém informações de eficiência e alvos para o resumo executivo
        info_eficiencia = self.processador.obter_eficiencia_com_variacao(df_dea, cnes, competencia)
        info_alvos = self.processador.obter_alvos_com_diferencas(df_dea, cnes, competencia)
        
        # Dados de alertas para o CNES
        df_alertas_cnes = df_alertas[
            (df_alertas['CNES'] == cnes) & (df_alertas['COMPETEN'] == competencia)
        ] if not df_alertas.empty else pd.DataFrame()
        
        # Gera visualizações
        self.logger.info("Gerando visualizações...")
        tabela_mm = visualizador.criar_tabela_media_movel(df_mm_12_meses, cnes)
        grafico_evolucao = visualizador.criar_grafico_evolucao_temporal(
            df_mm_cnes_historico, df_dea, cnes, competencia, 'evolucao_temporal.html'
        )
        tabela_benchmarks = visualizador.criar_tabela_benchmarks(
            df_dea, df_mm, cnes, competencia, benchmarks
        )
        grafico_spider = visualizador.criar_grafico_spider(
            df_dea, cnes, competencia, 'grafico_spider.html'
        )
        grafico_procedimentos_plotly = visualizador.criar_grafico_procedimentos_plotly(
            df_mm, cnes, competencia, benchmarks, 'procedimentos_interativo.html'
        )
        
        # Gera relatório HTML
        self.logger.info("Gerando relatório HTML...")
        html_relatorio = self._gerar_html_relatorio(
            cnes, competencia, tabela_mm, tabela_benchmarks, 
            df_alertas_cnes, len(benchmarks), output_dir, info_hospital,
            info_eficiencia, info_alvos
        )
        
        # Salva relatório
        arquivo_relatorio = output_dir / 'relatorio_completo.html'
        with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
            f.write(html_relatorio)
        
        self.logger.info(f"Relatório completo gerado: {arquivo_relatorio}")
        return str(arquivo_relatorio)
    
    def gerar_relatorio_embedded(self, cnes: str, competencia: str) -> str:
        """Gera relatório HTML com todas as visualizações incorporadas (sem arquivos externos)."""
        self.logger.info(f"Gerando relatório embedded para CNES {cnes}, competência {competencia}")
        
        # Usa diretório temporário para gerar visualizações
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = pathlib.Path(temp_dir)
            visualizador = GeradorVisualiza(temp_output)
            
            # Carrega dados
            self.logger.info("Carregando dados...")
            df_mm = self.carregador.carregar_media_movel()
            df_dea = self.carregador.carregar_dea()
            df_alertas = self.carregador.carregar_alertas()
            
            # Verifica se CNES existe
            if cnes not in df_mm['CNES'].values:
                raise ErroRelatorio(f"CNES {cnes} não encontrado nos dados.")
            
            # Processa dados
            self.logger.info("Processando dados...")
            df_mm_12_meses = self.processador.obter_ultimos_meses(df_mm, cnes, competencia, 12)
            df_mm_cnes_historico = self.processador.filtrar_dados_cnes(df_mm, cnes)
            df_mm_cnes_historico = df_mm_cnes_historico[df_mm_cnes_historico['COMPETEN'] <= competencia]
            benchmarks = self.processador.extrair_benchmarks(df_dea, cnes, competencia)
            
            # Obtém informações do hospital
            info_hospital = self.processador.obter_informacoes_hospital(df_mm, cnes, competencia)
            info_eficiencia = self.processador.obter_eficiencia_com_variacao(df_dea, cnes, competencia)
            info_alvos = self.processador.obter_alvos_com_diferencas(df_dea, cnes, competencia)
            
            # Dados de alertas para o CNES
            df_alertas_cnes = df_alertas[
                (df_alertas['CNES'] == cnes) & (df_alertas['COMPETEN'] == competencia)
            ] if not df_alertas.empty else pd.DataFrame()
            
            # Gera visualizações
            self.logger.info("Gerando visualizações...")
            tabela_mm = visualizador.criar_tabela_media_movel(df_mm_12_meses, cnes)
            
            # Gráfico de evolução temporal
            grafico_evolucao = visualizador.criar_grafico_evolucao_temporal(
                df_mm_cnes_historico, df_dea, cnes, competencia, 'evolucao_temporal.html'
            )
            
            tabela_benchmarks = visualizador.criar_tabela_benchmarks(
                df_dea, df_mm, cnes, competencia, benchmarks
            )
            
            # Gráfico spider
            grafico_spider = visualizador.criar_grafico_spider(
                df_dea, cnes, competencia, 'grafico_spider.html'
            )
            
            # Gráfico de procedimentos interativo (HTML)
            grafico_procedimentos_plotly = visualizador.criar_grafico_procedimentos_plotly(
                df_mm, cnes, competencia, benchmarks, 'procedimentos_interativo.html'
            )
            
            # Função para converter arquivo HTML para string
            def ler_html_como_string(caminho_arquivo: str) -> str:
                """Lê arquivo HTML e retorna como string."""
                try:
                    if caminho_arquivo and pathlib.Path(caminho_arquivo).exists():
                        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                            return f.read()
                    return ""
                except Exception as e:
                    self.logger.warning(f"Erro ao ler HTML {caminho_arquivo}: {e}")
                    return ""
            
            # Função para converter imagem para base64
            def converter_imagem_base64(caminho_arquivo: str) -> str:
                """Converte imagem para base64 data URL."""
                try:
                    if caminho_arquivo and pathlib.Path(caminho_arquivo).exists():
                        with open(caminho_arquivo, 'rb') as f:
                            img_data = f.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            # Detecta extensão para MIME type
                            ext = pathlib.Path(caminho_arquivo).suffix.lower()
                            mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
                            return f"data:{mime_type};base64,{img_base64}"
                    return ""
                except Exception as e:
                    self.logger.warning(f"Erro ao converter imagem {caminho_arquivo}: {e}")
                    return ""
            
            # Lê conteúdos dos arquivos
            html_evolucao = ler_html_como_string(grafico_evolucao) if grafico_evolucao else ""
            html_spider = ler_html_como_string(grafico_spider) if grafico_spider else ""
            html_procedimentos_plotly = ler_html_como_string(grafico_procedimentos_plotly) if grafico_procedimentos_plotly else ""
            
            # Gera relatório HTML embedded
            html_relatorio = self._gerar_html_relatorio_embedded(
                cnes, competencia, tabela_mm, tabela_benchmarks, 
                df_alertas_cnes, len(benchmarks), info_hospital,
                info_eficiencia, info_alvos, html_evolucao,
                html_spider, html_procedimentos_plotly
            )
            
            self.logger.info("Relatório embedded gerado com sucesso")
            return html_relatorio
    
    def _gerar_html_relatorio(self, cnes: str, competencia: str, tabela_mm: str, 
                             tabela_benchmarks: str, df_alertas: pd.DataFrame, 
                             num_benchmarks: int, output_dir: pathlib.Path, 
                             info_hospital: Dict[str, str], info_eficiencia: Dict[str, Any],
                             info_alvos: Dict[str, Dict[str, float]]) -> str:
        """Gera o HTML do relatório completo."""
        
        # Processa alertas
        alertas_html = ""
        if not df_alertas.empty:
            alertas_ativos = []
            total_alertas = 0
            
            # Processa cada coluna de alerta
            for col in df_alertas.columns:
                if col.startswith('Alerta_') and df_alertas.iloc[0][col] == 1:
                    numero = col.replace('Alerta_', '')
                    descricao = alertas_dict.get(col, f"Alerta {numero}: Descrição não disponível")
                    alertas_ativos.append(f"<li><strong>Alerta {numero}:</strong> {descricao}</li>")
                    total_alertas += 1
            
            # Verificar se existe a coluna total_alertas para validação
            if 'total_alertas' in df_alertas.columns:
                total_no_arquivo = df_alertas.iloc[0]['total_alertas']
                if total_alertas != total_no_arquivo:
                    self.logger.warning(f"Discrepância no total de alertas: calculado={total_alertas}, arquivo={total_no_arquivo}")
            
            if alertas_ativos:
                alertas_html = f"""
                <div class="alert alert-warning">
                    <h6><i class="fas fa-exclamation-triangle"></i> {total_alertas} alerta(s) ativo(s):</h6>
                    <ul class="mb-0">{''.join(alertas_ativos)}</ul>
                </div>
                """
            else:
                alertas_html = '<div class="alert alert-success"><i class="fas fa-check-circle"></i> Nenhum alerta ativo para este CNES.</div>'
        else:
            alertas_html = '<div class="alert alert-info"><i class="fas fa-info-circle"></i> Dados de alertas não disponíveis.</div>'
        
        # Prepara informações de eficiência com formatação
        eficiencia_html = ""
        if info_eficiencia.get('eficiencia_atual') is not None:
            eficiencia_atual = info_eficiencia['eficiencia_atual']
            eficiencia_html = f"<strong>Eficiência:</strong> {eficiencia_atual:.4f}"
            
            # Adiciona variação se disponível
            if info_eficiencia.get('variacao_pct') is not None:
                variacao = info_eficiencia['variacao_pct']
                tem_melhora = info_eficiencia['tem_melhora']
                
                if tem_melhora:
                    cor = "success"
                    seta = "↗"
                else:
                    cor = "danger" 
                    seta = "↘"
                
                eficiencia_html += f' <span class="text-{cor}">({seta} {variacao:+.2f}%)</span>'
            
            eficiencia_html += "<br>"
        else:
            eficiencia_html = "<strong>Eficiência:</strong> Não disponível<br>"
        
        # Prepara informações de alvos
        alvos_html = ""
        if info_alvos:
            alvos_items = []
            for campo, info in info_alvos.items():
                nome = info['nome_campo']
                diff_pct = info['diferenca_pct']
                
                # Se diferença é muito pequena (< 1%), considerar eficiente
                if abs(diff_pct) < 1.0:
                    alvos_items.append(f'<li><strong>{nome}:</strong> <span class="badge bg-success">✅ ≈ Meta atingida</span></li>')
                else:
                    # Lógica de cores CORRIGIDA (igual à tabela de benchmarks):
                    if campo in DEA_INPUT_COLS:
                        # Para INPUTS: redução = PROBLEMA (vermelho), aumento = neutro (laranja)
                        if diff_pct < 0:
                            # Meta menor que atual = PROBLEMA (recursos em excesso)
                            cor = "danger"
                            seta = "↘"
                            texto = f"{seta} Reduzir {abs(diff_pct):.1f}%"
                            tooltip = "Recursos em excesso - requer atenção"
                        else:
                            # Meta maior que atual = NEUTRO (mais recursos necessários)
                            cor = "warning"
                            seta = "↗"
                            texto = f"{seta} Aumentar {diff_pct:.1f}%"
                            tooltip = "Mais recursos necessários"
                    else:
                        # Para OUTPUT: aumento = PROBLEMA (vermelho), redução = bom (verde)
                        if diff_pct > 0:
                            # Meta maior que atual = PROBLEMA (produção insuficiente)
                            cor = "danger"
                            seta = "↗"
                            texto = f"{seta} Aumentar {diff_pct:.1f}%"
                            tooltip = "Produção insuficiente - requer atenção"
                        else:
                            # Meta menor que atual = BOM (produção adequada)
                            cor = "success"
                            seta = "↘"
                            texto = f"{seta} Produção adequada"
                            tooltip = "Produzindo acima da meta"
                    
                    alvos_items.append(f'<li><strong>{nome}:</strong> <span class="badge bg-{cor}" title="{tooltip}">{texto}</span></li>')
            
            alvos_html = f"<ul class='alvos-list'>{''.join(alvos_items)}</ul>"
        else:
            alvos_html = "<p>Dados de alvos não disponíveis.</p>"
        
        html_template = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório CNES {cnes} - {competencia}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; }}
        .card {{ margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
        .table {{ font-size: 0.9rem; }}
        .alert-section {{ background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; }}
        .alert ul {{ margin-bottom: 0; }}
        .alert li {{ margin-bottom: 0.5rem; }}
        .alvos-list li {{ margin-bottom: 0.8rem; }}
        .alvos-list .badge {{ font-size: 0.85em; padding: 0.4em 0.6em; cursor: help; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="text-center">Relatório de Análise Hospitalar</h1>
            <h3 class="text-center">{info_hospital.get('DESCESTAB', 'Nome não disponível')}</h3>
            <p class="text-center lead">CNES: {cnes} | Competência: {competencia}</p>
            <p class="text-center">{info_hospital.get('MUNICIPIO', 'Município não informado')} - {info_hospital.get('UF', 'UF não informada')} | {info_hospital.get('REGIAO', 'Região não informada')}</p>
            <p class="text-center">Relatório de Eficiência Hospitalar</p>
        </div>
    </div>
    
    <div class="container mt-4">
        <!-- Resumo Executivo -->
        <div class="card">
            <div class="card-header">
                <h2>📊 Resumo Executivo</h2>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Informações Básicas</h5>
                        <ul>
                            <li><strong>CNES:</strong> {cnes}</li>
                            <li><strong>Nome:</strong> {info_hospital.get('DESCESTAB', 'Não disponível')}</li>
                            <li><strong>Município:</strong> {info_hospital.get('MUNICIPIO', 'Não informado')}</li>
                            <li><strong>UF:</strong> {info_hospital.get('UF', 'Não informada')}</li>
                            <li><strong>Região:</strong> {info_hospital.get('REGIAO', 'Não informada')}</li>
                            <li><strong>Competência Analisada:</strong> {competencia}</li>
                            <li><strong>Benchmarks Identificados:</strong> {num_benchmarks}</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5>📊 Análise de Performance</h5>
                        <div class="alert-section">
                            {eficiencia_html}
                            <strong>Referências (Benchmarks):</strong>
                            {alvos_html}
                        </div>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-md-12">
                        <h5>🚨 Status de Alertas</h5>
                        <div class="alert-section">
                            {alertas_html}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Dados de Média Móvel -->
        <div class="card">
            <div class="card-header">
                <h2>📈 Dados de Média Móvel - Últimos 12 Meses</h2>
            </div>
            <div class="card-body">
                <p>Dados dos inputs DEA e valores de produção (SIA/SIH) dos últimos 12 meses:</p>
                {tabela_mm}
            </div>
        </div>
        
        <!-- Evolução Temporal -->
        <div class="card">
            <div class="card-header">
                <h2>⏱️ Evolução Temporal</h2>
            </div>
            <div class="card-body">
                <p>Análise da evolução temporal dos inputs/outputs DEA e eficiência:</p>
                <iframe src="evolucao_temporal.html" width="100%" height="800" frameborder="0"></iframe>
            </div>
        </div>
        
        <!-- Comparação com Benchmarks -->
        <div class="card">
            <div class="card-header">
                <h2>🎯 Comparação com Benchmarks</h2>
            </div>
            <div class="card-body">
                <p>Comparação dos inputs e outputs com hospitais benchmark:</p>
                {tabela_benchmarks}
            </div>
        </div>
        
        <!-- Gráfico Spider -->
        <div class="card">
            <div class="card-header">
                <h2>🕸️ Análise Spider - Inputs vs Alvos</h2>
            </div>
            <div class="card-body">
                <p>Comparação visual entre valores atuais e alvos de eficiência:</p>
                <iframe src="grafico_spider.html" width="100%" height="600" frameborder="0"></iframe>
            </div>
        </div>
        
        <!-- Análise de Procedimentos -->
        <div class="card">
            <div class="card-header">
                <h2>🏥 Análise dos Principais Procedimentos</h2>
            </div>
            <div class="card-body">
                <p>Visualização interativa dos principais procedimentos com códigos detalhados:</p>
                <iframe src="procedimentos_interativo.html" width="100%" height="850" frameborder="0"></iframe>
            </div>
        </div>
        
        <!-- Rodapé -->
        <div class="card">
            <div class="card-body text-center text-muted">
                <p>Relatório gerado pelo Sistema de Análise de Eficiência Hospitalar</p>
                <p>Arquivos de visualização salvos em: {output_dir}</p>
                <p><small>Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</small></p>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        return html_template
    
    def _gerar_html_relatorio_embedded(self, cnes: str, competencia: str, tabela_mm: str, 
                                      tabela_benchmarks: str, df_alertas: pd.DataFrame, 
                                      num_benchmarks: int, info_hospital: Dict[str, str], 
                                      info_eficiencia: Dict[str, Any], info_alvos: Dict[str, Dict[str, float]],
                                      html_evolucao: str, html_spider: str, html_procedimentos_plotly: str) -> str:
        """Gera o HTML do relatório com todas as visualizações incorporadas."""
        
        # Processa alertas (igual ao método original)
        alertas_html = ""
        if not df_alertas.empty:
            alertas_ativos = []
            total_alertas = 0
            
            for col in df_alertas.columns:
                if col.startswith('Alerta_') and df_alertas.iloc[0][col] == 1:
                    numero = col.replace('Alerta_', '')
                    descricao = alertas_dict.get(col, f"Alerta {numero}: Descrição não disponível")
                    alertas_ativos.append(f"<li><strong>Alerta {numero}:</strong> {descricao}</li>")
                    total_alertas += 1
            
            if 'total_alertas' in df_alertas.columns:
                total_no_arquivo = df_alertas.iloc[0]['total_alertas']
                if total_alertas != total_no_arquivo:
                    self.logger.warning(f"Discrepância no total de alertas: calculado={total_alertas}, arquivo={total_no_arquivo}")
            
            if alertas_ativos:
                alertas_html = f"""
                <div class="alert alert-warning">
                    <h6><i class="fas fa-exclamation-triangle"></i> {total_alertas} alerta(s) ativo(s):</h6>
                    <ul class="mb-0">{''.join(alertas_ativos)}</ul>
                </div>
                """
            else:
                alertas_html = '<div class="alert alert-success"><i class="fas fa-check-circle"></i> Nenhum alerta ativo para este CNES.</div>'
        else:
            alertas_html = '<div class="alert alert-info"><i class="fas fa-info-circle"></i> Dados de alertas não disponíveis.</div>'
        
        # Prepara informações de eficiência (igual ao método original)
        eficiencia_html = ""
        if info_eficiencia.get('eficiencia_atual') is not None:
            eficiencia_atual = info_eficiencia['eficiencia_atual']
            eficiencia_html = f"<strong>Eficiência:</strong> {eficiencia_atual:.4f}"
            
            if info_eficiencia.get('variacao_pct') is not None:
                variacao = info_eficiencia['variacao_pct']
                tem_melhora = info_eficiencia['tem_melhora']
                
                if tem_melhora:
                    cor = "success"
                    seta = "↗"
                else:
                    cor = "danger" 
                    seta = "↘"
                
                eficiencia_html += f' <span class="text-{cor}">({seta} {variacao:+.2f}%)</span>'
            
            eficiencia_html += "<br>"
        else:
            eficiencia_html = "<strong>Eficiência:</strong> Não disponível<br>"
        
        # Prepara informações de alvos
        alvos_html = ""
        if info_alvos:
            alvos_items = []
            for campo, info in info_alvos.items():
                nome = info['nome_campo']
                diff_pct = info['diferenca_pct']
                
                # Se diferença é muito pequena (< 1%), considerar eficiente
                if abs(diff_pct) < 1.0:
                    alvos_items.append(f'<li><strong>{nome}:</strong> <span class="badge bg-success">✅ ≈ Meta atingida</span></li>')
                else:
                    # Lógica de cores CORRIGIDA (igual à tabela de benchmarks):
                    if campo in DEA_INPUT_COLS:
                        # Para INPUTS: redução = PROBLEMA (vermelho), aumento = neutro (laranja)
                        if diff_pct < 0:
                            # Meta menor que atual = PROBLEMA (recursos em excesso)
                            cor = "danger"
                            seta = "↘"
                            texto = f"{seta} Reduzir {abs(diff_pct):.1f}%"
                            tooltip = "Recursos em excesso - requer atenção"
                        else:
                            # Meta maior que atual = NEUTRO (mais recursos necessários)
                            cor = "warning"
                            seta = "↗"
                            texto = f"{seta} Aumentar {diff_pct:.1f}%"
                            tooltip = "Mais recursos necessários"
                    else:
                        # Para OUTPUT: aumento = PROBLEMA (vermelho), redução = bom (verde)
                        if diff_pct > 0:
                            # Meta maior que atual = PROBLEMA (produção insuficiente)
                            cor = "danger"
                            seta = "↗"
                            texto = f"{seta} Aumentar {diff_pct:.1f}%"
                            tooltip = "Produção insuficiente - requer atenção"
                        else:
                            # Meta menor que atual = BOM (produção adequada)
                            cor = "success"
                            seta = "↘"
                            texto = f"{seta} Produção adequada"
                            tooltip = "Produzindo acima da meta"
                    
                    alvos_items.append(f'<li><strong>{nome}:</strong> <span class="badge bg-{cor}" title="{tooltip}">{texto}</span></li>')
            
            alvos_html = f"<ul class='alvos-list'>{''.join(alvos_items)}</ul>"
        else:
            alvos_html = "<p>Dados de alvos não disponíveis.</p>"

        # Função para extrair corpo dos HTMLs Plotly (remove <html>, <head>, etc.)
        def extrair_corpo_plotly(html_content: str) -> str:
            """Extrai o conteúdo principal do Plotly (body completo para garantir funcionamento)."""
            if not html_content:
                return "<p>Gráfico não disponível.</p>"
            
            import re
            
            # Para gráficos Plotly embedded, é mais seguro usar o body completo
            # pois há dependências entre scripts que podem ser quebradas
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)
            if body_match:
                body_content = body_match.group(1)
                # Verificar se contém elementos Plotly essenciais
                if 'plotly-graph-div' in body_content and 'Plotly.newPlot' in body_content:
                    return body_content
                else:
                    return "<p>Gráfico Plotly não encontrado no conteúdo.</p>"
            else:
                # Fallback: assume que é conteúdo já extraído
                if 'plotly-graph-div' in html_content and 'Plotly.newPlot' in html_content:
                    return html_content
                else:
                    return "<p>Gráfico não disponível.</p>"
        
        # Extrai corpos dos HTMLs Plotly
        corpo_evolucao = extrair_corpo_plotly(html_evolucao)
        corpo_spider = extrair_corpo_plotly(html_spider)
        corpo_procedimentos_plotly = extrair_corpo_plotly(html_procedimentos_plotly)

        html_template_embedded = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório CNES {cnes} - {competencia}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; }}
        .card {{ margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
        .table {{ font-size: 0.9rem; }}
        .alert-section {{ background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; }}
        .alert ul {{ margin-bottom: 0; }}
        .alert li {{ margin-bottom: 0.5rem; }}
        .alvos-list li {{ margin-bottom: 0.8rem; }}
        .alvos-list .badge {{ font-size: 0.85em; padding: 0.4em 0.6em; cursor: help; }}
        .plotly-container {{ min-height: 400px; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="text-center">Relatório de Análise Hospitalar</h1>
            <h3 class="text-center">{info_hospital.get('DESCESTAB', 'Nome não disponível')}</h3>
            <p class="text-center lead">CNES: {cnes} | Competência: {competencia}</p>
            <p class="text-center">{info_hospital.get('MUNICIPIO', 'Município não informado')} - {info_hospital.get('UF', 'UF não informada')} | {info_hospital.get('REGIAO', 'Região não informada')}</p>
            <p class="text-center">Relatório de Eficiência Hospitalar</p>
        </div>
    </div>
    
    <div class="container mt-4">
        <!-- Resumo Executivo -->
        <div class="card">
            <div class="card-header">
                <h2>📊 Resumo Executivo</h2>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Informações Básicas</h5>
                        <ul>
                            <li><strong>CNES:</strong> {cnes}</li>
                            <li><strong>Nome:</strong> {info_hospital.get('DESCESTAB', 'Não disponível')}</li>
                            <li><strong>Município:</strong> {info_hospital.get('MUNICIPIO', 'Não informado')}</li>
                            <li><strong>UF:</strong> {info_hospital.get('UF', 'Não informada')}</li>
                            <li><strong>Região:</strong> {info_hospital.get('REGIAO', 'Não informada')}</li>
                            <li><strong>Competência Analisada:</strong> {competencia}</li>
                            <li><strong>Benchmarks Identificados:</strong> {num_benchmarks}</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5>📊 Análise de Performance</h5>
                        <div class="alert-section">
                            {eficiencia_html}
                            <strong>Referências (Benchmarks):</strong>
                            {alvos_html}
                        </div>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-md-12">
                        <h5>🚨 Status de Alertas</h5>
                        <div class="alert-section">
                            {alertas_html}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Dados de Média Móvel -->
        <div class="card">
            <div class="card-header">
                <h2>📈 Dados de Média Móvel - Últimos 12 Meses</h2>
            </div>
            <div class="card-body">
                <p>Dados dos inputs DEA e valores de produção (SIA/SIH) dos últimos 12 meses:</p>
                {tabela_mm}
            </div>
        </div>
        
        <!-- Evolução Temporal -->
        <div class="card">
            <div class="card-header">
                <h2>⏱️ Evolução Temporal</h2>
            </div>
            <div class="card-body">
                <p>Análise da evolução temporal dos inputs/outputs DEA e eficiência:</p>
                <div class="plotly-container">
                    {corpo_evolucao}
                </div>
            </div>
        </div>
        
        <!-- Comparação com Benchmarks -->
        <div class="card">
            <div class="card-header">
                <h2>🎯 Comparação com Benchmarks</h2>
            </div>
            <div class="card-body">
                <p>Comparação dos inputs e outputs com hospitais benchmark:</p>
                {tabela_benchmarks}
            </div>
        </div>
        
        <!-- Gráfico Spider -->
        <div class="card">
            <div class="card-header">
                <h2>🕸️ Análise Spider - Inputs vs Alvos</h2>
            </div>
            <div class="card-body">
                <p>Comparação visual entre valores atuais e alvos de eficiência:</p>
                <div class="plotly-container">
                    {corpo_spider}
                </div>
            </div>
        </div>
        
        <!-- Análise de Procedimentos -->
        <div class="card">
            <div class="card-header">
                <h2>🏥 Análise dos Principais Procedimentos</h2>
            </div>
            <div class="card-body">
                <p>Visualização interativa dos principais procedimentos com códigos detalhados:</p>
                <div class="plotly-container">
                    {corpo_procedimentos_plotly}
                </div>
            </div>
        </div>
        
        <!-- Rodapé -->
        <div class="card">
            <div class="card-body text-center text-muted">
                <p>Relatório gerado pelo Sistema de Análise de Eficiência Hospitalar</p>
                <p><i class="fas fa-check-circle text-success"></i> Relatório autossuficiente - todas as visualizações incorporadas</p>
                <p><small>Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</small></p>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        return html_template_embedded


def configurar_logging():
    """Configura o sistema de logging."""
    pathlib.Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/relatorio_cnes.log', encoding='utf-8')
        ]
    )


def parsear_argumentos() -> argparse.Namespace:
    """Configura e parseia os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Gera relatório detalhado para um CNES específico com análises e visualizações.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cnes', required=True, help='Código CNES do hospital')
    parser.add_argument('--competencia', required=True, help='Competência para análise (formato AAAAMM)')
    parser.add_argument('--output_dir', help='Diretório de saída customizado')
    parser.add_argument('--config', default='config.yaml', help='Arquivo de configuração')
    
    return parser.parse_args()


def main():
    """Função principal."""
    args = parsear_argumentos()
    
    try:
        # Configuração
        configurar_logging()
        logging.info("=== INICIANDO GERAÇÃO DE RELATÓRIO POR CNES ===")
        
        config = CarregadorConfiguracao(args.config)
        gerador = GeradorRelatorio(config)
        
        # Gera relatório
        arquivo_relatorio = gerador.gerar_relatorio_completo(
            args.cnes, args.competencia, args.output_dir
        )
        
        print(f"\n✅ Relatório gerado com sucesso!")
        print(f"📁 Arquivo principal: {arquivo_relatorio}")
        print(f"🌐 Abra o arquivo no navegador para visualizar o relatório completo.")
        
        logging.info("=== GERAÇÃO DE RELATÓRIO CONCLUÍDA COM SUCESSO ===")
        
    except ErroRelatorio as e:
        logging.error(f"Erro no relatório: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Erro inesperado: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 