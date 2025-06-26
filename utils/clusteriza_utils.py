import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import logging
from .constantes import (
    NUM_REGISTROS_PROXIMOS, 
    INCREMENTO_FATOR_ESCALA, 
    cols_sia, 
    cols_sih, 
    colunas_cluster,
    MIN_VALOR_POSITIVO,
    MAX_FATOR_ESCALA
)

logger = logging.getLogger(__name__)

def filtrar_por_periodo(df: pd.DataFrame, data_inicio: str, data_fim: str, competencia_col: str = 'COMPETEN') -> pd.DataFrame:
    """
    Filtra o DataFrame por período de competência.
    
    Args:
        df: DataFrame com os dados a serem filtrados
        data_inicio: Data inicial no formato aaaamm
        data_fim: Data final no formato aaaamm
        competencia_col: Nome da coluna de competência
    
    Returns:
        DataFrame filtrado pelo período especificado
        
    Raises:
        ValueError: Se o formato das datas for inválido
    """
    try:
        # Validar formato das datas
        if not (len(data_inicio) == 6 and len(data_fim) == 6 and data_inicio.isdigit() and data_fim.isdigit()):
            raise ValueError(f"Datas devem estar no formato aaaamm. Recebido: {data_inicio} e {data_fim}")
            
        df_filtered = df[(df[competencia_col] >= data_inicio) & (df[competencia_col] <= data_fim)].copy()
        if df_filtered.empty:
            logger.warning(f"Nenhum dado encontrado no período {data_inicio} a {data_fim}")
        else:
            logger.info(f"Filtrados {len(df_filtered)} registros no período {data_inicio} a {data_fim}")
            
        return df_filtered
    except KeyError as e:
        logger.error(f"Coluna de competência '{competencia_col}' não encontrada: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro ao filtrar por período: {e}")
        raise

def validar_dados_numericos(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Valida e limpa valores numéricos no DataFrame.
    
    Args:
        df: DataFrame a ser validado
        colunas: Lista de colunas numéricas a serem validadas
    
    Returns:
        DataFrame com dados validados e limpos
    """
    df_validado = df.copy()
    
    # Verificar valores nulos
    nulos = df_validado[colunas].isnull().sum()
    if nulos.sum() > 0:
        # logger.warning(f"Encontrados valores nulos: {nulos[nulos > 0].to_dict()}")
        pass
        
    # Substituir nulos por zeros
    for col in colunas:
        if col in df_validado.columns:
            nulos_coluna = df_validado[col].isnull().sum()
            if nulos_coluna > 0:
                # logger.warning(f"Substituindo {nulos_coluna} valores nulos em '{col}' por zeros")
                df_validado[col] = df_validado[col].fillna(0)
    
    # Verificar e ajustar valores negativos
    for col in colunas:
        if col in df_validado.columns:
            negs = (df_validado[col] < 0).sum()
            if negs > 0:
                logger.warning(f"Encontrados {negs} valores negativos em '{col}', substituindo por {MIN_VALOR_POSITIVO}")
                df_validado.loc[df_validado[col] < 0, col] = MIN_VALOR_POSITIVO
    
    return df_validado

def gerar_cluster_kmeans(df: pd.DataFrame, colunas_interesse: List[str], data_inicio: str, data_fim: str, min_cluster_size: int = 15) -> pd.DataFrame:
    """
    Gera clusters usando KMeans com ajuste dinâmico do número de clusters.
    
    Args:
        df: DataFrame com os dados para clusterização
        colunas_interesse: Lista de colunas para usar na clusterização
        data_inicio: Data inicial no formato aaaamm
        data_fim: Data final no formato aaaamm
        min_cluster_size: Tamanho mínimo para um cluster
    
    Returns:
        DataFrame com os clusters gerados
    """
    try:
        df_filtered = filtrar_por_periodo(df, data_inicio, data_fim)
        if df_filtered.empty:
            return pd.DataFrame()
            
        # Validar dados numéricos
        df_filtered = validar_dados_numericos(df_filtered, colunas_interesse)
            
        # Verificar colunas necessárias
        colunas_faltantes = [col for col in colunas_interesse if col not in df_filtered.columns]
        if colunas_faltantes:
            logger.error(f"Colunas faltantes: {colunas_faltantes}")
            raise ValueError(f"Colunas necessárias ausentes: {colunas_faltantes}")

        k = 5
        while True:
            logger.info(f"Tentando para k igual a {k}")
            X = df_filtered[colunas_interesse]
            
            # Remover linhas com valores atípicos extremos
            X_filtered = X[(X < X.quantile(0.99)).all(axis=1)]
            if len(X_filtered) < len(X):
                logger.warning(f"Removidas {len(X) - len(X_filtered)} linhas com valores atípicos")
                if len(X_filtered) < min_cluster_size * 2:
                    logger.warning("Poucos registros após filtragem. Usando dados originais.")
                    X_filtered = X
            
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_filtered)
                counts = pd.Series(kmeans.labels_).value_counts()

                if (counts < min_cluster_size).any():
                    k -= 1
                    break
                k += 1
            except Exception as e:
                logger.error(f"Erro no KMeans: {e}")
                k -= 1
                break

        k = max(k, 2)
        logger.info(f"k final ajustado: {k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        # Calcular a distância de cada ponto ao centroide do seu cluster
        distances = kmeans.transform(X)  # Distâncias de todos os pontos a todos os centroides
        df_filtered['Distancia_Ref'] = distances.min(axis=1)  # Armazenar a menor distância para cada ponto

        # Adicionar os rótulos dos clusters ao DataFrame
        df_filtered['Cluster'] = kmeans.labels_.astype(str) + '-' + df_filtered['COMPETEN'].astype(str)
        df_filtered['Cluster'] = df_filtered['Cluster'].apply(lambda x: f"{x.split('-')[0].zfill(2)}-{x.split('-')[1]}")
        df_filtered['n'] = df_filtered.groupby('Cluster')['Cluster'].transform('count')
        df_filtered['f'] = 0

        # Calcular a distância média por cluster
        distancia_media = df_filtered.groupby('Cluster')['Distancia_Ref'].mean().reset_index()
        distancia_media.columns = ['Cluster', 'Distancia_Ref_Media']

        # Juntar a média de distância de volta ao DataFrame original
        df_filtered = df_filtered.merge(distancia_media, on='Cluster', how='left')

        logger.info("Distribuição dos clusters:\n" + str(df_filtered['Cluster'].value_counts()))
        return df_filtered[colunas_cluster + ['Distancia_Ref_Media']]
    except Exception as e:
        logger.error(f"Erro ao gerar clusters: {e}")
        return pd.DataFrame()

def identificar_maiores_e_somar_outros(df: pd.DataFrame, competencia: str, percentual_maiores: float = 0.9) -> pd.DataFrame:
    """
    Processa um DataFrame por competência, aplicando curva ABC e somando 'Outros'.
    
    Args:
        df: DataFrame com dados a serem processados
        competencia: Competência a ser processada (aaaamm)
        percentual_maiores: Percentual a ser considerado como "maiores" (0.0 a 1.0)
    
    Returns:
        DataFrame processado com a coluna 'Outros' adicionada
    """
    try:
        df_competencia = df[df['COMPETEN'] == competencia].copy()
        if df_competencia.empty:
            logger.info(f"Nenhum registro para a competência {competencia}")
            return pd.DataFrame()

        colunas_relevantes = [col for col in df_competencia.columns 
                            if (col.startswith('SIA-') or col.startswith('SIH-')) 
                            and not col.endswith('_p') 
                            and df_competencia[col].sum() > 0]
                            
        if not colunas_relevantes:
            logger.warning(f"Nenhuma coluna relevante encontrada para a competência {competencia}")
            return df_competencia
            
        colunas_originais = df_competencia.columns.tolist()
        resultados = []

        for cnes_valor in df_competencia['CNES'].unique():
            df_cnes = df_competencia[df_competencia['CNES'] == cnes_valor].copy()
            valores = df_cnes[colunas_relevantes].iloc[0]
            valores = valores[valores > 0]
            if valores.empty:
                df_cnes.loc[:, 'Outros'] = 0
                resultados.append(df_cnes)
                continue

            soma_total = valores.sum()
            valores_ordenados = valores.sort_values(ascending=False)
            soma_acumulada = valores_ordenados.cumsum()
            limite = percentual_maiores * soma_total
            idx_primeiro_acima = soma_acumulada[soma_acumulada >= limite].index[0] if any(soma_acumulada >= limite) else valores_ordenados.index[-1]
            colunas_maiores = valores_ordenados.index[:valores_ordenados.index.get_loc(idx_primeiro_acima) + 1].tolist()

            outras_colunas = list(set(colunas_relevantes) - set(colunas_maiores))
            df_cnes.loc[:, 'Outros'] = df_cnes[outras_colunas].sum(axis=1)
            df_cnes.loc[:, outras_colunas] = 0
            resultados.append(df_cnes)

        if not resultados:
            return pd.DataFrame()
        
        df_final = pd.concat(resultados, ignore_index=True)
        colunas_finais = [col for col in colunas_originais if col != 'Outros'] + ['Outros']
        return df_final[colunas_finais]
        
    except Exception as e:
        logger.error(f"Erro ao identificar maiores e somar outros para competência {competencia}: {e}")
        return pd.DataFrame()

def criar_cluster(df_original: pd.DataFrame, matriz_distancias: np.ndarray, indices_proximos: np.ndarray, indice_referencia: int) -> pd.DataFrame:
    """
    Cria um cluster com base nos índices mais próximos.
    
    Args:
        df_original: DataFrame original
        matriz_distancias: Matriz de distâncias calculadas
        indices_proximos: Índices dos registros mais próximos
        indice_referencia: Índice de referência
        
    Returns:
        DataFrame com o cluster criado
    """
    try:
        # Garante que estamos pegando o número certo de vizinhos, mesmo que seja menor que o total disponível
        num_vizinhos_a_pegar = min(len(df_original), NUM_REGISTROS_PROXIMOS)
        
        # `indices_proximos` é uma matriz (1, N) com os vizinhos para o `indice_referencia`.
        # Portanto, o acesso à linha deve ser sempre 0.
        indices_cluster = indices_proximos[0, :num_vizinhos_a_pegar]

        df_cluster = df_original.iloc[indices_cluster].copy()
        
        # Pega as distâncias correspondentes aos índices do cluster
        # Usa o `indice_referencia` para obter a linha correta da matriz de distâncias completa.
        distancias_referencia = matriz_distancias[indice_referencia][indices_cluster]

        df_cluster['Distancia_Ref'] = distancias_referencia
        
        # Calcula a soma dos valores de produção para a referência (primeira linha)
        valor_total_referencia = df_cluster.iloc[0][cols_sia + cols_sih].sum()
        
        # Evita divisão por zero se a soma for 0
        df_cluster['Dist_Relativa'] = (df_cluster['Distancia_Ref'] / (valor_total_referencia + MIN_VALOR_POSITIVO)).round(4)
        
        df_cluster['Cluster'] = f"{df_cluster.iloc[0]['CNES']}-{df_cluster.iloc[0]['COMPETEN']}"
        return df_cluster.sort_values(by='Distancia_Ref').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Erro ao criar cluster: {e}", exc_info=True)
        raise

def validar_registros(df_cluster: pd.DataFrame, fator_escala: float) -> pd.DataFrame:
    """
    Valida registros do cluster com base em similaridade.
    
    Args:
        df_cluster: DataFrame do cluster a ser validado
        fator_escala: Fator de escala para comparação
        
    Returns:
        DataFrame com coluna de validação (VALIDA) adicionada
    """
    try:
        horas_medicos_ref = max(df_cluster.iloc[0]['HORAS_MEDICOS'], MIN_VALOR_POSITIVO)
        leitos_sus_ref = max(df_cluster.iloc[0]['CNES_LEITOS_SUS'], MIN_VALOR_POSITIVO)
        
        limite_sup_horas = fator_escala * horas_medicos_ref
        limite_inf_horas = horas_medicos_ref / fator_escala
        limite_sup_leitos = fator_escala * leitos_sus_ref
        limite_inf_leitos = leitos_sus_ref / fator_escala
        
        condicoes = (
            (df_cluster['HORAS_MEDICOS'] < limite_sup_horas) &
            (df_cluster['HORAS_MEDICOS'] > limite_inf_horas) &
            (df_cluster['CNES_LEITOS_SUS'] < limite_sup_leitos) &
            (df_cluster['CNES_LEITOS_SUS'] > limite_inf_leitos)
        )
        df_cluster['VALIDA'] = condicoes
        return df_cluster
    except Exception as e:
        logger.error(f"Erro ao validar registros: {e}")
        raise

def is_hospital_geral(df: pd.DataFrame, cnes: str) -> bool:
    """
    Verifica se o CNES é um hospital geral.
    
    Args:
        df: DataFrame com os dados
        cnes: Código CNES do estabelecimento
        
    Returns:
        True se for hospital geral, False caso contrário
    """
    try:
        return not df[(df['CNES'] == cnes) & (df['TIPO_UNIDADE'] == 'HOSPITAL GERAL')].empty
    except Exception as e:
        logger.error(f"Erro ao verificar se {cnes} é hospital geral: {e}")
        return False

def is_hospital_complexidade(df: pd.DataFrame, cnes: str) -> pd.DataFrame:
    """
    Filtra hospitais com a mesma complexidade do CNES.
    
    Args:
        df: DataFrame com os dados
        cnes: Código CNES do estabelecimento
        
    Returns:
        DataFrame filtrado por complexidade
    """
    try:
        hospital_ref = df[df['CNES'] == cnes]
        if hospital_ref.empty:
            logger.warning(f"CNES {cnes} não encontrado no DataFrame")
            return pd.DataFrame()
            
        comp_amb_ref = hospital_ref['Complexidade_Ambulatorial'].iloc[0]
        comp_hosp_ref = hospital_ref['Complexidade_Hospitalar'].iloc[0]
        
        return df[(df['Complexidade_Ambulatorial'] == comp_amb_ref) & 
                (df['Complexidade_Hospitalar'] == comp_hosp_ref)]
    except KeyError as e:
        logger.error(f"Coluna de complexidade não encontrada: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erro ao filtrar por complexidade para CNES {cnes}: {e}")
        return pd.DataFrame()

def gerar_cluster(df_original: pd.DataFrame, colunas_interesse: List[str], cnes: str, max_fator_escala: float = 10, 
                 min_registros_validos: int = 15, tipo_cluster: str = 'cluster') -> pd.DataFrame:
    """
    Gera clusters por CNES usando distância euclidiana.
    
    Args:
        df_original: DataFrame original com dados
        colunas_interesse: Colunas para cálculo de distância
        cnes: Código CNES de referência
        max_fator_escala: Fator de escala máximo para validação
        min_registros_validos: Número mínimo de registros válidos
        tipo_cluster: Tipo de cluster ('cluster' ou 'cluster_sf')
        
    Returns:
        DataFrame com o cluster gerado
    """
    try:
        # Validar CNES
        if not cnes or not isinstance(cnes, str):
            raise ValueError(f"CNES inválido: {cnes}")
            
        if is_hospital_geral(df_original, cnes):
            df_original = df_original[df_original['TIPO_UNIDADE'] == 'HOSPITAL GERAL']
        
        df_filtered = is_hospital_complexidade(df_original, cnes)
        if df_filtered.empty:
            logger.warning(f"Nenhum hospital encontrado com a mesma complexidade do CNES {cnes}")
            return pd.DataFrame()

        # Validar dados numéricos
        df_filtered = validar_dados_numericos(df_filtered, colunas_interesse)

        matriz_distancias = euclidean_distances(df_filtered[colunas_interesse])
        indices_mais_proximos = np.argsort(matriz_distancias, axis=1)[:, :NUM_REGISTROS_PROXIMOS]
        
        indice_encontrado = False
        for i in range(len(df_filtered)):
            if df_filtered.iloc[i]['CNES'] == cnes:
                indice_encontrado = True
                df_cluster = criar_cluster(df_filtered, matriz_distancias, indices_mais_proximos, i)
                
                if tipo_cluster == 'cluster':
                    fator_escala = 2
                    condicao_atendida = False
                    while not condicao_atendida:
                        df_cluster = validar_registros(df_cluster, fator_escala)
                        if (df_cluster[df_cluster['VALIDA'] == True].shape[0] >= min_registros_validos or 
                            fator_escala > max_fator_escala):
                            condicao_atendida = True
                            if fator_escala > max_fator_escala:
                                logger.warning(f"Fator de escala máximo atingido ({max_fator_escala}), forçando VALIDA=True")
                                df_cluster['VALIDA'] = True
                        else:
                            fator_escala += INCREMENTO_FATOR_ESCALA
                    
                    df_filtrado = df_cluster[df_cluster['VALIDA'] == True].copy()
                    df_filtrado['n'] = df_filtrado.shape[0]
                    df_filtrado['f'] = fator_escala
                    
                    if df_filtrado[df_filtrado['Dist_Relativa'] < 1].shape[0] < min_registros_validos:
                        logger.warning(f"Poucos registros com distância relativa < 1, selecionando os {min_registros_validos} mais próximos")
                        df_filtrado = df_filtrado.nsmallest(min_registros_validos, 'Dist_Relativa')
                        df_filtrado['n'] = min_registros_validos
                        df_filtrado['f'] = -1
                    
                    return df_filtrado[colunas_cluster]
                
                elif tipo_cluster == 'cluster_sf':
                    df_filtrado = df_cluster.head(50).copy()
                    df_filtrado['n'] = 50
                    df_filtrado['f'] = 1
                    return df_filtrado[colunas_cluster]
                
        if not indice_encontrado:
            logger.warning(f"CNES {cnes} não encontrado no DataFrame filtrado")
        
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Erro ao gerar cluster para CNES {cnes}: {e}")
        return pd.DataFrame()

def gerar_cluster_de_matriz_precalculada(
    df_candidatos: pd.DataFrame, 
    matriz_distancias: np.ndarray,
    indices_proximos: np.ndarray,
    indice_referencia: int,
    cnes_referencia: str,
    min_registros_validos: int
) -> Optional[pd.DataFrame]:
    """
    Gera um cluster para um CNES de referência usando uma matriz de distância pré-calculada.
    Esta função contém a lógica de validação por fator de escala e distância relativa.
    Retorna um DataFrame com as informações do cluster ou None se não for possível gerar.
    """
    try:
        # 1. Criar o cluster inicial com os N mais próximos
        df_cluster = criar_cluster(df_candidatos, matriz_distancias, indices_proximos, indice_referencia)
        if df_cluster.empty:
            return None

        # 2. Iterar com fator de escala para validar estrutura (leitos, médicos)
        fator_escala = 2.0
        condicao_atendida = False
        df_validado_estrutura = pd.DataFrame()

        while not condicao_atendida:
            df_cluster_validado_temp = validar_registros(df_cluster.copy(), fator_escala)
            registros_ok = df_cluster_validado_temp[df_cluster_validado_temp['VALIDA'] == True]
            
            # Condição de parada: tem registros suficientes OU fator de escala muito alto
            if len(registros_ok) >= (min_registros_validos + 1) or fator_escala > MAX_FATOR_ESCALA:
                condicao_atendida = True
                df_validado_estrutura = registros_ok
                if fator_escala > MAX_FATOR_ESCALA:
                    logger.warning(f"Fator de escala máximo ({MAX_FATOR_ESCALA}) atingido para CNES {cnes_referencia}. Usando {len(registros_ok)} registros.")
            else:
                fator_escala += INCREMENTO_FATOR_ESCALA

        if df_validado_estrutura.empty:
            logger.warning(f"Nenhum vizinho passou na validação de estrutura para o CNES {cnes_referencia}.")
            return None
            
        # 3. Aplicar filtro de similaridade de produção (Dist_Relativa < 1)
        df_alta_similaridade = df_validado_estrutura[df_validado_estrutura['Dist_Relativa'] < 1]

        # 4. Determinar o DataFrame final
        df_final = pd.DataFrame()
        f_val = fator_escala

        if len(df_alta_similaridade) >= (min_registros_validos + 1):
            # Caso ideal: temos vizinhos suficientes com alta similaridade
            df_final = df_alta_similaridade
        else:
            # Caso "melhor esforço": não temos o ideal, então pegamos os mais próximos do grupo que passou na estrutura
            logger.warning(f"Poucos vizinhos com alta similaridade para CNES {cnes_referencia} ({len(df_alta_similaridade)} encontrados). Selecionando os {min_registros_validos+1} mais próximos do grupo validado estruturalmente.")
            df_final = df_validado_estrutura.head(min_registros_validos + 1)
            f_val = -1 # Sinaliza que a regra de similaridade de produção não foi o critério final

        if df_final.empty:
            return None

        # Remover o próprio hospital de referência da lista de vizinhos
        df_final = df_final[df_final['CNES'] != cnes_referencia]
        
        # Garantir que o resultado final tenha o número mínimo de vizinhos
        if len(df_final) < min_registros_validos:
             logger.warning(f"Resultado final para CNES {cnes_referencia} tem menos de {min_registros_validos} vizinhos.")
             return None

        # 5. Formatar o resultado final
        resultado = pd.DataFrame([{
            'n': len(df_final),
            'f': f_val
        }])
        
        # Adiciona a lista de vizinhos (CNES e Distancia)
        resultado['CNES'] = cnes_referencia # Adicionado para merge posterior, se necessário
        resultado['Distancia_Ref'] = 0 # Distância da referência para ela mesma
        
        # Concatena os vizinhos, usando a Distância Relativa
        vizinhos_df = df_final[['CNES', 'Dist_Relativa']].copy()
        vizinhos_df.rename(columns={'Dist_Relativa': 'Distancia_Relativa'}, inplace=True)
        
        # Cria o resultado final no formato esperado
        resultado_final_dict = {
            'n': len(df_final),
            'f': f_val,
            'CNES': [cnes_referencia] + vizinhos_df['CNES'].tolist(),
            'Distancia_Relativa': [0] + vizinhos_df['Distancia_Relativa'].tolist()
        }

        # Retorna um DataFrame com uma linha para o hospital de referência, contendo o cluster
        return pd.DataFrame({
            'n': [len(df_final)],
            'f': [f_val],
            'CNES': [cnes_referencia],
            'Distancia_Ref': [0], # Placeholder, a distância é relativa a cada vizinho
            'Cluster_JSON_List': [vizinhos_df.to_dict('records')] # Coluna com o JSON
        })

    except Exception as e:
        logger.error(f"Erro ao gerar cluster de matriz pré-calculada para CNES {cnes_referencia}: {e}", exc_info=True)
        return None