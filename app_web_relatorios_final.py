#!/usr/bin/env python3
"""
Aplicação Web para Geração de Relatórios de Eficiência Hospitalar - VERSÃO FINAL
Interface compacta com visualização inline e exportação simplificada.
"""

import streamlit as st
import pandas as pd
import pathlib
import sys
import tempfile
import os
import base64
from datetime import datetime
import logging
import streamlit.components.v1 as components

# Configuração da página
st.set_page_config(
    page_title="Relatórios de Eficiência Hospitalar",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adiciona o diretório raiz do projeto ao sys.path
project_root_path = pathlib.Path(__file__).resolve().parent
if str(project_root_path) not in sys.path:
    sys.path.append(str(project_root_path))

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .report-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Importações do projeto
try:
    from scripts.relatorio_cnes import CarregadorConfiguracao, CarregadorDados, GeradorRelatorio
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

@st.cache_data(ttl=3600)
def carregar_configuracao():
    """Carrega configuração com cache."""
    return CarregadorConfiguracao('config.yaml')

@st.cache_data(ttl=3600)
def carregar_dados_base():
    """Carrega dados base com cache."""
    try:
        config = carregar_configuracao()
        carregador = CarregadorDados(config)
        
        df_mm = carregador.carregar_media_movel()
        
        cnes_disponiveis = sorted(df_mm['CNES'].unique())
        competencias_disponiveis = sorted(df_mm['COMPETEN'].unique(), reverse=True)
        
        hospitais_info = {}
        for cnes in cnes_disponiveis:
            df_hospital = df_mm[df_mm['CNES'] == cnes]
            if not df_hospital.empty:
                nome = 'Hospital não identificado'
                municipio = 'N/A'
                uf = 'N/A'
                
                if 'DESCESTAB' in df_hospital.columns:
                    nome = str(df_hospital['DESCESTAB'].iloc[0])
                if 'MUNICIPIO' in df_hospital.columns and pd.notna(df_hospital['MUNICIPIO'].iloc[0]):
                    municipio = str(df_hospital['MUNICIPIO'].iloc[0])
                if 'UF' in df_hospital.columns and pd.notna(df_hospital['UF'].iloc[0]):
                    uf = str(df_hospital['UF'].iloc[0])
                
                hospitais_info[cnes] = {
                    'nome': nome,
                    'municipio': municipio,
                    'uf': uf,
                    'display': f"{nome} - {municipio}/{uf}"
                }
            else:
                hospitais_info[cnes] = {
                    'nome': f"CNES {cnes}",
                    'municipio': 'N/A',
                    'uf': 'N/A',
                    'display': f"CNES {cnes}"
                }
        
        return cnes_disponiveis, competencias_disponiveis, hospitais_info, df_mm
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return [], [], {}, pd.DataFrame()

def formatar_competencia(competencia):
    """Formata competência para exibição."""
    if len(str(competencia)) == 6:
        ano = str(competencia)[:4]
        mes = str(competencia)[4:]
        meses = {
            '01': 'Jan', '02': 'Fev', '03': 'Mar', '04': 'Abr',
            '05': 'Mai', '06': 'Jun', '07': 'Jul', '08': 'Ago',
            '09': 'Set', '10': 'Out', '11': 'Nov', '12': 'Dez'
        }
        return f"{meses.get(mes, mes)}/{ano}"
    return str(competencia)

def gerar_relatorio_web(cnes, competencia):
    """Gera relatório embedded e retorna o HTML."""
    try:
        config = carregar_configuracao()
        gerador = GeradorRelatorio(config)
        
        # Gera relatório embedded (tudo incorporado)
        html_content = gerador.gerar_relatorio_embedded(cnes, competencia)
        
        return html_content
            
    except Exception as e:
        st.error(f"Erro ao gerar relatório: {e}")
        return None

def main():
    """Função principal da aplicação."""
    
    # Cabeçalho principal compacto
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Relatórios de Eficiência Hospitalar</h1>
        <p>Sistema de análise hospitalar com visualização interativa</p>
    </div>
    """, unsafe_allow_html=True)
    
    # === SIDEBAR - FILTROS ===
    with st.sidebar:
        st.markdown("### 🔧 Filtros")
        
        # Carrega dados
        with st.spinner('Carregando dados...'):
            cnes_disponiveis, competencias_disponiveis, hospitais_info, df_mm = carregar_dados_base()
        
        if not cnes_disponiveis:
            st.error("❌ Dados não disponíveis")
            return
        
        # Seção de estatísticas
        st.markdown("""
        <div class="sidebar-section">
            <h4>📊 Base de Dados</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{len(cnes_disponiveis)}</strong><br>
                <small>Hospitais</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{len(competencias_disponiveis)}</strong><br>
                <small>Competências</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Seção de filtros
        st.markdown("""
        <div class="sidebar-section">
            <h4>🏥 Seleção do Hospital</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Busca compacta
        busca = st.text_input("🔍 Buscar:", placeholder="Nome, município ou CNES")
        
        # Filtrar hospitais
        cnes_filtrados = cnes_disponiveis
        if busca:
            busca_lower = busca.lower()
            cnes_filtrados = [
                cnes for cnes in cnes_disponiveis 
                if (busca_lower in hospitais_info[cnes]['nome'].lower() or
                    busca_lower in hospitais_info[cnes]['municipio'].lower() or
                    busca_lower in hospitais_info[cnes]['uf'].lower() or
                    busca in cnes)
            ]
            
            if not cnes_filtrados:
                st.warning("⚠️ Nenhum resultado")
                cnes_filtrados = cnes_disponiveis[:20]
        
        # Seleção de hospital compacta
        if cnes_filtrados:
            opcoes_display = [hospitais_info[cnes]['display'] for cnes in cnes_filtrados[:30]]
            hospital_selecionado = st.selectbox(
                "Hospital:",
                options=opcoes_display,
                key="hospital_select"
            )
            
            cnes_selecionado = None
            for cnes in cnes_filtrados:
                if hospitais_info[cnes]['display'] == hospital_selecionado:
                    cnes_selecionado = cnes
                    break
        else:
            cnes_selecionado = None
        
        # Seleção de competência compacta
        competencia_selecionada = None
        if cnes_selecionado:
            competencias_cnes = sorted(
                df_mm[df_mm['CNES'] == cnes_selecionado]['COMPETEN'].unique(), 
                reverse=True
            )
            
            if competencias_cnes:
                competencias_display = [formatar_competencia(comp) for comp in competencias_cnes]
                comp_selecionada_display = st.selectbox(
                    "Competência:",
                    options=competencias_display,
                    key="comp_select"
                )
                
                idx = competencias_display.index(comp_selecionada_display)
                competencia_selecionada = competencias_cnes[idx]
            else:
                st.warning("⚠️ Sem competências")
        
        # Info do hospital selecionado
        if cnes_selecionado and competencia_selecionada:
            st.markdown("""
            <div class="sidebar-section">
                <h4>📋 Hospital Selecionado</h4>
            </div>
            """, unsafe_allow_html=True)
            
            info = hospitais_info[cnes_selecionado]
            st.markdown(f"""
            **CNES:** {cnes_selecionado}  
            **Nome:** {info['nome']}  
            **Local:** {info['municipio']}/{info['uf']}  
            **Período:** {formatar_competencia(competencia_selecionada)}
            """)
            
            # Botão de gerar relatório
            if st.button("📊 Gerar Relatório", type="primary", use_container_width=True):
                st.session_state.gerar_relatorio = True
                st.session_state.cnes_atual = cnes_selecionado
                st.session_state.competencia_atual = competencia_selecionada
                st.rerun()
            
            # Botão de exportar (só aparece se relatório foi gerado)
            if st.session_state.get('relatorio_html'):
                st.markdown("---")
                nome_arquivo = f"relatorio_{cnes_selecionado}_{competencia_selecionada}.html"
                
                st.download_button(
                    label="📥 Exportar HTML",
                    data=st.session_state.relatorio_html.encode('utf-8'),
                    file_name=nome_arquivo,
                    mime="text/html",
                    use_container_width=True,
                    help="Download do relatório completo"
                )
        
        # Rodapé da sidebar
        st.markdown("---")
        st.markdown("""
        <small>
        💡 **Dicas:**  
        • Use a busca para filtrar hospitais  
        • Relatórios são gerados em tempo real  
        • Visualização completa na página principal
        </small>
        """, unsafe_allow_html=True)
    
    # === ÁREA PRINCIPAL ===
    
    # Verifica se deve gerar relatório
    if st.session_state.get('gerar_relatorio', False):
        cnes = st.session_state.get('cnes_atual')
        competencia = st.session_state.get('competencia_atual')
        
        if cnes and competencia:
            with st.spinner('🔄 Gerando relatório... Aguarde alguns segundos.'):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text('📊 Carregando dados...')
                progress_bar.progress(25)
                
                status_text.text('📈 Processando análises...')
                progress_bar.progress(50)
                
                status_text.text('🎨 Gerando visualizações...')
                progress_bar.progress(75)
                
                html_relatorio = gerar_relatorio_web(cnes, competencia)
                
                if html_relatorio:
                    status_text.text('✅ Relatório gerado com sucesso!')
                    progress_bar.progress(100)
                    
                    # Armazena no session state
                    st.session_state.relatorio_html = html_relatorio
                    st.session_state.relatorio_cnes = cnes
                    st.session_state.relatorio_competencia = competencia
                    st.session_state.gerar_relatorio = False
                    
                    # Remove progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.rerun()
                else:
                    st.error("❌ Erro ao gerar relatório")
                    st.session_state.gerar_relatorio = False
    
    # Mostra relatório se disponível
    if st.session_state.get('relatorio_html'):
        cnes = st.session_state.get('relatorio_cnes')
        competencia = st.session_state.get('relatorio_competencia')
        
        st.success(f"✅ Relatório gerado para CNES {cnes} - {formatar_competencia(competencia)}")
        
        # Visualização do relatório em tela cheia
        st.markdown("""
        <div class="report-container">
        """, unsafe_allow_html=True)
        
        # Renderiza o HTML do relatório
        components.html(
            st.session_state.relatorio_html,
            height=800,
            scrolling=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # Tela inicial
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 0;">
                <h2>🏥 Sistema de Relatórios</h2>
                <p style="font-size: 1.2em; color: #666;">
                    Selecione um hospital e competência na barra lateral para gerar o relatório de eficiência.
                </p>
                <div style="margin: 2rem 0;">
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                padding: 2rem; border-radius: 10px; margin: 1rem 0;">
                        <h4>📊 Funcionalidades</h4>
                        <ul style="text-align: left; display: inline-block;">
                            <li>✅ Análise de eficiência DEA</li>
                            <li>✅ Evolução temporal (12 meses)</li>
                            <li>✅ Comparação com benchmarks</li>
                            <li>✅ Gráficos interativos</li>
                            <li>✅ Sistema de alertas</li>
                            <li>✅ Análise de procedimentos</li>
                        </ul>
                    </div>
                </div>
                <p style="color: #888;">
                    👈 Use os filtros na barra lateral para começar
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Inicializa session state
    if 'relatorio_html' not in st.session_state:
        st.session_state.relatorio_html = None
    if 'gerar_relatorio' not in st.session_state:
        st.session_state.gerar_relatorio = False
    
    main() 