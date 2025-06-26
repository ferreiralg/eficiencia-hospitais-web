#!/usr/bin/env python3
"""
Aplicação Web para Geração de Relatórios de Eficiência Hospitalar - VERSÃO FINAL
Versão simplificada que foca no download e preview básico funcionando.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pathlib
import sys
import tempfile
import os
import base64
import zipfile
from datetime import datetime
import logging

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
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .report-preview {
        background-color: #f8f9fa;
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Importações do projeto
try:
    from scripts.relatorio_cnes import CarregadorConfiguracao, CarregadorDados, GeradorRelatorio
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dados_disponiveis():
    """Carrega dados disponíveis para seleção."""
    try:
        config = CarregadorConfiguracao('config.yaml')
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

@st.cache_data(ttl=1800)  # Cache por 30 minutos
def carregar_configuracao():
    """Carrega configuração com cache."""
    try:
        return CarregadorConfiguracao('config.yaml')
    except Exception as e:
        st.error(f"Erro ao carregar configuração: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dados_base():
    """Carrega dados base com cache otimizado."""
    try:
        config = carregar_configuracao()
        if not config:
            return None, None, None
            
        carregador = CarregadorDados(config)
        
        # Carrega todos os dados base uma vez
        df_mm = carregador.carregar_media_movel()
        df_dea = carregador.carregar_dea()
        df_alertas = carregador.carregar_alertas()
        
        return df_mm, df_dea, df_alertas
    except Exception as e:
        st.error(f"Erro ao carregar dados base: {e}")
        return None, None, None

def formatar_competencia(competencia):
    """Formata competência para exibição."""
    if len(str(competencia)) == 6:
        ano = str(competencia)[:4]
        mes = str(competencia)[4:]
        meses = {
            '01': 'Janeiro', '02': 'Fevereiro', '03': 'Março', '04': 'Abril',
            '05': 'Maio', '06': 'Junho', '07': 'Julho', '08': 'Agosto',
            '09': 'Setembro', '10': 'Outubro', '11': 'Novembro', '12': 'Dezembro'
        }
        return f"{meses.get(mes, mes)}/{ano}"
    return str(competencia)

def criar_pacote_download(html_path):
    """Cria pacote ZIP com todos os arquivos do relatório."""
    try:
        report_dir = pathlib.Path(html_path).parent
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = report_dir / f"relatorio_pacote_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Adicionar arquivo HTML principal
            zipf.write(html_path, 'relatorio_completo.html')
            
            # Adicionar todos os arquivos auxiliares
            for file_path in report_dir.glob('*'):
                if file_path.is_file() and file_path != zip_path:
                    if file_path.suffix in ['.html', '.png', '.css', '.js']:
                        zipf.write(file_path, file_path.name)
        
        return zip_path
    
    except Exception as e:
        st.error(f"Erro ao criar pacote: {e}")
        return None

def extrair_preview_do_relatorio(html_path):
    """Extrai informações básicas do relatório para preview."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extrair informações básicas
        info = {
            'tamanho_arquivo': len(html_content),
            'num_graficos': html_content.count('<iframe'),
            'num_imagens': html_content.count('src="procedimentos'),
            'tem_tabelas': 'table' in html_content,
            'tem_spider': 'spider' in html_content.lower(),
            'tem_evolucao': 'evolucao' in html_content.lower(),
            'tem_benchmarks': 'benchmarks' in html_content.lower(),
        }
        
        return info
    
    except Exception as e:
        st.error(f"Erro ao extrair preview: {e}")
        return None

def gerar_relatorio_web(cnes, competencia):
    """Gera relatório incorporado para visualização na página."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('🔄 Inicializando geração do relatório...')
        progress_bar.progress(10)
        
        # Usa configuração cached
        config = carregar_configuracao()
        if not config:
            st.error("Erro ao carregar configuração")
            return None
            
        gerador = GeradorRelatorio(config)
        
        status_text.text('📊 Carregando dados...')
        progress_bar.progress(30)
        
        status_text.text('📈 Processando dados e gerando visualizações...')
        progress_bar.progress(50)
        
        # Sempre gera relatório embedded para visualização
        status_text.text('📦 Gerando relatório incorporado...')
        progress_bar.progress(70)
        
        html_content = gerador.gerar_relatorio_embedded(cnes, competencia)
        
        status_text.text('✅ Relatório gerado com sucesso!')
        progress_bar.progress(100)
        
        # Informações do relatório
        tamanho_kb = len(html_content.encode('utf-8')) / 1024
        
        return {
            'html_content': html_content,
            'tamanho_kb': tamanho_kb,
            'preview_info': {
                'tamanho_arquivo': len(html_content),
                'num_graficos': html_content.count('plotly-container'),
                'num_imagens': html_content.count('data:image'),
                'tem_tabelas': 'table' in html_content,
                'tem_spider': 'spider' in html_content.lower(),
                'tem_evolucao': 'temporal' in html_content.lower() or 'evolucao' in html_content.lower(),
                'tem_benchmarks': 'benchmarks' in html_content.lower() or 'benchmark' in html_content.lower(),
            }
        }
            
    except Exception as e:
        st.error(f"Erro ao gerar relatório: {e}")
        return None

def main():
    """Função principal da aplicação."""
    
    # Cabeçalho
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Sistema de Relatórios de Eficiência Hospitalar</h1>
        <p>Gere relatórios detalhados de eficiência para hospitais específicos</p>
        <p><strong>📥 Download direto • 🎯 Mesma qualidade do modo local</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Carrega dados disponíveis com cache otimizado
    with st.spinner('🔄 Carregando dados disponíveis...'):
        # Usa cache otimizado - dados ficam em memória
        if 'dados_iniciais_carregados' not in st.session_state:
            cnes_disponiveis, competencias_disponiveis, hospitais_info, df_mm = carregar_dados_disponiveis()
            st.session_state.cnes_disponiveis = cnes_disponiveis
            st.session_state.competencias_disponiveis = competencias_disponiveis  
            st.session_state.hospitais_info = hospitais_info
            st.session_state.df_mm = df_mm
            st.session_state.dados_iniciais_carregados = True
        else:
            # Usa dados do session_state (mais rápido)
            cnes_disponiveis = st.session_state.cnes_disponiveis
            competencias_disponiveis = st.session_state.competencias_disponiveis
            hospitais_info = st.session_state.hospitais_info
            df_mm = st.session_state.df_mm
    
    if not cnes_disponiveis:
        st.error("❌ Não foi possível carregar os dados. Verifique se os arquivos estão disponíveis.")
        return
    
    # Layout em colunas
    col_filtros, col_info = st.columns([3, 1])
    
    with col_filtros:
        st.subheader("🔧 Configurações do Relatório")
        
        # Busca de hospitais
        busca = st.text_input("🔍 Buscar hospital por nome, município ou CNES:")
        
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
                st.warning("⚠️ Nenhum hospital encontrado com o termo de busca.")
                cnes_filtrados = cnes_disponiveis[:10]  # Mostrar primeiros 10
        
        # Seleção de CNES
        if cnes_filtrados:
            opcoes_display = [hospitais_info[cnes]['display'] for cnes in cnes_filtrados[:50]]  # Limitar a 50
            hospital_selecionado = st.selectbox(
                "🏥 Selecionar Hospital:",
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
        
        # Seleção de competência
        competencia_selecionada = None
        if cnes_selecionado:
            competencias_cnes = sorted(
                df_mm[df_mm['CNES'] == cnes_selecionado]['COMPETEN'].unique(), 
                reverse=True
            )
            
            if competencias_cnes:
                competencias_display = [formatar_competencia(comp) for comp in competencias_cnes]
                comp_selecionada_display = st.selectbox(
                    "📅 Selecionar Competência:",
                    options=competencias_display,
                    key="comp_select"
                )
                
                idx = competencias_display.index(comp_selecionada_display)
                competencia_selecionada = competencias_cnes[idx]
            else:
                st.warning("⚠️ Nenhuma competência disponível para este hospital.")
    
    with col_info:
        st.subheader("📊 Estatísticas do Sistema")
        st.metric("🏥 Hospitais", len(cnes_disponiveis))
        st.metric("📅 Competências", len(competencias_disponiveis))
        
        if cnes_selecionado:
            registros_hospital = len(df_mm[df_mm['CNES'] == cnes_selecionado])
            st.metric("📋 Registros do Hospital", registros_hospital)
        
        # Info do conteúdo
        st.markdown("""
        **📊 Conteúdo do Relatório:**
        - ✅ Resumo executivo com eficiência
        - ✅ Evolução temporal (12 meses)
        - ✅ Análise DEA e benchmarks
        - ✅ Gráficos de procedimentos
        - ✅ Spider chart de alvos
        - ✅ Sistema de alertas
        
        **🎯 Funcionalidades:**
        - 🖥️ **Visualização na página**: Veja o relatório diretamente aqui
        - 📥 **Download**: Baixe o arquivo HTML completo
        - 📤 **Compartilhamento**: Arquivo autossuficiente
        """)
    
    # Informações do hospital selecionado
    if cnes_selecionado and competencia_selecionada:
        st.markdown("---")
        
        info = hospitais_info[cnes_selecionado]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4>📋 Hospital Selecionado</h4>
                <p><strong>CNES:</strong> {cnes_selecionado}</p>
                <p><strong>Nome:</strong> {info['nome']}</p>
                <p><strong>Município:</strong> {info['municipio']}</p>
                <p><strong>UF:</strong> {info['uf']}</p>
                <p><strong>Competência:</strong> {formatar_competencia(competencia_selecionada)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Botão de geração
            st.markdown("**📊 Gerar Relatório:**")
            
            if st.button("🚀 Gerar Relatório Completo", type="primary", use_container_width=True, help="Gera relatório para visualização na página"):
                st.session_state.gerar_relatorio = True
                st.session_state.relatorio_gerado = False
        
        # Geração do relatório
        if st.session_state.get('gerar_relatorio', False) and not st.session_state.get('relatorio_gerado', False):
            st.markdown("### 🔄 Gerando Relatório...")
            
            resultados = gerar_relatorio_web(cnes_selecionado, competencia_selecionada)
            
            if resultados:
                st.session_state.resultados_relatorio = resultados
                st.session_state.relatorio_gerado = True
                st.session_state.gerar_relatorio = False
                st.rerun()  # Recarregar para mostrar resultados
            else:
                st.error("❌ Erro ao gerar o relatório. Tente novamente.")
                st.session_state.gerar_relatorio = False
        
        # Mostrar resultados se relatório foi gerado
        if st.session_state.get('relatorio_gerado', False) and 'resultados_relatorio' in st.session_state:
            resultados = st.session_state.resultados_relatorio
            
            st.markdown("""
            <div class="success-box">
                <h3>✅ Relatório Gerado com Sucesso!</h3>
                <p>Relatório completo gerado com todas as funcionalidades do modo local.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Preview das informações
            if resultados['preview_info']:
                info_preview = resultados['preview_info']
                tamanho_info = f"📁 Tamanho: {resultados['tamanho_kb']:.1f} KB"
                
                col_info, col_download = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"""
                    <div class="report-preview">
                        <h4>📄 Informações do Relatório</h4>
                        <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                            <div><strong>📊 Gráficos:</strong> {info_preview['num_graficos']}</div>
                            <div><strong>🖼️ Imagens:</strong> {info_preview['num_imagens']}</div>
                            <div><strong>📋 Tabelas:</strong> {'✅' if info_preview['tem_tabelas'] else '❌'}</div>
                        </div>
                        <div style="display: flex; justify-content: space-around;">
                            <div><strong>🕷️ Spider Chart:</strong> {'✅' if info_preview['tem_spider'] else '❌'}</div>
                            <div><strong>📈 Evolução:</strong> {'✅' if info_preview['tem_evolucao'] else '❌'}</div>
                            <div><strong>🎯 Benchmarks:</strong> {'✅' if info_preview['tem_benchmarks'] else '❌'}</div>
                        </div>
                        <p style="margin-top: 1rem;"><strong>{tamanho_info}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_download:
                    # Botão de download
                    nome_html = f"relatorio_{cnes_selecionado}_{competencia_selecionada}.html"
                    st.download_button(
                        label="📥 Baixar Relatório",
                        data=resultados['html_content'].encode('utf-8'),
                        file_name=nome_html,
                        mime="text/html",
                        use_container_width=True,
                        help="Download do arquivo HTML completo"
                    )
                    
                    # Botão para novo relatório
                    if st.button("🔄 Novo Relatório", use_container_width=True):
                        # Limpar estado
                        for key in ['gerar_relatorio', 'relatorio_gerado', 'resultados_relatorio']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
            
            # Divisor
            st.markdown("---")
            
            # Visualização do relatório na página
            st.markdown("### 📊 Visualização do Relatório")
            
            # Opções de visualização
            col_view_options, col_view_action = st.columns([3, 1])
            
            with col_view_options:
                view_mode = st.radio(
                    "Modo de visualização:",
                    ["🖥️ Visualizar na página", "📱 Visualização compacta"],
                    horizontal=True,
                    help="Escolha como visualizar o relatório"
                )
            
            with col_view_action:
                if st.button("🔄 Atualizar Visualização", use_container_width=True):
                    st.rerun()
            
            # Exibir o relatório
            if view_mode == "🖥️ Visualizar na página":
                # Visualização completa
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h5>📋 Relatório Completo</h5>
                    <p>Visualização completa do relatório com todos os gráficos e tabelas interativas.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Exibir o HTML do relatório
                components.html(resultados['html_content'], height=800, scrolling=True)
                
            else:
                # Visualização compacta
                st.markdown("""
                <div style="background-color: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h5>📱 Visualização Compacta</h5>
                    <p>Visualização otimizada para dispositivos menores ou navegação rápida.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Exibir o HTML do relatório em tamanho menor
                components.html(resultados['html_content'], height=600, scrolling=True)
            
            # Informações de uso
            st.markdown("""
            ---
            ### 📝 Como usar:
            
            - **🖥️ Visualização na página**: Veja o relatório completo diretamente aqui
            - **📥 Download**: Use o botão "Baixar Relatório" para salvar o arquivo
            - **📤 Compartilhamento**: O arquivo baixado é autossuficiente e pode ser enviado por email
            - **🔄 Novo relatório**: Use o botão "Novo Relatório" para gerar outro
            
            > **✅ Garantia**: Relatório **completo e idêntico** ao modo local!
            """)
    
    else:
        # Mensagens de orientação
        if not cnes_selecionado:
            st.info("👆 Selecione um hospital acima para continuar.")
        elif not competencia_selecionada:
            st.info("👆 Selecione uma competência para o hospital escolhido.")

if __name__ == "__main__":
    main() 