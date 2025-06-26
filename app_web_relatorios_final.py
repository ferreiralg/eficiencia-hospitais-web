#!/usr/bin/env python3
"""
Aplicação Web para Geração de Relatórios de Eficiência Hospitalar - VERSÃO FINAL
Versão simplificada que foca no download e preview básico funcionando.
"""

import streamlit as st
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

@st.cache_data
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

def gerar_relatorio_web(cnes, competencia, tipo='padrao'):
    """Gera relatório e retorna informações do arquivo."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('🔄 Inicializando geração do relatório...')
        progress_bar.progress(10)
        
        config = CarregadorConfiguracao('config.yaml')
        gerador = GeradorRelatorio(config)
        
        status_text.text('📊 Carregando dados...')
        progress_bar.progress(30)
        
        status_text.text('📈 Processando dados e gerando visualizações...')
        progress_bar.progress(50)
        
        if tipo == 'embedded':
            # Gera relatório embedded (tudo incorporado)
            status_text.text('📦 Gerando relatório incorporado...')
            html_content = gerador.gerar_relatorio_embedded(cnes, competencia)
            
            status_text.text('✅ Relatório incorporado gerado com sucesso!')
            progress_bar.progress(100)
            
            # Informações do relatório embedded
            tamanho_kb = len(html_content.encode('utf-8')) / 1024
            
            return {
                'html_content': html_content,
                'tipo': 'embedded',
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
        else:
            # Gera relatório padrão (com arquivos externos)
            arquivo_relatorio = gerador.gerar_relatorio_completo(cnes, competencia)
            
            status_text.text('📦 Preparando arquivos para download...')
            progress_bar.progress(80)
            
            if arquivo_relatorio and os.path.exists(arquivo_relatorio):
                # Criar pacote ZIP
                zip_path = criar_pacote_download(arquivo_relatorio)
                
                # Extrair informações para preview
                preview_info = extrair_preview_do_relatorio(arquivo_relatorio)
                
                status_text.text('✅ Relatório padrão gerado com sucesso!')
                progress_bar.progress(100)
                
                return {
                    'html_path': arquivo_relatorio,
                    'zip_path': zip_path,
                    'tipo': 'padrao',
                    'preview_info': preview_info
                }
            else:
                status_text.text('❌ Erro: arquivo não foi criado')
                return None
            
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
    
    # Carrega dados disponíveis
    with st.spinner('🔄 Carregando dados disponíveis...'):
        cnes_disponiveis, competencias_disponiveis, hospitais_info, df_mm = carregar_dados_disponiveis()
    
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
        
        **🎯 Tipos Disponíveis:**
        - 📄 **Padrão**: HTML + arquivos externos
        - 📦 **Incorporado**: Tudo em um arquivo
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
            # Botões de geração
            st.markdown("**📊 Tipo de Relatório:**")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("📄 Padrão", type="secondary", use_container_width=True, help="Relatório com arquivos externos"):
                    st.session_state.gerar_relatorio = True
                    st.session_state.tipo_relatorio = 'padrao'
                    st.session_state.relatorio_gerado = False
            
            with col_btn2:
                if st.button("📦 Incorporado", type="primary", use_container_width=True, help="Tudo em um arquivo"):
                    st.session_state.gerar_relatorio = True
                    st.session_state.tipo_relatorio = 'embedded'
                    st.session_state.relatorio_gerado = False
        
        # Geração do relatório
        if st.session_state.get('gerar_relatorio', False) and not st.session_state.get('relatorio_gerado', False):
            tipo_relatorio = st.session_state.get('tipo_relatorio', 'padrao')
            tipo_nome = "Incorporado" if tipo_relatorio == 'embedded' else "Padrão"
            
            st.markdown(f"### 🔄 Gerando Relatório {tipo_nome}...")
            
            resultados = gerar_relatorio_web(cnes_selecionado, competencia_selecionada, tipo_relatorio)
            
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
            tipo_relatorio = resultados.get('tipo', 'padrao')
            tipo_nome = "Incorporado" if tipo_relatorio == 'embedded' else "Padrão"
            
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Relatório {tipo_nome} Gerado com Sucesso!</h3>
                <p>Relatório completo gerado com todas as funcionalidades do modo local.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Preview das informações
            if resultados['preview_info']:
                info_preview = resultados['preview_info']
                
                if tipo_relatorio == 'embedded':
                    tamanho_info = f"📁 Tamanho: {resultados['tamanho_kb']:.1f} KB (arquivo único)"
                else:
                    tamanho_info = f"📁 Tamanho: {info_preview['tamanho_arquivo']:,} caracteres"
                
                st.markdown(f"""
                <div class="report-preview">
                    <h4>📄 Informações do Relatório {tipo_nome}</h4>
                    <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                        <div><strong>📊 Gráficos Interativos:</strong> {info_preview['num_graficos']}</div>
                        <div><strong>🖼️ Imagens:</strong> {info_preview['num_imagens']}</div>
                        <div><strong>📋 Tabelas:</strong> {'✅' if info_preview['tem_tabelas'] else '❌'}</div>
                    </div>
                    <div style="display: flex; justify-content: space-around;">
                        <div><strong>🕷️ Spider Chart:</strong> {'✅' if info_preview['tem_spider'] else '❌'}</div>
                        <div><strong>📈 Evolução:</strong> {'✅' if info_preview['tem_evolucao'] else '❌'}</div>
                        <div><strong>🎯 Benchmarks:</strong> {'✅' if info_preview['tem_benchmarks'] else '❌'}</div>
                    </div>
                    <p style="margin-top: 1rem;"><strong>{tamanho_info}</strong></p>
                    <p><strong>🔄 Tipo:</strong> {'📦 Autossuficiente (sem dependências)' if tipo_relatorio == 'embedded' else '📄 Padrão (com arquivos externos)'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Botões de download
            if tipo_relatorio == 'embedded':
                # Relatório incorporado - apenas um download
                col_down1, col_down2 = st.columns([2, 1])
                
                with col_down1:
                    # Download HTML incorporado
                    nome_html = f"relatorio_incorporado_{cnes_selecionado}_{competencia_selecionada}.html"
                    st.download_button(
                        label="📦 Baixar Relatório Incorporado",
                        data=resultados['html_content'].encode('utf-8'),
                        file_name=nome_html,
                        mime="text/html",
                        use_container_width=True,
                        help="Arquivo HTML único com todas as visualizações incorporadas"
                    )
                
                with col_down2:
                    # Botão para novo relatório
                    if st.button("🔄 Gerar Novo Relatório", use_container_width=True):
                        # Limpar estado
                        for key in ['gerar_relatorio', 'relatorio_gerado', 'resultados_relatorio', 'tipo_relatorio']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                
                st.markdown("""
                <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h5>📦 Vantagens do Relatório Incorporado:</h5>
                    <ul>
                        <li>✅ <strong>Arquivo único</strong> - sem dependências externas</li>
                        <li>✅ <strong>Autossuficiente</strong> - todas as imagens e gráficos incluídos</li>
                        <li>✅ <strong>Ideal para compartilhamento</strong> - envie por email facilmente</li>
                        <li>✅ <strong>Arquivamento permanente</strong> - não perde componentes</li>
                        <li>✅ <strong>Menor tamanho total</strong> - compactado e otimizado</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                # Relatório padrão - múltiplos downloads
                col_down1, col_down2, col_down3 = st.columns(3)
                
                with col_down1:
                    # Download HTML principal
                    with open(resultados['html_path'], 'rb') as f:
                        html_bytes = f.read()
                    
                    nome_html = f"relatorio_{cnes_selecionado}_{competencia_selecionada}.html"
                    st.download_button(
                        label="📄 Baixar HTML Principal",
                        data=html_bytes,
                        file_name=nome_html,
                        mime="text/html",
                        use_container_width=True,
                        help="Arquivo HTML principal do relatório"
                    )
                
                with col_down2:
                    # Download ZIP completo
                    if resultados['zip_path'] and os.path.exists(resultados['zip_path']):
                        with open(resultados['zip_path'], 'rb') as f:
                            zip_bytes = f.read()
                        
                        nome_zip = f"relatorio_completo_{cnes_selecionado}_{competencia_selecionada}.zip"
                        st.download_button(
                            label="📦 Baixar Pacote ZIP",
                            data=zip_bytes,
                            file_name=nome_zip,
                            mime="application/zip",
                            use_container_width=True,
                            help="Pacote completo com todos os arquivos"
                        )
                
                with col_down3:
                    # Botão para novo relatório
                    if st.button("🔄 Gerar Novo Relatório", use_container_width=True):
                        # Limpar estado
                        for key in ['gerar_relatorio', 'relatorio_gerado', 'resultados_relatorio', 'tipo_relatorio']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
            
            # Instruções baseadas no tipo
            if tipo_relatorio == 'embedded':
                st.markdown("""
                ---
                ### 📝 Instruções de Uso - Relatório Incorporado:
                
                1. **📦 Baixar arquivo único**: Clique no botão acima para baixar
                2. **🌐 Abrir no navegador**: Clique duas vezes no arquivo baixado
                3. **📤 Compartilhar facilmente**: Envie o arquivo por email ou mensagem
                4. **📁 Arquivar permanentemente**: Não depende de outros arquivos
                
                > **✅ Garantia**: Relatório **completo e autossuficiente** - idêntico ao modo local!
                """)
            else:
                st.markdown("""
                ---
                ### 📝 Instruções de Uso - Relatório Padrão:
                
                1. **📄 HTML Principal**: Baixe e abra no navegador para visualização completa
                2. **📦 Pacote ZIP**: Contém todos os arquivos (ideal para arquivamento)
                3. **🌐 Melhor experiência**: Abra o arquivo HTML baixado no seu navegador preferido
                
                > **✅ Garantia**: Os relatórios gerados são **idênticos** aos criados pelo modo local!
                """)
    
    else:
        # Mensagens de orientação
        if not cnes_selecionado:
            st.info("👆 Selecione um hospital acima para continuar.")
        elif not competencia_selecionada:
            st.info("👆 Selecione uma competência para o hospital escolhido.")

if __name__ == "__main__":
    main() 