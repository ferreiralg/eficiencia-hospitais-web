# 🏥 Eficiência Hospitalar - App Web

Sistema web para geração de relatórios de eficiência hospitalar baseado em análise DEA (Data Envelopment Analysis).

## 🚀 Deploy Rápido

[![Deploy no Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

### Deploy no Streamlit Community Cloud:
1. Fork este repositório
2. Vá para [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu GitHub
4. Selecione este repositório
5. App file: `app_web_relatorios_final.py`
6. Deploy automático!

## 💻 Execução Local

```bash
# 1. Clonar repositório
git clone https://github.com/SEU-USUARIO/eficiencia-hospitais-web.git
cd eficiencia-hospitais-web

# 2. Instalar dependências
pip install -r requirements_web.txt

# 3. Executar app
streamlit run app_web_relatorios_final.py
```

## ⚡ **Otimizações Implementadas**

### 🧠 **Cache Inteligente**
- ✅ **Cache TTL**: Dados ficam em cache por 1-3 horas
- ✅ **Session State**: Dados persistem durante a sessão
- ✅ **Carregamento lazy**: Só carrega quando necessário
- ✅ **Configuração cached**: Evita recarregamentos desnecessários

### 📦 **Dependências Completas**
- ✅ **Matplotlib**: Incluído para gráficos
- ✅ **Plotly + Kaleido**: Para gráficos interativos
- ✅ **Scipy + PuLP**: Para análise DEA
- ✅ **Pillow**: Para processamento de imagens
- ✅ **Cache tools**: Para performance otimizada

## 📊 **Funcionalidades**

### 🔍 **Busca Inteligente**
- Pesquisa por nome do hospital
- Filtro por município e UF  
- Busca por código CNES

### 📈 **Relatórios Completos**
- **Padrão**: HTML + arquivos externos (para arquivamento)
- **Incorporado**: Arquivo único autossuficiente (para compartilhamento)

### 📊 **Conteúdo dos Relatórios**
- ✅ Resumo executivo com eficiência DEA
- ✅ Evolução temporal (12 meses)
- ✅ Análise de benchmarks
- ✅ Gráficos de procedimentos
- ✅ Spider chart de alvos
- ✅ Sistema de alertas

## 🛠️ **Estrutura do Projeto**

```
eficiencia-hospitais-web/
├── 🚀 app_web_relatorios_final.py     # App principal (com cache otimizado)
├── 📦 requirements_web.txt            # Dependências completas
├── ⚙️ config.yaml                     # Configuração do sistema
├── 📁 scripts/
│   └── relatorio_cnes.py             # Módulo de relatórios
├── 📁 utils/                          # Utilitários
└── 📁 resultados/                     # Dados (~70MB)
    ├── media_movel/                   # Dados de média móvel
    ├── dea/                          # Resultados DEA
    └── alertas/                      # Sistema de alertas
```

## 🔧 **Resolução de Problemas**

### ❌ **"No module named 'matplotlib'"**
**✅ Solução**: Instalar requirements completo:
```bash
pip install -r requirements_web.txt
```

### ⏳ **App lento no primeiro acesso**
**✅ Solução**: Cache implementado! Dados ficam em cache por 1 hora.

### 📱 **Deploy no Streamlit Cloud**
**✅ Solução**: Requirements otimizado para deploy automático.

## 🌟 **Características Técnicas**

- 🚀 **Performance**: Cache inteligente com TTL
- 📱 **Responsivo**: Interface adaptável  
- 🎯 **Foco**: Apenas arquivos essenciais (~70MB)
- ⚡ **Rápido**: Session state para dados frequentes
- 🔒 **Estável**: Todas as dependências versionadas

## 📄 **Licença**

Este projeto é desenvolvido pelo Tribunal de Contas da União (TCU).

---

**🎯 Pronto para produção** • **📊 Dados reais** • **⚡ Performance otimizada** 