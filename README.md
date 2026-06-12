# 💧 Dashboard de Saneamento Básico e Saúde Pública — Espírito Santo

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.17+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.1+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Ativo-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Dashboard analítico interativo para visualização e análise da relação entre saneamento básico e saúde pública nos municípios do Espírito Santo.**

[🚀 Demonstração](#-demonstração) • [📋 Funcionalidades](#-funcionalidades) • [🛠️ Instalação](#️-instalação) • [📖 Documentação](#-documentação) • [🤝 Contribuindo](#-contribuindo)

</div>

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Problema Social](#-problema-social)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação e Configuração](#️-instalação-e-configuração)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Design System](#-design-system)
- [Metodologia](#-metodologia)
- [Documentação](#-documentação)
- [Desenvolvimento](#-desenvolvimento)
- [Contribuindo](#-contribuindo)
- [Autores](#-autores)
- [Licença](#-licença)
- [Agradecimentos](#-agradecimentos)

---

## 🎯 Sobre o Projeto

Este projeto é um **dashboard analítico interativo** desenvolvido para visualizar e analisar a relação entre **saneamento básico** e **saúde pública** nos municípios do Espírito Santo, Brasil.

Utilizando dados do **SNIS** (Sistema Nacional de Informações sobre Saneamento) e **DATASUS** (Departamento de Informática do SUS), o dashboard aplica técnicas de **Machine Learning** (K-Means Clustering) e **análise estatística** para:

- 🎯 Identificar municípios em situação crítica
- 📊 Correlacionar déficits de saneamento com internações hospitalares
- 🗺️ Mapear zonas de vulnerabilidade social
- 💡 Fornecer insights para tomada de decisão baseada em dados

### 🌟 Diferenciais

✅ **Interface moderna e profissional** com modo claro/escuro  
✅ **Acessibilidade WCAG 2.1 AA** garantida em todos os componentes  
✅ **Visualizações interativas** com Plotly  
✅ **Análise estatística robusta** (Spearman, Shapiro-Wilk, Kruskal-Wallis)  
✅ **Machine Learning** para clusterização de municípios  
✅ **Mapa interativo** com Folium  
✅ **Documentação completa** do código e design system  

---

## 🌍 Problema Social

### Por que isso importa?

#### 🏥 Internações Evitáveis
Doenças como **diarreia, cólera, hepatite A** e outras doenças de veiculação hídrica são causadas pela falta de água tratada e esgoto coletado — e são **100% evitáveis** com saneamento adequado.

#### 💸 Custo Social
- Cada **R$ 1 investido em saneamento** poupa **R$ 4 em saúde pública**
- A falta de saneamento onera o sistema de saúde e reduz a produtividade
- Impacto direto na qualidade de vida da população

#### 🎯 Decisão Baseada em Dados
Este dashboard identifica os municípios mais críticos para **priorização de investimentos**, maximizando o impacto social de cada real gasto em saneamento básico.

---

## ✨ Funcionalidades

### 🏠 Visão Geral
- **KPIs principais**: Total de municípios, risco social médio, municípios críticos
- **Distribuição do risco**: Histograma com quartis
- **Ranking de municípios**: Top piores e melhores índices
- **Cobertura de água e esgoto**: Métricas estaduais

### 🗺️ Mapa Interativo
- Visualização geográfica dos municípios do ES
- Círculos coloridos por zona de vulnerabilidade
- Informações detalhadas ao clicar em cada município
- Filtros por ano e zona

### 📈 Correlação
- **Heatmap de Spearman**: Correlação entre variáveis de saneamento e saúde
- Análise da relação entre déficits e internações
- Seleção customizável de variáveis
- Interpretação guiada dos resultados

### 🔬 Análise Estatística
- **Teste de Shapiro-Wilk**: Normalidade dos dados
- **Teste de Kruskal-Wallis**: Diferenças entre zonas
- **Boxplots comparativos**: Distribuição por zona
- Significância estatística destacada

### 🤖 Clusterização
- **Visualização 3D** dos clusters K-Means
- Perfil detalhado de cada zona de vulnerabilidade
- Métricas médias por cluster
- Validação com Silhouette Score

### 🔍 Perfil do Município
- Ficha completa de qualquer município
- **Gauge de risco** visual e intuitivo
- Histórico temporal de indicadores
- Comparação com média estadual
- Gráfico radar multidimensional

### 📊 Análises Avançadas
- Ranking completo de municípios
- Linha do tempo de indicadores
- Análise de investimento vs. risco
- Tendências estaduais

---

## 🛠️ Tecnologias Utilizadas

### Core
- **[Python 3.11+](https://www.python.org/)**: Linguagem principal
- **[Streamlit 1.28+](https://streamlit.io/)**: Framework web para dashboards
- **[Pandas 2.1+](https://pandas.pydata.org/)**: Manipulação de dados
- **[NumPy 1.24+](https://numpy.org/)**: Computação numérica

### Visualização
- **[Plotly 5.17+](https://plotly.com/python/)**: Gráficos interativos
- **[Folium](https://python-visualization.github.io/folium/)**: Mapas interativos
- **[Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium)**: Integração Folium + Streamlit

### Machine Learning & Estatística
- **[Scikit-learn 1.3+](https://scikit-learn.org/)**: K-Means, StandardScaler
- **[SciPy 1.11+](https://scipy.org/)**: Testes estatísticos (Spearman, Shapiro-Wilk, Kruskal-Wallis)

### Design & UX
- **[Google Fonts - Inter](https://fonts.google.com/specimen/Inter)**: Tipografia
- **Design System customizado**: Tokens, componentes, tema claro/escuro
- **WCAG 2.1 AA**: Acessibilidade garantida

### Dados
- **[SNIS](http://www.snis.gov.br/)**: Indicadores de saneamento (2006-2023)
- **[DATASUS/TabNet](https://datasus.saude.gov.br/)**: Internações por doenças de veiculação hídrica
- **[IBGE](https://www.ibge.gov.br/)**: Dados populacionais

---

## 🚀 Instalação e Configuração

### Pré-requisitos

- **Python 3.11** ou superior
- **pip** (gerenciador de pacotes Python)
- **Git** (opcional, para clonar o repositório)

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/analise-saneamento-es.git
cd analise-saneamento-es
```

### 2. Crie um Ambiente Virtual (Recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

#### Dependências Principais

```txt
streamlit>=1.28.0
pandas>=2.1.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
scipy>=1.11.0
folium>=0.14.0
streamlit-folium>=0.15.0
pyarrow>=14.0.0
```

### 4. Execute o Dashboard

```bash
cd dashboard
streamlit run app.py
```

O dashboard será aberto automaticamente no navegador em `http://localhost:8501`

### 5. (Opcional) Limpar Cache

Se encontrar problemas após atualizações:

```bash
# Limpar cache Python
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Limpar cache Streamlit
streamlit cache clear
```

---

## 📖 Como Usar

### Interface Principal

1. **Home (app.py)**
   - Visão geral do projeto
   - Seções disponíveis
   - Contexto e importância social

2. **Sidebar (Barra Lateral)**
   - **🎨 Tema**: Alternar entre claro/escuro
   - **Navegação**: Acesse as páginas do menu

### Navegação por Páginas

#### 🏠 Visão Geral
1. Selecione o ano desejado
2. Visualize os KPIs principais
3. Analise a distribuição do risco
4. Confira o ranking de municípios

#### 🗺️ Mapa Interativo
1. Escolha o ano
2. Filtre por zonas de risco
3. Clique nos círculos para ver detalhes
4. Use zoom e pan para explorar

#### 📈 Correlação
1. Selecione as variáveis de interesse
2. Analise o heatmap de correlação
3. Identifique relações fortes/fracas
4. Use tooltips para valores exatos

#### 🔬 Análise Estatística
1. Escolha o ano
2. Veja resultados dos testes estatísticos
3. Compare boxplots por zona
4. Entenda a significância

#### 🤖 Clusterização
1. Visualize os clusters 3D
2. Rotacione o gráfico para explorar
3. Analise perfil de cada zona
4. Compare métricas médias

#### 🔍 Perfil do Município
1. Selecione um município
2. Veja gauge de risco social
3. Analise histórico temporal
4. Compare com média estadual

#### 📊 Análises Avançadas
1. Explore ranking completo
2. Visualize tendências temporais
3. Analise investimento vs. risco
4. Identifique padrões estaduais

### Modo Claro vs. Escuro

- **Modo Claro**: Ideal para ambientes bem iluminados
- **Modo Escuro**: Confortável para uso prolongado, reduz fadiga visual

Alterne usando o radio button **🎨 Tema** na sidebar.

---

## 📁 Estrutura do Projeto

```
analise-saneamento-es/
│
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências Python
├── .gitignore                          # Arquivos ignorados pelo Git
│
├── data/                               # Dados brutos e processados
│   ├── saneamento_es_zonas.parquet    # Dataset principal (zonas)
│   └── saneamento_diamante.parquet    # Dataset para análises detalhadas
│
├── notebooks/                          # Jupyter Notebooks (exploração)
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_clusterizacao.ipynb
│   └── 03_analise_estatistica.ipynb
│
└── dashboard/                          # Aplicação Streamlit
    │
    ├── app.py                          # 🏠 Página principal (Home)
    │
    ├── .streamlit/                     # Configurações Streamlit
    │   └── config.toml                 # Tema base, servidor, etc.
    │
    ├── pages/                          # Páginas do dashboard
    │   ├── 1_visao_geral.py           # 🏠 Visão Geral
    │   ├── 2_mapa_interativo.py       # 🗺️ Mapa Interativo
    │   ├── 3_correlacao.py            # 📈 Correlação
    │   ├── 4_analise_estatistica.py   # 🔬 Análise Estatística
    │   ├── 5_clusterizacao.py         # 🤖 Clusterização
    │   ├── 6_perfil_municipio.py      # 🔍 Perfil do Município
    │   └── 7_analises_avancadas.py    # 📊 Análises Avançadas
    │
    ├── utils/                          # Módulos auxiliares
    │   ├── __init__.py
    │   ├── theme.py                    # ⭐ Sistema de tokens e tema
    │   ├── components.py               # ⭐ Componentes UI reutilizáveis
    │   ├── charts.py                   # ⭐ Gráficos Plotly
    │   └── data_loader.py              # Carregamento de dados
    │
    └── docs/                           # Documentação
        ├── DESIGN_SYSTEM.md            # Sistema de design completo
        ├── CHANGELOG_UI.md             # Histórico de mudanças UI/UX
        ├── TROUBLESHOOTING.md          # Guia de solução de problemas
        └── README_DESIGN.md            # Resumo do design system
```

---

## 🎨 Design System

### Princípios

1. **Acessibilidade First**: WCAG 2.1 Level AA
2. **Consistência**: Mesma identidade visual em toda aplicação
3. **Legibilidade**: Tipografia otimizada (Inter)
4. **Feedback Visual**: Estados interativos claros
5. **Modo Claro/Escuro**: Suporte completo

### Paleta de Cores

#### Modo Claro
```css
Background:  #F8FAFC
Surface:     #FFFFFF
Text:        #0F172A (contraste 15.84:1)
Primary:     #2563EB
Success:     #16A34A
Warning:     #D97706
Danger:      #DC2626
```

#### Modo Escuro
```css
Background:  #0F172A
Surface:     rgba(30,41,59,0.85)
Text:        #F8FAFC (contraste 15.84:1)
Primary:     #3B82F6
Success:     #22C55E
Warning:     #F59E0B
Danger:      #EF4444
```

### Componentes

- **KPI Cards**: Métricas destacadas com hover effects
- **Badges de Zona**: Coloridos por nível de risco
- **Cabeçalhos**: Contexto visual para cada página
- **Caixas de Explicação**: Guias inline por tipo
- **Gráficos Plotly**: Tema consistente e interativo

### Tipografia

- **Família**: Inter (Google Fonts)
- **Escala**: 0.8rem → 2rem
- **Pesos**: 400 (regular), 600 (semibold), 700 (bold), 800 (extrabold)

📚 **Documentação completa**: [DESIGN_SYSTEM.md](dashboard/DESIGN_SYSTEM.md)

---

## 🔬 Metodologia

### Fontes de Dados

#### SNIS (Sistema Nacional de Informações sobre Saneamento)
- Índice de atendimento de água
- Índice de atendimento de esgoto
- Índice de tratamento de esgoto
- Índice de perdas na distribuição
- Investimentos em saneamento
- **Período**: 2006 - 2023

#### DATASUS/TabNet
- Internações por doenças de veiculação hídrica:
  - Diarreia e gastroenterite
  - Cólera
  - Hepatite A
  - Febre tifoide
- Taxa de morbidade por 100 mil habitantes
- **Período**: 2006 - 2023

#### IBGE
- População dos municípios
- Dados demográficos

### Pipeline de Análise

#### 1. Pré-processamento
```
Dados Brutos → Limpeza → Interpolação → Normalização → Dataset Final
```

- Tratamento de valores ausentes
- Interpolação linear para séries temporais
- Normalização com StandardScaler (para clustering)

#### 2. Cálculo do Índice de Risco Social
```python
RISCO_SOCIAL_FINAL = (
    0.25 × deficit_agua_normalizado +
    0.30 × deficit_esgoto_normalizado +
    0.15 × vazio_sanitario_normalizado +
    0.30 × taxa_morbidade_normalizada
)
```

**Escala**: 0 (baixo risco) → 100 (risco crítico)

#### 3. Clusterização (K-Means)
- **Algoritmo**: K-Means (k=4)
- **Features**: deficit_agua, deficit_esgoto, vazio_sanitario, taxa_morbidade
- **Validação**: Silhouette Score, Método do Cotovelo
- **Resultado**: 4 zonas de vulnerabilidade

#### 4. Zonas de Vulnerabilidade

| Zona | Risco Social | Cor | Características |
|------|--------------|-----|-----------------|
| 🟢 **Verde** | 0 - 25 | `#22C55E` | Baixo risco, bom saneamento |
| 🟡 **Amarela** | 25 - 50 | `#EAB308` | Risco moderado, atenção necessária |
| 🟠 **Laranja** | 50 - 75 | `#F97316` | Risco elevado, déficits significativos |
| 🔴 **Vermelha** | 75 - 100 | `#EF4444` | Risco crítico, situação emergencial |

#### 5. Testes Estatísticos

- **Shapiro-Wilk**: Testa normalidade dos dados (H₀: dados são normais)
- **Spearman**: Correlação não-paramétrica (mede relações monotônicas)
- **Kruskal-Wallis**: Testa diferenças entre zonas (H₀: médias iguais)

---

## 📚 Documentação

### Para Usuários
- **[README.md](README.md)** (este arquivo): Visão geral e uso

### Para Desenvolvedores
- **[DESIGN_SYSTEM.md](dashboard/DESIGN_SYSTEM.md)**: Sistema de design completo
- **[CHANGELOG_UI.md](dashboard/CHANGELOG_UI.md)**: Histórico de mudanças
- **[TROUBLESHOOTING.md](dashboard/TROUBLESHOOTING.md)**: Solução de problemas
- **[README_DESIGN.md](dashboard/README_DESIGN.md)**: Resumo executivo do design

### Docstrings
Todos os módulos, classes e funções possuem docstrings detalhadas.

```python
def grafico_distribuicao_risco(df: pd.DataFrame, ano: int | None = None) -> go.Figure:
    """
    Histograma + KDE da distribuição do Índice de Risco Social.
    Mostra quão concentrados estão os municípios em cada faixa de risco.
    
    Args:
        df: DataFrame com coluna RISCO_SOCIAL_FINAL
        ano: Ano para filtrar (opcional)
    
    Returns:
        go.Figure: Gráfico Plotly configurado
    """
```

---

## 🛠️ Desenvolvimento

### Configurar Ambiente de Desenvolvimento

```bash
# 1. Clone e entre no diretório
git clone https://github.com/seu-usuario/analise-saneamento-es.git
cd analise-saneamento-es

# 2. Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Instale dependências + dev tools
pip install -r requirements.txt
pip install black flake8 mypy pytest jupyter

# 4. Execute em modo dev
cd dashboard
streamlit run app.py --server.runOnSave true
```

### Estrutura de uma Página

```python
"""
pages/X_nome_pagina.py
Descrição breve da página.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.theme import apply_theme
from utils.components import cabecalho_pagina
from utils.data_loader import carregar_dados_zonas

# Configuração
st.set_page_config(
    page_title="Nome da Página | Saneamento ES",
    page_icon="🎯",
    layout="wide",
)

# Aplicar tema (obrigatório)
apply_theme(show_toggle=True)

# Carregar dados
df = carregar_dados_zonas()

# Cabeçalho
cabecalho_pagina(
    titulo="Título da Página",
    descricao="Descrição do que a página faz...",
    icone="🎯",
)

# Conteúdo da página
st.markdown("### Seção Principal")
# ... resto do código
```

### Adicionar Novo Componente

```python
# Em utils/components.py

def novo_componente(param1: str, param2: int) -> None:
    """
    Descrição do componente.
    
    Args:
        param1: Descrição do parâmetro 1
        param2: Descrição do parâmetro 2
    """
    tokens = _get_tokens()
    
    st.markdown(f"""
    <div style="
        background: {tokens['surface']};
        border: 1px solid {tokens['surface_border']};
        padding: 1rem;
    ">
        {param1} - {param2}
    </div>
    """, unsafe_allow_html=True)
```

### Adicionar Novo Gráfico

```python
# Em utils/charts.py

def grafico_novo(df: pd.DataFrame, coluna: str) -> go.Figure:
    """
    Descrição do gráfico.
    
    Args:
        df: DataFrame com os dados
        coluna: Nome da coluna a plotar
    
    Returns:
        go.Figure: Gráfico Plotly configurado
    """
    tokens = _get_tokens()
    
    fig = go.Figure()
    
    # Adicionar traces
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[coluna],
        mode='lines',
        line=dict(color=tokens['primary'], width=2),
    ))
    
    # Aplicar layout
    _aplicar_layout(fig, titulo="Título do Gráfico")
    
    return fig
```

### Boas Práticas

✅ **Sempre** chame `apply_theme()` no início de cada página  
✅ Use componentes de `utils/components.py` em vez de HTML direto  
✅ Use `_aplicar_layout()` para todos os gráficos Plotly  
✅ Adicione docstrings em todas as funções  
✅ Use type hints (`str`, `int`, `pd.DataFrame`, etc.)  
✅ Mantenha funções pequenas e focadas (< 50 linhas)  
✅ Cache dados com `@st.cache_data`  
✅ Teste em ambos os modos (claro e escuro)  

---

## 🤝 Contribuindo

Contribuições são **muito bem-vindas**! Este projeto é open-source e qualquer ajuda é apreciada.

### Como Contribuir

1. **Fork** o repositório
2. **Clone** seu fork: `git clone https://github.com/seu-usuario/analise-saneamento-es.git`
3. Crie uma **branch** para sua feature: `git checkout -b feature/minha-feature`
4. **Commit** suas mudanças: `git commit -m 'feat: adiciona nova análise'`
5. **Push** para a branch: `git push origin feature/minha-feature`
6. Abra um **Pull Request**

### Tipos de Contribuição

- 🐛 **Reportar bugs**: Abra uma issue descrevendo o problema
- 💡 **Sugerir features**: Compartilhe ideias de melhorias
- 📝 **Melhorar documentação**: Corrija erros ou adicione exemplos
- 🎨 **Design**: Melhore UI/UX ou componentes visuais
- 🔬 **Análises**: Adicione novos gráficos ou análises estatísticas
- 🧪 **Testes**: Adicione testes unitários ou de integração

### Padrões de Commit

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: adiciona nova página de análise temporal
fix: corrige contraste no modo escuro
docs: atualiza README com instruções de instalação
style: formata código com black
refactor: reorganiza estrutura de componentes
test: adiciona testes para data_loader
chore: atualiza dependências
```

### Reportar Problemas

Ao abrir uma **issue**, inclua:
- ✅ Descrição clara do problema
- ✅ Passos para reproduzir
- ✅ Comportamento esperado vs. atual
- ✅ Screenshots (se aplicável)
- ✅ Versão Python, SO, navegador
- ✅ Mensagens de erro completas

---

## 👥 Autores

### Desenvolvimento Principal
- **[Seu Nome]** - Desenvolvimento full-stack, análise de dados, ML

### Design System & UI/UX
- **[Nome do Designer]** - Design system, acessibilidade, componentes

### Análise de Dados
- **[Nome do Analista]** - ETL, análise estatística, modelagem

### Contribuidores
Veja a lista completa de [contribuidores](https://github.com/seu-usuario/analise-saneamento-es/contributors) que participaram deste projeto.

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

```
MIT License

Copyright (c) 2024 [Seu Nome]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[...]
```

---

## 🙏 Agradecimentos

- **[SNIS](http://www.snis.gov.br/)** pelos dados de saneamento
- **[DATASUS](https://datasus.saude.gov.br/)** pelos dados de saúde pública
- **[IBGE](https://www.ibge.gov.br/)** pelos dados demográficos
- **[Streamlit](https://streamlit.io/)** pela framework incrível
- **[Plotly](https://plotly.com/)** pelas visualizações interativas
- **Comunidade open-source** pelas ferramentas e bibliotecas

---

## 📞 Contato

### Issues e Suporte
- 🐛 Bugs: [GitHub Issues](https://github.com/seu-usuario/analise-saneamento-es/issues)
- 💡 Features: [GitHub Discussions](https://github.com/seu-usuario/analise-saneamento-es/discussions)

### Redes Sociais
- 📧 Email: seu.email@exemplo.com
- 💼 LinkedIn: [seu-perfil](https://linkedin.com/in/seu-perfil)
- 🐙 GitHub: [@seu-usuario](https://github.com/seu-usuario)

---

## 📊 Status do Projeto

![GitHub last commit](https://img.shields.io/github/last-commit/seu-usuario/analise-saneamento-es?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/seu-usuario/analise-saneamento-es?style=flat-square)
![GitHub pull requests](https://img.shields.io/github/issues-pr/seu-usuario/analise-saneamento-es?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/seu-usuario/analise-saneamento-es?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/seu-usuario/analise-saneamento-es?style=flat-square)

---

## 🗺️ Roadmap

### v1.0 (Atual) ✅
- [x] Dashboard básico com 7 páginas
- [x] Design system completo
- [x] Modo claro/escuro
- [x] Análises estatísticas
- [x] Machine Learning (K-Means)
- [x] Documentação completa

### v1.1 (Planejado) 🔄
- [ ] Testes unitários (pytest)
- [ ] CI/CD (GitHub Actions)
- [ ] Docker containerization
- [ ] Deploy em cloud (Streamlit Cloud / Heroku)

### v2.0 (Futuro) 🚀
- [ ] API REST para dados
- [ ] Exportar relatórios PDF
- [ ] Comparação entre estados
- [ ] Predição de tendências (Time Series)
- [ ] Mobile-first responsive design
- [ ] Multi-idioma (PT/EN/ES)

---

<div align="center">

### ⭐ Se este projeto foi útil, considere dar uma estrela!

**Desenvolvido com 💧 e 📊 para um Espírito Santo mais saudável**

[⬆ Voltar ao topo](#-dashboard-de-saneamento-básico-e-saúde-pública--espírito-santo)

</div>
