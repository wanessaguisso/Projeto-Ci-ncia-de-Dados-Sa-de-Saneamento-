"""
app.py — Ponto de entrada do Dashboard de Saneamento do ES
Execute com: streamlit run app.py
"""

import sys
from pathlib import Path

# Garante que o diretório do dashboard esteja no sys.path
DASHBOARD_DIR = Path(__file__).parent
sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st
import streamlit.components.v1 as components
from utils.theme import apply_theme, get_theme_mode

# ─── Configuração Global ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saneamento e Saúde no ES — Dashboard Analítico",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Dashboard desenvolvido para análise do saneamento básico no Espírito Santo. "
            "Combina dados do SNIS, DATASUS e algoritmos de Machine Learning para "
            "identificar e classificar zonas de vulnerabilidade social."
        )
    },
)

# ─── Tema Global ─────────────────────────────────────────────────────────────
from utils.theme import THEME_LIGHT, THEME_DARK

apply_theme(show_toggle=True)
is_dark = get_theme_mode() == "Escuro"
tokens = THEME_DARK if is_dark else THEME_LIGHT

home_panel_bg = (
    "linear-gradient(135deg, rgba(30,41,59,0.85) 0%, rgba(15,23,42,0.8) 50%, rgba(30,41,59,0.85) 100%)"
    if is_dark
    else "linear-gradient(135deg, rgba(219,234,254,0.7) 0%, rgba(255,255,255,0.95) 50%, rgba(219,234,254,0.7) 100%)"
)
home_panel_border = tokens['surface_border']
title_color = tokens['title']
subtitle_color = tokens['text']
body_color = tokens['muted']
card_bg = tokens['surface']
card_border = tokens['surface_border']
card_shadow = tokens['surface_shadow']

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem;">
        <div style="font-size:3rem;">💧</div>
        <div style="
            font-family:'Inter',sans-serif;
            font-size:1.1rem;
            font-weight:700;
            color:""" + title_color + """;
            line-height:1.3;
        ">Saneamento ES</div>
        <div style="color:""" + subtitle_color + """; font-size:0.78rem; margin-top:4px;">
            Dashboard Analítico
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(148,163,184,0.1);margin:0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="color:""" + body_color + """; font-size:0.8rem; padding:0 0.5rem 0.5rem;">
        <b style="color:#3b82f6;">📌 Navegação</b><br>
        Use o menu lateral para acessar as seções do dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(148,163,184,0.1);margin:1rem 0 0.5rem;">
    <div style="color:#475569; font-size:0.72rem; text-align:center; padding:0.5rem;">
        Dados: SNIS + DATASUS<br>
        Algoritmo: K-Means Clustering<br>
        <span style="color:#334155;">Espírito Santo, Brasil</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Página Principal (Home) ──────────────────────────────────────────────────
st.markdown("""
<div style="
    background: """ + home_panel_bg + """;
    border: 1px solid """ + home_panel_border + """;
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
">
    <div style="font-size:4rem; margin-bottom:1rem;">💧</div>
    <h1 style="
        font-family:'Inter',sans-serif;
        font-size:2.4rem;
        font-weight:800;
        color:""" + title_color + """;
        margin:0 0 1rem 0;
        line-height:1.2;
    ">Saneamento Básico e Saúde Pública</h1>
    <h2 style="
        font-family:'Inter',sans-serif;
        font-size:1.2rem;
        font-weight:400;
        color:""" + subtitle_color + """;
        margin:0 0 1.5rem 0;
    ">Dashboard Analítico — Espírito Santo, Brasil</h2>
    <p style="
        color:""" + body_color + """;
        max-width:600px;
        margin:0 auto;
        line-height:1.8;
        font-size:0.95rem;
    ">
        Transformamos dados brutos do SNIS e DATASUS em inteligência acionável.
        Este dashboard revela como a falta de saneamento impacta diretamente a saúde
        da população e classifica os municípios por nível de risco social.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Cards de Seções ──────────────────────────────────────────────────────────
st.markdown(f"<h3 style='color:{title_color}; margin-top:2rem; margin-bottom:1rem;'>🗺️ O que você encontra aqui</h3>", unsafe_allow_html=True)

secoes = [
    ("🏠", "Visão Geral", "KPIs principais, distribuição do risco social e ranking de municípios.", tokens['primary']),
    ("🗺️", "Mapa Interativo", "Mapa geográfico do ES com municípios coloridos por zona de vulnerabilidade.", tokens['success']),
    ("📈", "Correlação", "Heatmap de Spearman e análise da relação entre saneamento e internações.", "#8B5CF6"),
    ("🔬", "Análise Estatística", "Testes de normalidade (Shapiro-Wilk) e hipótese (Kruskal-Wallis) com boxplots.", tokens['warning']),
    ("🤖", "Clusterização", "Visualização dos clusters K-Means e perfil de cada zona de vulnerabilidade.", tokens['danger']),
    ("🔍", "Perfil do Município", "Ficha completa de qualquer município: histórico, gauge de risco e comparações.", tokens['info']),
    ("📊", "Análises Avançadas", "Ranking, linha do tempo, investimento × risco e tendências estaduais.", "#F97316"),
]

cols_row1 = st.columns(4)
cols_row2 = st.columns(3)
cols = cols_row1 + cols_row2

for (icone, titulo, descricao, cor), col in zip(secoes, cols):
    with col:
        st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 1px solid {card_border};
            border-top: 3px solid {cor};
            border-radius: 12px;
            padding: 1.4rem;
            margin-bottom: 1rem;
            box-shadow: {card_shadow};
            transition: all 0.2s ease;
            cursor: default;
        " onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 6px 18px rgba(0,0,0,0.15)'"
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='{card_shadow}'">
            <div style="font-size:2.2rem; margin-bottom:0.6rem;">{icone}</div>
            <div style="
                font-family:'Inter',sans-serif;
                font-size:1rem;
                font-weight:700;
                color:{title_color};
                margin-bottom:0.5rem;
            ">{titulo}</div>
            <div style="
                color:{body_color};
                font-size:0.82rem;
                line-height:1.6;
            ">{descricao}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Problema Social ──────────────────────────────────────────────────────────
st.markdown(f"<hr style='border:none; border-top:1px solid {card_border}; margin:2.5rem 0;'>", unsafe_allow_html=True)

danger_bg = "rgba(239,68,68,0.12)" if is_dark else "rgba(239,68,68,0.08)"
danger_border = tokens['danger']
danger_text = "#FCA5A5" if is_dark else "#991B1B"

warning_bg = "rgba(234,179,8,0.12)" if is_dark else "rgba(234,179,8,0.08)"
warning_border = tokens['warning']
warning_text = "#FCD34D" if is_dark else "#92400E"

success_bg = "rgba(34,197,94,0.12)" if is_dark else "rgba(34,197,94,0.08)"
success_border = tokens['success']
success_text = "#86EFAC" if is_dark else "#166534"

social_impact_html = f"""
<div style="max-width:1000px; margin:0 auto;">
    <h2 style="
        font-family:'Inter',sans-serif;
        color:{title_color};
        font-size:1.8rem;
        font-weight:700;
        margin-bottom:1.5rem;
        text-align:center;
    ">🌍 Por que isso importa?</h2>

    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.5rem; margin-bottom:2rem;">
        <div style="
            background:{danger_bg};
            border:1.5px solid {danger_border};
            border-radius:14px;
            padding:1.5rem;
            transition: transform 0.2s ease;
        " onmouseover="this.style.transform='translateY(-3px)'"
           onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size:2.5rem; margin-bottom:0.6rem;">🏥</div>
            <div style="color:{danger_text}; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;">Internações Evitáveis</div>
            <div style="color:{body_color}; font-size:0.88rem; line-height:1.7;">
                Doenças como diarreia, cólera e hepatite A são causadas pela falta de água
                tratada e esgoto coletado — e são 100% evitáveis com saneamento adequado.
            </div>
        </div>
        <div style="
            background:{warning_bg};
            border:1.5px solid {warning_border};
            border-radius:14px;
            padding:1.5rem;
            transition: transform 0.2s ease;
        " onmouseover="this.style.transform='translateY(-3px)'"
           onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size:2.5rem; margin-bottom:0.6rem;">💸</div>
            <div style="color:{warning_text}; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;">Custo Social</div>
            <div style="color:{body_color}; font-size:0.88rem; line-height:1.7;">
                Cada R$ 1 investido em saneamento poupa R$ 4 em saúde pública.
                A falta de saneamento onera o sistema de saúde e reduz a produtividade.
            </div>
        </div>
        <div style="
            background:{success_bg};
            border:1.5px solid {success_border};
            border-radius:14px;
            padding:1.5rem;
            transition: transform 0.2s ease;
        " onmouseover="this.style.transform='translateY(-3px)'"
           onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size:2.5rem; margin-bottom:0.6rem;">🎯</div>
            <div style="color:{success_text}; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;">Decisão Baseada em Dados</div>
            <div style="color:{body_color}; font-size:0.88rem; line-height:1.7;">
                Este dashboard identifica os municípios mais críticos para priorização
                de investimentos, maximizando o impacto social de cada real gasto.
            </div>
        </div>
    </div>
</div>
"""
components.html(social_impact_html, height=420, scrolling=False)

# ─── Metodologia ─────────────────────────────────────────────────────────────
with st.expander("🔬 Metodologia e Fontes de Dados"):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        **📦 Fontes de Dados**
        - **SNIS** (Sistema Nacional de Informações sobre Saneamento): indicadores de água e esgoto
        - **DATASUS/TabNet**: internações hospitalares por doenças de veiculação hídrica
        - **IBGE**: dados populacionais dos municípios

        **📅 Período**
        - Dados históricos a partir de 2006
        - Análise de corte transversal no ano mais recente disponível
        """)
    with col_m2:
        st.markdown("""
        **🤖 Metodologia**
        - **Limpeza de dados**: Interpolação e tratamento de valores ausentes
        - **Índice de Risco Social**: Combinação ponderada de déficits de saneamento e morbidade
        - **Clusterização**: K-Means com normalização StandardScaler
        - **Validação**: Silhouette Score e Método do Cotovelo

        **📐 Testes Estatísticos**
        - Shapiro-Wilk (normalidade)
        - Spearman (correlação)
        - Kruskal-Wallis (hipótese)
        """)

st.markdown(f"""
<div style="
    text-align:center;
    color:{body_color};
    font-size:0.8rem;
    margin-top:3rem;
    padding-top:1.5rem;
    border-top:1px solid {card_border};
    font-weight:500;
">
    Dashboard de Saneamento Básico — Espírito Santo<br>
    <span style="color:{tokens['muted']}; font-size:0.75rem;">Dados: SNIS + DATASUS · Desenvolvido com 💧 e dados</span>
</div>
""", unsafe_allow_html=True)
