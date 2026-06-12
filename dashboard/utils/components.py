"""
utils/components.py
Componentes de UI reutilizáveis: KPI cards, tooltips, seções, etc.
Todos os componentes seguem o Design System com suporte a modo claro/escuro.
"""

import streamlit as st
from utils.theme import get_theme_mode, THEME_LIGHT, THEME_DARK


def _get_tokens():
    """Retorna os tokens do tema ativo."""
    mode = get_theme_mode()
    return THEME_LIGHT if mode == "Claro" else THEME_DARK


def kpi_card(titulo: str, valor: str, subtitulo: str = "", cor: str = "#60a5fa", icone: str = ""):
    """
    Renderiza um card de KPI estilizado com cor temática.
    Suporta modo claro e escuro com excelente contraste.
    """
    tokens = _get_tokens()
    is_dark = get_theme_mode() == "Escuro"

    card_bg = tokens['surface']
    card_border = tokens['surface_border']
    card_shadow = tokens['surface_shadow']
    text_color = tokens['text']

    st.markdown(
        f"""
        <div style="
            background: {card_bg};
            border: 1px solid {card_border};
            border-top: 3px solid {cor};
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.5rem;
            box-shadow: {card_shadow};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(15,23,42,0.12)'"
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='{card_shadow}'">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                <span style="font-size:1.5rem;">{icone}</span>
                <span style="color:{tokens['muted']}; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                    {titulo}
                </span>
            </div>
            <div style="font-size:2.2rem; font-weight:700; color:{cor}; line-height:1.1; margin:0.3rem 0;">
                {valor}
            </div>
            <div style="color:{text_color}; font-size:0.8rem; margin-top:0.4rem; font-weight:500;">{subtitulo}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def secao_com_tooltip(titulo: str, tooltip: str, nivel: int = 2):
    """
    Renderiza um título de seção com ícone de interrogação (tooltip no hover).
    """
    tokens = _get_tokens()
    tag = f"h{nivel}"
    st.markdown(
        f"""
        <{tag} style="
            color: {tokens['title']};
            font-family: 'Inter', sans-serif;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.8rem;
            font-weight: 700;
        ">
            {titulo}
            <span
                title="{tooltip}"
                style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 22px;
                    height: 22px;
                    background: {tokens['primary']}18;
                    border: 1.5px solid {tokens['primary']}40;
                    border-radius: 50%;
                    font-size: 0.7rem;
                    color: {tokens['primary']};
                    cursor: help;
                    font-weight: 700;
                    transition: all 0.2s ease;
                "
                onmouseover="this.style.background='{tokens['primary']}25'; this.style.borderColor='{tokens['primary']}60'"
                onmouseout="this.style.background='{tokens['primary']}18'; this.style.borderColor='{tokens['primary']}40'"
            >?</span>
        </{tag}>
        """,
        unsafe_allow_html=True,
    )


def badge_zona(zona: str):
    """Renderiza um badge colorido para a zona de vulnerabilidade com excelente contraste."""
    is_dark = get_theme_mode() == "Escuro"

    # Cores ajustadas para modo claro e escuro com contraste WCAG AA
    cores_light = {
        "Zona Verde - Baixo Risco": ("#DCFCE7", "#166534", "#16A34A"),
        "Zona Amarela - Risco Moderado": ("#FEF3C7", "#713F12", "#D97706"),
        "Zona Laranja - Risco Elevado": ("#FFEDD5", "#7C2D12", "#F97316"),
        "Zona Vermelha - Risco Crítico": ("#FEE2E2", "#7F1D1D", "#DC2626"),
    }

    cores_dark = {
        "Zona Verde - Baixo Risco": ("#166534", "#86EFAC", "#22C55E"),
        "Zona Amarela - Risco Moderado": ("#713F12", "#FDE047", "#F59E0B"),
        "Zona Laranja - Risco Elevado": ("#7C2D12", "#FDBA74", "#F97316"),
        "Zona Vermelha - Risco Crítico": ("#7F1D1D", "#FCA5A5", "#EF4444"),
    }

    cores = cores_dark if is_dark else cores_light
    bg, text, border = cores.get(zona, ("#1E293B", "#94A3B8", "#475569"))

    st.markdown(
        f"""
        <span style="
            background: {bg};
            color: {text};
            border: 1.5px solid {border};
            border-radius: 999px;
            padding: 0.4rem 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">{zona}</span>
        """,
        unsafe_allow_html=True,
    )


def cabecalho_pagina(titulo: str, descricao: str, icone: str = ""):
    """Renderiza o cabeçalho de uma página com título e descrição."""
    tokens = _get_tokens()
    is_dark = get_theme_mode() == "Escuro"

    header_bg = (
        "linear-gradient(135deg, rgba(30,41,59,0.7), rgba(15,23,42,0.6))"
        if is_dark
        else "linear-gradient(135deg, rgba(219,234,254,0.6), rgba(255,255,255,0.9))"
    )

    st.markdown(
        f"""
        <div style="
            background: {header_bg};
            border: 1px solid {tokens['surface_border']};
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            box-shadow: {tokens['surface_shadow']};
        ">
            <div style="font-size:3rem; margin-bottom:0.6rem;">{icone}</div>
            <h1 style="
                color: {tokens['title']};
                font-family: 'Inter', sans-serif;
                font-size: 2rem;
                font-weight: 800;
                margin: 0 0 0.6rem 0;
                line-height: 1.2;
            ">{titulo}</h1>
            <p style="
                color: {tokens['text']};
                font-size: 1rem;
                line-height: 1.7;
                margin: 0;
                max-width: 750px;
            ">{descricao}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def caixa_explicacao(texto: str, tipo: str = "info"):
    """
    Exibe uma caixa de explicação colorida.
    tipo: 'info', 'aviso', 'sucesso', 'critico'
    """
    is_dark = get_theme_mode() == "Escuro"

    configs_light = {
        "info": ("💡", "rgba(59,130,246,0.08)", "#2563EB", "#1E40AF"),
        "aviso": ("⚠️", "rgba(217,119,6,0.08)", "#D97706", "#92400E"),
        "sucesso": ("✅", "rgba(22,163,74,0.08)", "#16A34A", "#166534"),
        "critico": ("🔴", "rgba(220,38,38,0.08)", "#DC2626", "#991B1B"),
    }

    configs_dark = {
        "info": ("💡", "rgba(59,130,246,0.15)", "#3B82F6", "#93C5FD"),
        "aviso": ("⚠️", "rgba(245,158,11,0.15)", "#F59E0B", "#FCD34D"),
        "sucesso": ("✅", "rgba(34,197,94,0.15)", "#22C55E", "#86EFAC"),
        "critico": ("🔴", "rgba(239,68,68,0.15)", "#EF4444", "#FCA5A5"),
    }

    configs = configs_dark if is_dark else configs_light
    icone, bg, border, cor_texto = configs.get(tipo, configs["info"])

    st.markdown(
        f"""
        <div style="
            background: {bg};
            border-left: 4px solid {border};
            border-radius: 0 10px 10px 0;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
        ">
            <span style="font-size:1.1rem; margin-right:0.5rem;">{icone}</span>
            <span style="color:{cor_texto}; font-size:0.9rem; line-height:1.7; font-weight:500;"> {texto}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def separador():
    """Linha separadora estilizada."""
    tokens = _get_tokens()
    st.markdown(
        f"""<hr style="border:none; border-top:1px solid {tokens['surface_border']}; margin:1.5rem 0;">""",
        unsafe_allow_html=True,
    )


def metrica_inline(rotulo: str, valor: str, cor: str = ""):
    """Renderiza uma métrica compacta inline."""
    tokens = _get_tokens()
    cor_final = cor if cor else tokens['primary']

    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            background: {tokens['surface']};
            border: 1px solid {tokens['surface_border']};
            border-radius: 10px;
            padding: 0.8rem 1.1rem;
            box-shadow: {tokens['surface_shadow']};
            transition: transform 0.2s ease;
        " onmouseover="this.style.transform='translateY(-2px)'"
           onmouseout="this.style.transform='translateY(0)'">
            <span style="color:{tokens['muted']}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">{rotulo}</span>
            <span style="color:{cor_final}; font-size:1.3rem; font-weight:700; margin-top:0.3rem;">{valor}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
