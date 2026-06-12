"""Tema global do dashboard (claro/escuro)."""

from __future__ import annotations

import streamlit as st


# ─── Design System — Paleta de Cores ─────────────────────────────────────────

THEME_LIGHT = {
    # Backgrounds
    "bg": "linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 50%, #F8FAFC 100%)",
    "sidebar_bg": "linear-gradient(180deg, #FFFFFF, #F8FAFC)",
    "sidebar_border": "rgba(148,163,184,0.2)",

    # Typography
    "title": "#0F172A",          # Text Primary
    "text": "#334155",            # Text Secondary (mais escuro para melhor contraste)
    "muted": "#475569",           # Text Muted

    # Surfaces
    "surface": "rgba(255,255,255,0.95)",
    "surface_hover": "rgba(248,250,252,0.98)",
    "surface_border": "rgba(148,163,184,0.24)",
    "surface_shadow": "0 2px 8px rgba(15,23,42,0.08)",

    # Interactive Elements
    "tab_bg": "rgba(226,232,240,0.5)",
    "tab_text": "#475569",
    "tab_active_bg": "rgba(37,99,235,0.12)",
    "tab_active_text": "#1E40AF",

    # Semantic Colors
    "primary": "#2563EB",
    "secondary": "#64748B",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "info": "#0EA5E9",

    # Inputs
    "input_bg": "#FFFFFF",
    "input_border": "rgba(148,163,184,0.28)",
    "input_focus": "#2563EB",
    "input_text": "#0F172A",
}

THEME_DARK = {
    # Backgrounds
    "bg": "linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%)",
    "sidebar_bg": "linear-gradient(180deg, #1E293B, #0F172A)",
    "sidebar_border": "rgba(148,163,184,0.12)",

    # Typography
    "title": "#F8FAFC",           # Text Primary
    "text": "#CBD5E1",             # Text Secondary
    "muted": "#94A3B8",            # Text Muted

    # Surfaces
    "surface": "rgba(30,41,59,0.85)",
    "surface_hover": "rgba(30,41,59,0.95)",
    "surface_border": "rgba(148,163,184,0.15)",
    "surface_shadow": "0 2px 12px rgba(0,0,0,0.3)",

    # Interactive Elements
    "tab_bg": "rgba(30,41,59,0.5)",
    "tab_text": "#94A3B8",
    "tab_active_bg": "rgba(59,130,246,0.18)",
    "tab_active_text": "#60A5FA",

    # Semantic Colors
    "primary": "#3B82F6",
    "secondary": "#94A3B8",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#06B6D4",

    # Inputs
    "input_bg": "rgba(30,41,59,0.6)",
    "input_border": "rgba(148,163,184,0.2)",
    "input_focus": "#3B82F6",
    "input_text": "#F8FAFC",
}


def get_theme_mode() -> str:
    return st.session_state.get("theme_mode", "Claro")


def apply_theme(show_toggle: bool = True) -> str:
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "Claro"

    if show_toggle:
        st.sidebar.radio(
            "🎨 Tema",
            options=["Claro", "Escuro"],
            key="theme_mode",
            horizontal=True,
        )

    mode = get_theme_mode()
    tokens = THEME_LIGHT if mode == "Claro" else THEME_DARK

    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Global ─────────────────────────────────────────────────────── */
    .stApp {{
        background: {tokens['bg']};
        font-family: 'Inter', sans-serif;
    }}

    /* ─── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {tokens['sidebar_bg']} !important;
        border-right: 1px solid {tokens['sidebar_border']} !important;
    }}

    [data-testid="stSidebar"] * {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* ─── Typography ─────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {{
        color: {tokens['title']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }}

    p, li, span, div {{
        color: {tokens['text']};
        font-family: 'Inter', sans-serif;
    }}

    /* ─── Inputs (Select, MultiSelect, TextInput) ───────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input {{
        background: {tokens['input_bg']} !important;
        border: 1.5px solid {tokens['input_border']} !important;
        color: {tokens['input_text']} !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within,
    .stTextInput > div > div > input:focus {{
        border-color: {tokens['input_focus']} !important;
        box-shadow: 0 0 0 3px {tokens['input_focus']}20 !important;
        outline: none !important;
    }}

    /* ─── Tabs ───────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {tokens['tab_bg']} !important;
        border-radius: 10px !important;
        padding: 5px !important;
        gap: 6px !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {tokens['tab_text']} !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: {tokens['tab_active_bg']}60 !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: {tokens['tab_active_bg']} !important;
        color: {tokens['tab_active_text']} !important;
        font-weight: 600 !important;
    }}

    /* ─── Metrics ────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {tokens['surface']} !important;
        border: 1px solid {tokens['surface_border']} !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: {tokens['surface_shadow']};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15,23,42,0.12);
    }}

    [data-testid="stMetricValue"] {{
        color: {tokens['title']} !important;
        font-weight: 700 !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {tokens['muted']} !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem !important;
    }}

    /* ─── Sidebar Navigation ─────────────────────────────────────────── */
    [data-testid="stSidebarNav"] a {{
        color: {tokens['text']} !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.8rem !important;
        margin: 2px 0 !important;
        transition: all 0.2s ease;
        font-weight: 500 !important;
    }}

    [data-testid="stSidebarNav"] a:hover {{
        background: {tokens['tab_active_bg']}60 !important;
        color: {tokens['tab_active_text']} !important;
    }}

    [data-testid="stSidebarNav"] a[aria-selected="true"] {{
        background: {tokens['tab_active_bg']} !important;
        color: {tokens['tab_active_text']} !important;
        font-weight: 600 !important;
        border-left: 3px solid {tokens['primary']} !important;
    }}

    /* ─── Buttons ────────────────────────────────────────────────────── */
    .stButton > button {{
        background: {tokens['primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    .stButton > button:hover {{
        background: {tokens['primary']}dd !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ─── DataFrames / Tables ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {tokens['surface_border']} !important;
        border-radius: 10px !important;
        overflow: hidden;
    }}

    /* ─── Expanders ──────────────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {tokens['surface']} !important;
        border: 1px solid {tokens['surface_border']} !important;
        border-radius: 8px !important;
        color: {tokens['title']} !important;
        font-weight: 600 !important;
    }}

    /* ─── Radio Buttons (Theme Toggle) ───────────────────────────────── */
    [role="radiogroup"] label {{
        background: {tokens['surface']} !important;
        border: 1.5px solid {tokens['surface_border']} !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease;
    }}

    [role="radiogroup"] label:has(input:checked) {{
        background: {tokens['tab_active_bg']} !important;
        border-color: {tokens['primary']} !important;
        color: {tokens['tab_active_text']} !important;
        font-weight: 600 !important;
    }}

    /* ─── Hide Streamlit Branding ────────────────────────────────────── */
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}

    /* ─── Scrollbar (Webkit browsers) ────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: {tokens['surface']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {tokens['surface_border']};
        border-radius: 5px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {tokens['muted']};
    }}
</style>
        """,
        unsafe_allow_html=True,
    )

    return mode
