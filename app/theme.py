"""App 统一视觉主题（火星沙丘 · 暖色科技感）。

底色是深暖沙丘色，配暗琥珀橙主色 + 沙金点缀，像火星基地/太阳能设备的暖光。
字体微软雅黑保持清晰（配合 DPI 校准不发虚）。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# ---- 配色 ----
INK       = "#F5EDE0"   # 标题 / 强调文字（暖白）
ACCENT    = "#D97706"   # 暗琥珀橙：主按钮 / 强调线
ACCENT_DK = "#B45309"   # 暗琥珀（悬停 / 按下）
GOLD      = "#FBBF24"   # 沙金：点缀 / 辉光
BG        = "#241A0E"   # 页面背景：深暖沙（不黑）
CARD      = "#2E2010"   # 卡片背景：暖炭
TEXT      = "#F5EDE0"   # 正文：暖白
MUTED     = "#C9BFA8"   # 次要文字：暖灰
LINE      = "#3A2A17"   # 边框：暖深棕
SUB       = "#A9967A"   # 副标题
GREEN     = "#84CC16"   # 成功 / 已连接
LIGHT     = "#FFFFFF"   # 高亮字（备用）

FONT      = "Microsoft YaHei UI"   # 中文最清晰，不发虚
FONT_MONO = "Consolas"


def setup(root: tk.Tk) -> ttk.Style:
    """应用火星沙丘主题。固定用内置 clam 主题，保证各机器显示一致。"""
    style = ttk.Style(root)
    style.theme_use("clam")

    root.configure(bg=BG)
    style.configure(".", font=(FONT, 10))

    # 输入框
    style.configure("TEntry", fieldbackground=CARD, foreground=TEXT,
                    insertcolor=TEXT, padding=6,
                    bordercolor=LINE, lightcolor=LINE, darkcolor=LINE)

    # 普通按钮
    style.configure("TButton", background="#241A0E", foreground=TEXT,
                    padding=(14, 7), bordercolor=LINE)
    style.map("TButton",
              background=[("active", "#2E2112"), ("disabled", "#1A1108")],
              foreground=[("disabled", "#6E6249")])

    # 主强调按钮（暗琥珀，深色字保证对比）
    style.configure("Accent.TButton", background=ACCENT, foreground="#1A1005",
                    font=(FONT, 10, "bold"))
    style.map("Accent.TButton",
              background=[("active", ACCENT_DK), ("disabled", "#3A2A17")],
              foreground=[("disabled", "#A9967A")])

    # 小按钮（退出登录等）
    style.configure("Small.TButton", background="#241A0E", foreground=MUTED,
                    padding=(10, 4), font=(FONT, 9))
    style.map("Small.TButton",
              background=[("active", "#2E2112")],
              foreground=[("active", GOLD)])

    # 底部导航按钮
    style.configure("Nav.TButton", background=BG, foreground=MUTED,
                    font=(FONT, 11, "bold"), padding=(10, 12),
                    bordercolor=LINE)
    style.map("Nav.TButton",
              background=[("active", "#241A0E")],
              foreground=[("active", GOLD)])

    # 表格
    style.configure("Treeview", background=CARD, fieldbackground=CARD,
                    foreground=TEXT, rowheight=32, borderwidth=0,
                    font=(FONT, 10))
    style.configure("Treeview.Heading", background="#241A0E",
                    foreground=GOLD, font=(FONT, 10, "bold"),
                    relief="flat", padding=6)
    style.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#1A1005")])
    return style


def build_header(parent: tk.Widget, title: str, subtitle: str = "") -> tk.Frame:
    """沙丘色标题栏 + 暗琥珀强调线，返回该 Frame（已 pack）。"""
    header = tk.Frame(parent, bg=BG)
    header.pack(fill="x")
    tk.Label(header, text=title, bg=BG, fg=INK,
             font=(FONT, 17, "bold")).pack(anchor="w", padx=20, pady=(16, 2))
    if subtitle:
        tk.Label(header, text=subtitle, bg=BG, fg=SUB,
                 font=(FONT, 9)).pack(anchor="w", padx=20, pady=(0, 8))
    tk.Frame(header, bg=ACCENT, height=3).pack(fill="x")
    return header
