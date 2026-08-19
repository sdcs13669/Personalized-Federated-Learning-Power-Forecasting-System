"""App 主界面：登录注册 / 任务广场 / 当前任务 三页切换。

竞赛主题：面向能源可信数据空间的多方协同与隐私保护技术创新解决方案。
登录页：沙丘暖色渐变背景 + Canvas 自绘输电塔/电线（满屏自适应）+ 内容随窗口缩放。
运行：python app\\main.py
"""
from __future__ import annotations

import math
import os
import sys
# 让 "python app\\main.py" 也能找到 app 包（把项目根目录加进搜索路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 高分屏（Windows 缩放 >100%）下让 tkinter 字体不发虚：声明 DPI 感知
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from tkinter import messagebox

from app import theme
from app import api


def build_nav(parent, app) -> tk.Frame:
    """底部导航：任务广场 / 当前任务。"""
    nav = tk.Frame(parent, bg=theme.BG,
                   highlightbackground=theme.LINE, highlightthickness=1)
    nav.pack(side="bottom", fill="x")
    for text, target in [("任务广场", "square"), ("当前任务", "current")]:
        ttk.Button(nav, text=text, style="Nav.TButton",
                   command=lambda t=target: app.show_page(t)).pack(
            side="left", expand=True, fill="x")
    return nav


class App(tk.Tk):
    """主窗口：三个页面叠放，靠 show_page 切换（显示一个、隐藏其他）。"""

    def __init__(self) -> None:
        super().__init__()
        # 按真实 DPI 校准 tk 缩放，字体在高分屏下保持清晰
        self.tk.call("tk", "scaling", self.winfo_fpixels("1i") / 72)
        self.title("面向能源可信数据空间 · 联邦学习")
        self.geometry("620x860")
        self.minsize(560, 760)
        theme.setup(self)
        self.current_user: str | None = None
        self.token: str | None = None   # 登录后保存 access_token

        self.container = tk.Frame(self, bg=theme.BG)
        self.container.pack(fill="both", expand=True)

        self.pages: dict[str, tk.Frame] = {}
        for name, cls in [("login", LoginPage),
                          ("square", SquarePage),
                          ("current", CurrentPage)]:
            page = cls(self.container, self)
            self.pages[name] = page
        self.show_page("login")

    def show_page(self, name: str) -> None:
        # 先通知其他页"离开"（停掉动画等）
        for n, p in self.pages.items():
            if n != name and hasattr(p, "on_hide"):
                p.on_hide()
        # 用 pack 显示目标页、隐藏其他页
        for p in self.pages.values():
            p.pack_forget()
        page = self.pages[name]
        page.pack(fill="both", expand=True)
        page.on_show()   # 每次进入刷新内容


class LoginPage(tk.Canvas):
    """登录注册页：沙丘暖色渐变背景 + 输电塔电线（满屏自适应）+ 内容随窗口缩放。"""

    def __init__(self, parent, app):
        super().__init__(parent, bg=theme.BG, highlightthickness=0)
        self.app = app
        self.mode = "login"  # "login" | "register"
        self._anim_id = None
        self._frame = 0
        self._bg_items: list[int] = []
        self._pulse: int | None = None
        self._fonts: list[tkfont.Font] = []
        self._scale = 1.0
        self._cw, self._ch = 620, 860
        self._wire_path = [(0, 0), (0, 0), (0, 0)]
        self._build_ui()
        self.bind("<Configure>", self._on_resize)

    # ---------- 动态字体（随窗口放大） ----------
    def _make_font(self, size: int, weight: str = "normal") -> tkfont.Font:
        f = tkfont.Font(family=theme.FONT, size=size, weight=weight)
        f.base_size = size   # 记录原始字号，缩放用
        self._fonts.append(f)
        return f

    def _update_scale(self) -> None:
        scale = min(self._cw / 620, self._ch / 860)
        scale = max(0.72, min(scale, 1.7))
        if abs(scale - self._scale) < 0.04:
            return
        self._scale = scale
        for f in self._fonts:
            f.configure(size=max(6, int(f.base_size * scale)))

    # ---------- 背景：沙丘渐变 + 输电塔（满屏自适应） ----------
    def _on_resize(self, event) -> None:
        if event.width < 10 or event.height < 10:
            return
        self._cw, self._ch = event.width, event.height
        self._redraw_background()
        self._update_scale()
        # 内容重新居中（偏上），面板宽度跟随窗口
        if hasattr(self, "_content_window"):
            self.coords(self._content_window, self._cw / 2, self._ch * 0.15)
            self.itemconfig(self._content_window, width=int(self._cw * 0.74))

    def _redraw_background(self) -> None:
        for item in self._bg_items:
            self.delete(item)
        self._bg_items = []
        if self._pulse is not None:
            self.delete(self._pulse)
            self._pulse = None
        w, h = self._cw, self._ch

        # 沙丘暖色渐变背景（顶部暖沙亮 -> 底部深沙，无日落）
        top = (72, 50, 28)
        bot = (26, 17, 9)
        steps = 48
        for i in range(steps):
            t = i / (steps - 1)
            c = tuple(int(top[j] + (bot[j] - top[j]) * t) for j in range(3))
            y0 = int(h * i / steps)
            y1 = int(h * (i + 1) / steps)
            self._bg_items.append(self.create_rectangle(
                0, y0, w, y1, fill="#%02x%02x%02x" % c, outline=""))

        # 沙丘剪影：三层（深暖色）
        dunes = [
            ("#3A2A15", [(0, h * 0.80), (w * 0.22, h * 0.76), (w * 0.5, h * 0.80),
                         (w * 0.74, h * 0.76), (w, h * 0.79), (w, h), (0, h)]),
            ("#2A1C0D", [(0, h * 0.87), (w * 0.3, h * 0.83), (w * 0.6, h * 0.86),
                         (w * 0.85, h * 0.83), (w, h * 0.86), (w, h), (0, h)]),
            ("#20150A", [(0, h * 0.94), (w * 0.4, h * 0.91), (w * 0.7, h * 0.93),
                         (w, h * 0.90), (w, h), (0, h)]),
        ]
        for color, pts in dunes:
            self._bg_items.append(self.create_polygon(pts, fill=color, outline=""))

        # 输电塔：三座，右边那座最高（高压塔）
        towers = [(0.14, 0.82, 0.30), (0.55, 0.88, 0.22), (0.88, 0.83, 0.40)]
        tops = []
        for tx, ty, th in towers:
            tops.append(self._draw_tower(w * tx, h * ty, h * th))
        self._wire_path = tops

        # 电线：塔顶相连（轻微下垂）
        (x1, y1), (x2, y2), (x3, y3) = tops
        self._bg_items.append(
            self.create_line(x1, y1, (x1 + x2) / 2, (y1 + y2) / 2 + h * 0.03,
                             x2, y2, smooth=True, fill="#5A4428", width=1))
        self._bg_items.append(
            self.create_line(x2, y2, (x2 + x3) / 2, (y2 + y3) / 2 + h * 0.03,
                             x3, y3, smooth=True, fill="#5A4428", width=1))

        # 能量脉冲点（沿第一段电线流动）
        self._pulse = self.create_oval(0, 0, 10, 10, fill=theme.ACCENT,
                                       outline=theme.GOLD)

    def _draw_tower(self, x: float, base_y: float, height: float,
                    color: str = "#150E06") -> tuple[float, float]:
        top_y = base_y - height
        self._bg_items.append(self.create_line(x - 22, base_y, x, top_y,
                                               fill=color, width=2))
        self._bg_items.append(self.create_line(x + 22, base_y, x, top_y,
                                               fill=color, width=2))
        for k in [0.5, 0.72, 0.9]:
            yy = base_y - height * k
            self._bg_items.append(self.create_line(x - 14, yy, x + 14, yy,
                                                   fill=color, width=2))
        self._bg_items.append(self.create_line(x, top_y, x, top_y - 12,
                                               fill=color, width=2))
        return (x, top_y)

    # ---------- 登录内容（用 tk 控件 + 动态字体，随窗口放大） ----------
    def _build_ui(self) -> None:
        # 深棕"面板"（用户看到的框）：整体加宽 + 内容与边缘留白加大
        PADX = 46
        outer = tk.Frame(self, bg=theme.CARD)
        self._content = outer

        # 品牌徽标：暗琥珀圆 + 深色网络节点
        emblem = tk.Canvas(outer, width=84, height=84, bg=theme.CARD,
                           highlightthickness=0)
        emblem.pack(pady=(34, 12))
        emblem.create_oval(4, 4, 80, 80, fill=theme.ACCENT, outline="")
        emblem.create_line(22, 30, 44, 24, 48, 52, 22, 30,
                           fill="#241A0E", width=2)
        for x, y in [(22, 30), (44, 24), (48, 52)]:
            emblem.create_oval(x - 5, y - 5, x + 5, y + 5,
                               fill="#241A0E", outline="")
        emblem.create_oval(38, 36, 46, 44, fill="#241A0E", outline="")
        self._glow = emblem.create_oval(2, 2, 82, 82, outline=theme.GOLD,
                                        width=4, state="hidden")
        self._emblem = emblem

        tk.Label(outer, text="面向能源可信数据空间", bg=theme.CARD, fg=theme.INK,
                 font=self._make_font(21, "bold")).pack(pady=(14, 4))
        tk.Label(outer, text="多方协同 · 隐私保护 创新解决方案", bg=theme.CARD,
                 fg=theme.MUTED, font=self._make_font(10)).pack(pady=(0, 26))

        tk.Label(outer, text="用户名", bg=theme.CARD, fg=theme.MUTED,
                 font=self._make_font(10)).pack(anchor="w", padx=PADX, pady=(0, 6))
        self.user_var = tk.StringVar()
        tk.Entry(outer, textvariable=self.user_var, bg=theme.BG, fg=theme.TEXT,
                 insertbackground=theme.TEXT, relief="flat",
                 font=self._make_font(11)).pack(fill="x", padx=PADX, pady=(0, 12), ipady=7)

        tk.Label(outer, text="密码", bg=theme.CARD, fg=theme.MUTED,
                 font=self._make_font(10)).pack(anchor="w", padx=PADX, pady=(0, 6))
        self.pass_var = tk.StringVar()
        tk.Entry(outer, textvariable=self.pass_var, show="*", bg=theme.BG,
                 fg=theme.TEXT, insertbackground=theme.TEXT, relief="flat",
                 font=self._make_font(11)).pack(fill="x", padx=PADX, pady=(0, 12), ipady=7)

        self.main_btn = tk.Button(outer, text="登录", bg=theme.ACCENT, fg="#1A1005",
                                  activebackground=theme.ACCENT_DK,
                                  activeforeground="#1A1005", relief="flat",
                                  cursor="hand2", font=self._make_font(11, "bold"),
                                  command=self._on_main)
        self.main_btn.pack(fill="x", padx=PADX, pady=(14, 6), ipady=5)

        self.switch_btn = tk.Button(outer, text="没有账号？去注册", bg=theme.CARD,
                                    fg=theme.MUTED, activebackground=theme.CARD,
                                    activeforeground=theme.GOLD, relief="flat",
                                    cursor="hand2", bd=0,
                                    font=self._make_font(9),
                                    command=self._toggle_mode)
        self.switch_btn.pack(pady=(4, 26))

        tk.Label(outer, text="联邦学习 · 隐私保护 · 数据不出本机", bg=theme.CARD,
                 fg=theme.MUTED, font=self._make_font(9)).pack(pady=(0, 30))

        # 面板放上画布：宽度 = 窗口宽 74%，随窗口缩放
        self._content_window = self.create_window(
            310, 150, window=outer, anchor="n", width=int(self._cw * 0.74))

    # ---------- 小动画：能量脉冲 + 徽标辉光呼吸 ----------
    def _start_anim(self) -> None:
        if self._anim_id is None:
            self._anim_pulse()

    def _anim_pulse(self) -> None:
        self._frame += 1
        # 能量脉冲沿第一段电线流动（带下垂弧线）
        (x1, y1), (x2, y2) = self._wire_path[0], self._wire_path[1]
        t = (self._frame % 100) / 100.0
        x = x1 + (x2 - x1) * t
        sag = self._ch * 0.03
        y = y1 + (y2 - y1) * t + sag * math.sin(math.pi * t)
        if self._pulse is None:
            self._pulse = self.create_oval(0, 0, 10, 10, fill=theme.ACCENT,
                                           outline=theme.GOLD)
        self.coords(self._pulse, x - 5, y - 5, x + 5, y + 5)

        # 徽标外圈辉光呼吸
        if self._frame % 20 == 0:
            state = "normal" if (self._frame // 20) % 2 == 0 else "hidden"
            self._emblem.itemconfig(self._glow, state=state)

        self._anim_id = self.after(30, self._anim_pulse)

    def on_show(self) -> None:
        self._start_anim()

    def on_hide(self) -> None:
        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None

    # ---------- 动作 ----------
    def _toggle_mode(self) -> None:
        self.mode = "register" if self.mode == "login" else "login"
        self.main_btn.config(text="注册" if self.mode == "register" else "登录")
        self.switch_btn.config(text="已有账号？去登录" if self.mode == "register"
                               else "没有账号？去注册")

    def _on_main(self) -> None:
        user = self.user_var.get().strip()
        pwd = self.pass_var.get()
        if not user or not pwd:
            messagebox.showwarning("提示", "请输入用户名和密码")
            return
        try:
            if self.mode == "register":
                token = api.register(user, pwd)
            else:
                token = api.login(user, pwd)
        except api.ApiError as e:
            messagebox.showerror("失败",
                                 f"{'注册' if self.mode=='register' else '登录'}失败：{e.message}")
            return
        except Exception:
            messagebox.showerror("连不上后端",
                                 "无法连接后端服务，请确认后端已启动")
            return
        self.app.token = token
        self.app.current_user = user
        self.app.show_page("square")
        messagebox.showinfo("成功", f"欢迎，{user}！")


class SquarePage(tk.Frame):
    """任务广场：展示所有任务（真后端）。"""

    def __init__(self, parent, app):
        super().__init__(parent, bg=theme.BG)
        self.app = app
        self._build_ui()

    def _build_ui(self) -> None:
        build_nav(self, self.app)
        theme.build_header(self, "任务广场",
                           "浏览所有联邦学习任务 · 能源可信数据空间")

        bar = tk.Frame(self, bg=theme.BG)
        bar.pack(fill="x", padx=18, pady=(12, 0))
        self.user_label = tk.Label(bar, text="", bg=theme.BG, fg=theme.MUTED,
                                   font=(theme.FONT, 10))
        self.user_label.pack(side="left")
        ttk.Button(bar, text="退出登录", style="Small.TButton",
                   command=self._logout).pack(side="right")

        card = tk.Frame(self, bg=theme.CARD,
                        highlightbackground=theme.LINE, highlightthickness=1)
        card.pack(fill="both", expand=True, padx=18, pady=10)

        cols = ("name", "owner", "rounds", "status")
        heads = ["任务名", "发起人", "轮次", "状态"]
        widths = [230, 90, 70, 90]
        self.tree = ttk.Treeview(card, columns=cols, show="headings", height=9)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=12, pady=12)

        self.empty_label = tk.Label(card, text="加载中…", bg=theme.CARD,
                                    fg=theme.MUTED, font=(theme.FONT, 9))
        self.empty_label.pack(pady=(0, 8))

        ttk.Button(card, text="＋ 发起新任务", style="Accent.TButton",
                   command=self._new_task).pack(padx=12, pady=(0, 12))

    def on_show(self) -> None:
        self.user_label.config(text=f"你好，{self.app.current_user or '用户'}")
        self._refresh()

    def _refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if not self.app.token:
            self.empty_label.config(text="请先登录")
            return
        try:
            tasks = api.get_tasks(self.app.token)
        except Exception:
            self.empty_label.config(text="连不上后端，请确认后端已启动")
            return
        for t in tasks:
            self.tree.insert("", "end",
                             values=(t.get("name", ""), t.get("creator", ""),
                                     t.get("rounds", ""), t.get("status", "")))
        self.empty_label.config(text="暂无招募中的任务" if not tasks else "")

    def _logout(self) -> None:
        self.app.current_user = None
        self.app.token = None
        self.app.show_page("login")

    def _new_task(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("发起新任务")
        dlg.configure(bg=theme.BG)
        dlg.geometry("400x250")
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text="发起新任务", bg=theme.BG, fg=theme.INK,
                 font=(theme.FONT, 14, "bold")).pack(pady=(18, 10))

        form = tk.Frame(dlg, bg=theme.BG)
        form.pack(fill="x", padx=34)
        tk.Label(form, text="任务名", bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT, 10)).grid(row=0, column=0, sticky="e", padx=(0, 10), pady=5)
        name_var = tk.StringVar()
        ttk.Entry(form, textvariable=name_var, width=22).grid(row=0, column=1, pady=5)
        tk.Label(form, text="轮次数", bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT, 10)).grid(row=1, column=0, sticky="e", padx=(0, 10), pady=5)
        rounds_var = tk.StringVar(value="30")
        ttk.Entry(form, textvariable=rounds_var, width=22).grid(row=1, column=1, pady=5)

        def submit():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("提示", "请输入任务名")
                return
            try:
                rounds = int(rounds_var.get().strip() or "30")
            except ValueError:
                messagebox.showwarning("提示", "轮次数必须是数字")
                return
            try:
                task = api.create_task(self.app.token, name, rounds)
            except api.ApiError as e:
                messagebox.showerror("失败", f"创建任务失败：{e.message}")
                return
            except Exception:
                messagebox.showerror("连不上后端",
                                     "无法连接后端服务，请确认后端已启动")
                return
            dlg.destroy()
            messagebox.showinfo("发起成功",
                                f"任务「{task.get('name')}」已创建！\n\n"
                                f"参与密钥：{task.get('key')}\n"
                                f"请把 服务器地址+端口+密钥 发给参与者。")
            self._refresh()

    def on_hide(self) -> None:
        pass


class CurrentPage(tk.Frame):
    """当前任务页：我参与的任务 + 隐私预算审计（待后端接口）。"""

    def __init__(self, parent, app):
        super().__init__(parent, bg=theme.BG)
        self.app = app
        self._build_ui()

    def _build_ui(self) -> None:
        build_nav(self, self.app)
        theme.build_header(self, "当前任务",
                           "我参与的任务与隐私预算审计")

        bar = tk.Frame(self, bg=theme.BG)
        bar.pack(fill="x", padx=18, pady=(12, 0))
        self.user_label = tk.Label(bar, text="", bg=theme.BG, fg=theme.MUTED,
                                   font=(theme.FONT, 10))
        self.user_label.pack(side="left")

        card = tk.Frame(self, bg=theme.CARD,
                        highlightbackground=theme.LINE, highlightthickness=1)
        card.pack(fill="both", expand=True, padx=18, pady=10)

        cols = ("name", "role", "eps", "budget", "progress")
        heads = ["任务名", "角色", "已用 ε", "预算 ε", "进度"]
        widths = [220, 60, 80, 80, 90]
        self.tree = ttk.Treeview(card, columns=cols, show="headings", height=8)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=12, pady=12)

        tk.Label(card, text="暂无参与的任务（待后端提供『我的任务』接口后展示）", bg=theme.CARD,
                 fg=theme.MUTED, font=(theme.FONT, 9)).pack(anchor="w", padx=16, pady=(0, 6))

        tk.Label(card, text="隐私预算审计：联邦学习每训练一轮会消耗一点隐私预算（ε），"
                            "ε 越小越安全。这里展示每个任务用了多少、还剩多少。",
                 bg=theme.CARD, fg=theme.MUTED, font=(theme.FONT, 9),
                 justify="left", wraplength=440).pack(anchor="w", padx=16, pady=(0, 12))

    def on_show(self) -> None:
        self.user_label.config(text=f"你好，{self.app.current_user or '用户'}")
        self.tree.delete(*self.tree.get_children())

    def on_hide(self) -> None:
        pass


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
