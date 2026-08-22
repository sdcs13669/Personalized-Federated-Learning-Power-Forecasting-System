// 全局状态
const App = {
  token: localStorage.getItem("token") || null,
  user: null,           // {id, username, role}
  mode: "server",       // "server" | "client"（探测 /local/status）
  polling: null,
};

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (App.token) headers["Authorization"] = "Bearer " + App.token;
  const r = await fetch(path, { ...opts, headers });
  if (r.status === 401) { App.token = null; localStorage.removeItem("token"); location.hash = "#/login"; throw new Error("未登录或登录已过期"); }
  if (!r.ok) {
    let detail = "请求失败";
    try { detail = (await r.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return r.json();
}

function showToast(msg, isErr = false) {
  const t = document.createElement("div");
  t.className = "toast";
  t.style.background = isErr ? "#c0392b" : "#333";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

// 简单 hash 路由
function router() {
  const hash = location.hash || "#/plaza";
  App.user = null;
  if (App.token) {
    api("/api/me").then(u => {
      App.user = u;
      renderNav();
      const view = hash;
      if (view === "#/login") location.hash = "#/plaza";
      dispatch(view);
    }).catch(() => { location.hash = "#/login"; });
  } else if (hash !== "#/login") {
    location.hash = "#/login";
  } else {
    renderLogin();
  }
}

function dispatch(hash) {
  highlightNav(hash);
  const parts = hash.split("/");
  // 登录页使用专属全屏布局，其它页面切回普通布局
  document.body.classList.toggle("login-page", parts[1] === "login");
  // 性能：离开登录页停止粒子动画；离开"我的任务"停止轮询
  if (parts[1] !== "login" && typeof stopParticleNet === "function") stopParticleNet();
  if (parts[1] !== "my" && App.agentPolling) { clearInterval(App.agentPolling); App.agentPolling = null; }
  if (parts[1] === "plaza") renderPlaza();
  else if (parts[1] === "my") renderMyTasks();
  else if (parts[1] === "admin") renderAdmin();
  else if (parts[1] === "task") renderTaskDetail(parts[2]);
  else if (parts[1] === "login") renderLogin();
  else location.hash = "#/plaza";
}

// 高亮当前页面对应的导航项
function highlightNav(hash) {
  document.querySelectorAll("header nav a").forEach(a => {
    a.classList.toggle("active", a.getAttribute("href") === (hash || "#/plaza"));
  });
}

function renderNav() {
  document.getElementById("topbar").classList.remove("hidden");
  document.getElementById("user-label").textContent = App.user ? App.user.username + (App.user.role === "admin" ? "（管理员）" : "") : "";
  document.getElementById("logout-btn").classList.remove("hidden");
  const navMine = document.getElementById("nav-mine");
  const navAdmin = document.getElementById("nav-admin");
  navMine.classList.toggle("hidden", App.mode !== "client");
  navAdmin.classList.toggle("hidden", !(App.user && App.user.role === "admin"));
}

document.getElementById("logout-btn").addEventListener("click", () => {
  App.token = null; localStorage.removeItem("token"); location.hash = "#/login";
});
window.addEventListener("hashchange", router);

// 探测本地 agent（客户端模式）
async function detectMode() {
  try {
    const r = await fetch("/local/status", { method: "GET" });
    if (r.ok) { App.mode = "client"; return; }
  } catch (e) {}
  App.mode = "server";
}

detectMode().then(() => router());
