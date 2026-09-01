// ===== 登录/注册 =====
function renderLogin() {
  document.body.classList.add("login-page");
  document.getElementById("topbar").classList.add("hidden");
  document.getElementById("view").innerHTML = `
    <div class="login-page">
      <canvas id="net-canvas" class="net-canvas" aria-hidden="true"></canvas>
      <div class="energy-ring" aria-hidden="true"></div>
      <div class="auth-card anim-in">
        <div class="card-status">
          <span class="status-dot"></span>
          <span class="status-tag">FL-NET · NODE</span>
        </div>
        <div class="auth-logo">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="7" cy="12" r="2.4" stroke="#ffffff" stroke-width="2"/>
            <circle cx="17" cy="7" r="2.4" stroke="#ffffff" stroke-width="2"/>
            <circle cx="17" cy="17" r="2.4" stroke="#ffffff" stroke-width="2"/>
            <path d="M9.2 11.2l5.6-3M9.2 12.8l5.6 3.2" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <h1 class="auth-title">联邦学习平台</h1>
        <p class="auth-sub">Federated Learning · Power Grid</p>
        <div class="form-row"><label for="login-user">用户名</label><input id="login-user" autocomplete="username" placeholder="请输入用户名"></div>
        <div class="form-row"><label for="login-pass">密码</label><input id="login-pass" type="password" autocomplete="current-password" placeholder="请输入密码"></div>
        <button class="btn-block" onclick="doLogin()">登 录</button>
        <button class="secondary btn-block" onclick="doRegister()">注册新账号</button>
      </div>
      <div class="net-status" aria-hidden="true">
        <span class="ns-dot"></span>
        <span id="ns-text">FL-NET v2.13 · 联邦节点 04 · ε 审计开启</span>
      </div>
    </div>`;
  initParticleNet();
}

// ===== 登录页粒子网络动画 =====
let _netHandle = null;
let _netResize = null;
function initParticleNet() {
  const canvas = document.getElementById("net-canvas");
  if (!canvas) return;
  stopParticleNet(); // 清理上一次动画与 resize 监听
  // 尊重系统"减少动态"
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return; // 不启动动画
  }
  const ctx = canvas.getContext("2d");
  let w, h;
  const N = 38;
  const parts = [];
  function resize() {
    // 用画布实际渲染尺寸，保证逻辑坐标 == 屏幕坐标（不会压缩错位）
    w = canvas.width = Math.max(1, canvas.clientWidth);
    h = canvas.height = Math.max(1, canvas.clientHeight);
  }
  resize();
  _netResize = resize;
  window.addEventListener("resize", _netResize);
  for (let i = 0; i < N; i++) {
    parts.push({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - .5) * .32,
      vy: (Math.random() - .5) * .32,
      r: Math.random() * 1.6 + .5,
    });
  }
  const LINK = 120;
  const LINK2 = LINK * LINK;
  function tick() {
    ctx.clearRect(0, 0, w, h);
    for (const p of parts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < -10 || p.x > w + 10) p.vx *= -1;
      if (p.y < -10 || p.y > h + 10) p.vy *= -1;
    }
    for (let i = 0; i < parts.length; i++) {
      for (let j = i + 1; j < parts.length; j++) {
        const a = parts[i], b = parts[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < LINK2) {
          const alpha = (1 - Math.sqrt(d2) / LINK) * .2;
          ctx.strokeStyle = "rgba(37,99,235," + alpha.toFixed(3) + ")";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    for (const p of parts) {
      ctx.fillStyle = "rgba(37,99,235,.55)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    _netHandle = requestAnimationFrame(tick);
  }
  tick();
}

// 离开登录页时停止粒子动画，避免后台空转
function stopParticleNet() {
  if (_netHandle) { cancelAnimationFrame(_netHandle); _netHandle = null; }
  if (_netResize) { window.removeEventListener("resize", _netResize); _netResize = null; }
}

async function doLogin() {
  try {
    const u = document.getElementById("login-user").value.trim();
    const p = document.getElementById("login-pass").value;
    if (!u || !p) return showToast("请输入用户名和密码", true);
    const r = await fetch("/api/auth/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "登录失败");
    App.token = d.access_token;
    localStorage.setItem("token", App.token);
    location.hash = "#/plaza";
  } catch (e) { showToast(e.message, true); }
}

async function doRegister() {
  try {
    const u = document.getElementById("login-user").value.trim();
    const p = document.getElementById("login-pass").value;
    if (!u || !p) return showToast("请输入用户名和密码", true);
    const r = await fetch("/api/auth/register", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "注册失败");
    App.token = d.access_token;
    localStorage.setItem("token", App.token);
    showToast("注册成功");
    location.hash = "#/plaza";
  } catch (e) { showToast(e.message, true); }
}

// ===== 广场 =====
function renderPlaza() {
  document.getElementById("view").innerHTML = `
    <div class="page-head anim-in">
      <div>
        <h2>任务广场</h2>
        <p class="page-sub">浏览可加入的联邦学习任务，或发起新任务</p>
      </div>
      <button onclick="showCreateTask()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>
        发起新任务
      </button>
    </div>
    <div class="stat-cards anim-in anim-d1">
      <div class="stat"><div class="num" id="st-total">–</div><div class="lbl">全部任务</div></div>
      <div class="stat"><div class="num" id="st-recruiting">–</div><div class="lbl">招募中</div></div>
      <div class="stat"><div class="num" id="st-training">–</div><div class="lbl">训练中</div></div>
    </div>
    <div class="card anim-in anim-d2">
      <table>
        <thead><tr>
          <th>任务名</th><th>发起者</th><th>轮次</th><th>DP ε</th><th>状态</th><th>参与</th><th>操作</th>
        </tr></thead>
        <tbody id="task-rows"></tbody>
      </table>
    </div>`;
  loadPlaza();
}

async function loadPlaza() {
  try {
    const tasks = await api("/api/tasks");
    const recruiting = tasks.filter(t => t.status === "recruiting").length;
    const training = tasks.filter(t => t.status === "training").length;
    document.getElementById("st-total").textContent = tasks.length;
    document.getElementById("st-recruiting").textContent = recruiting;
    document.getElementById("st-training").textContent = training;
    if (!tasks.length) {
      document.getElementById("task-rows").innerHTML =
        `<tr><td colspan="7" class="empty-row">还没有任务，点右上角"发起新任务"创建第一个吧</td></tr>`;
      return;
    }
    document.getElementById("task-rows").innerHTML = tasks.map((t, i) => `
      <tr class="anim-in" style="animation-delay:${Math.min(i, 8) * 45}ms">
        <td>${t.name}</td><td>${t.creator}</td><td>${t.rounds}</td>
        <td>${t.dp_epsilon ?? "未启用"}</td><td>${statusBadge(t.status)}</td><td>${t.participant_count}</td>
        <td>${t.status === "recruiting" ? `<button class="secondary" onclick="showJoinTask(${t.id},'${t.name}')">加入</button>` : ""}
            ${App.user && (App.user.role === "admin" || t.creator === App.user.username) ? `<button class="secondary" onclick="location.hash='#/task/${t.id}'">详情</button>` : ""}</td>
      </tr>`).join("");
  } catch (e) { showToast(e.message, true); }
}

function showCreateTask() {
  openModal(`
    <h3 style="margin-bottom:14px;">发起新任务</h3>
    <div class="form-row"><label>任务名</label><input id="f-name"></div>
    <div class="form-row"><label>轮次</label><input id="f-rounds" type="number" value="20"></div>
    <div class="form-row"><label>DP 目标 ε（空 = 不启用 DP）</label><input id="f-eps" type="number" step="0.1"></div>
    <div class="form-row"><label>DP δ</label><input id="f-delta" type="number" value="1e-5" step="1e-6"></div>
    <div class="form-row"><label>裁剪范数 C</label><input id="f-clip" type="number" value="1.0" step="0.1"></div>
    <div class="form-row"><label>自适应裁剪</label><input id="f-adaptive" type="checkbox"></div>
    <div class="form-row"><label>本地 epochs</label><input id="f-epochs" type="number" value="1"></div>
    <div class="form-row"><label>batch size</label><input id="f-batch" type="number" value="64"></div>
    <div style="display:flex;gap:10px;justify-content:flex-end;">
      <button class="secondary" onclick="closeModal()">取消</button>
      <button onclick="doCreateTask()">创建</button>
    </div>`);
}

async function doCreateTask() {
  const body = {
    name: document.getElementById("f-name").value.trim() || "fl_task",
    rounds: parseInt(document.getElementById("f-rounds").value || "20", 10),
    dp_epsilon: document.getElementById("f-eps").value ? parseFloat(document.getElementById("f-eps").value) : null,
    dp_delta: parseFloat(document.getElementById("f-delta").value || "1e-5"),
    dp_clip: parseFloat(document.getElementById("f-clip").value || "1.0"),
    dp_adaptive_clip: document.getElementById("f-adaptive").checked,
    local_epochs: parseInt(document.getElementById("f-epochs").value || "1", 10),
    batch_size: parseInt(document.getElementById("f-batch").value || "64", 10),
  };
  try {
    const t = await api("/api/tasks", { method: "POST", body: JSON.stringify(body) });
    closeModal();
    showKeyModal(t.key);
  } catch (e) { showToast(e.message, true); }
}

function showJoinTask(id, name) {
  openModal(`
    <h3 style="margin-bottom:14px;">加入任务：${name}</h3>
    <div class="form-row"><label>密钥</label><input id="j-key" autocomplete="off"></div>
    <div class="form-row"><label>角色 ID（client_id，如 steel_ind_0 / tetouan_0）</label><input id="j-cid" autocomplete="off"></div>
    <div style="display:flex;gap:10px;justify-content:flex-end;">
      <button class="secondary" onclick="closeModal()">取消</button>
      <button onclick="doJoinTask(${id})">加入</button>
    </div>`);
}

async function doJoinTask(id) {
  try {
    const key = document.getElementById("j-key").value.trim();
    const clientId = document.getElementById("j-cid").value.trim();
    if (!key || !clientId) return showToast("请填写密钥和角色 ID", true);
    // 后端存的是 key 的 SHA-256 哈希，加入时必须先哈希再提交
    const keyHash = await sha256Hex(key);
    const r = await api(`/api/tasks/${id}/join`, {
      method: "POST",
      body: JSON.stringify({ key_hash: keyHash, client_id: clientId }),
    });
    closeModal();
    App.grpcAddr = r.grpc_addr;
    showToast("加入成功，训练通道：" + r.grpc_addr);
  } catch (e) { showToast(e.message, true); }
}

// 计算 SHA-256 十六进制（与后端 hashlib.sha256(...).hexdigest() 一致）
async function sha256Hex(str) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, "0")).join("");
}

// ===== 状态徽章 =====
function statusBadge(s) {
  const map = {
    recruiting: ["招募中", "badge-blue"],
    training: ["训练中", "badge-teal"],
    completed: ["已完成", "badge-green"],
    cancelled: ["已取消", "badge-gray"],
  };
  const [label, cls] = map[s] || [s, "badge-gray"];
  return `<span class="badge ${cls}">${label}</span>`;
}

// ===== 任务密钥弹窗（必须点"确定"才关闭） =====
function showKeyModal(key) {
  openModal(`
    <div class="key-dialog">
      <svg width="46" height="46" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="11" fill="#e4f5e9"/>
        <path d="M8 12.3l2.6 2.6L16.4 9.4" stroke="#16a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div class="key-title">任务创建成功</div>
      <p class="key-hint">密钥仅显示一次，请复制并发给参与者：</p>
      <span class="key-box" id="task-key">${key}</span>
      <p class="key-warn">密钥丢失无法找回，务必先复制保存</p>
      <button class="btn-block" onclick="copyKeyAndClose()">复制密钥并关闭</button>
    </div>`);
}

function selectKey() {
  const el = document.getElementById("task-key");
  if (!el) return;
  const range = document.createRange();
  range.selectNodeContents(el);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
}

async function copyKeyAndClose() {
  const el = document.getElementById("task-key");
  const key = el ? el.textContent.trim() : "";
  try {
    await navigator.clipboard.writeText(key);
    showToast("密钥已复制");
  } catch (e) {
    selectKey();
    showToast("请手动选中密钥复制", true);
  }
  closeModal();
  loadPlaza();
}

// ===== 弹窗 =====
function openModal(html) {
  const m = document.getElementById("modal");
  m.innerHTML = `<div class="box">${html}</div>`;
  m.classList.remove("hidden");
}
function closeModal() {
  document.getElementById("modal").classList.add("hidden");
}
document.getElementById("modal").addEventListener("click", e => {
  // 密钥弹窗必须点"复制密钥并关闭"按钮才关闭，点背景不会关
  if (e.target.id === "modal" && !document.querySelector("#modal .key-dialog")) closeModal();
});

// ===== 我的任务（Task 11） =====
function renderMyTasks() {
  document.getElementById("view").innerHTML = `
    <div class="page-head anim-in">
      <div>
        <h2>我的任务</h2>
        <p class="page-sub">我发起或参与的任务，以及本地客户端代理控制</p>
      </div>
      <button class="secondary" onclick="location.hash='#/plaza'">去广场</button>
    </div>
    <div class="card anim-in anim-d1">
      <table>
        <thead><tr><th>任务名</th><th>角色</th><th>状态</th><th>轮次</th><th>操作</th></tr></thead>
        <tbody id="my-rows"></tbody>
      </table>
    </div>
    <div class="card anim-in anim-d2">
      <h3>本地客户端代理</h3>
      <p class="page-sub" style="margin-bottom:14px;">管理数据采集与联邦训练</p>
      <div id="agent-status" style="font-family:var(--mono);color:var(--muted);font-size:13px;line-height:1.9;">检测中...</div>
      <div id="agent-actions" class="hidden" style="display:flex;gap:10px;margin-top:16px;">
        <button onclick="showCollect()">采集数据</button>
        <button onclick="doStartTrain()">开始训练</button>
      </div>
      <div id="train-status" style="margin-top:14px;font-family:var(--mono);color:var(--muted);font-size:13px;"></div>
    </div>`;
  loadMyTasks();
  loadAgentStatus();
}

async function loadMyTasks() {
  const body = document.getElementById("my-rows");
  if (!body) return;
  try {
    const tasks = await api("/api/my/tasks");
    if (!tasks.length) {
      body.innerHTML = `<tr><td colspan="5" class="empty-row">还没有参与的任务，去广场加入一个吧</td></tr>`;
      return;
    }
    body.innerHTML = tasks.map((t, i) => `
      <tr class="anim-in" style="animation-delay:${Math.min(i, 8) * 45}ms">
        <td>${t.name}</td>
        <td>${t.my_role === "creator" ? "发起者" : "参与者"}</td>
        <td>${statusBadge(t.status)}</td>
        <td>${t.current_round || 0}/${t.rounds}</td>
        <td><button class="secondary" onclick="location.hash='#/task/${t.id}'">详情</button></td>
      </tr>`).join("");
  } catch (e) {
    body.innerHTML = `<tr><td colspan="5" class="empty-row">我的任务接口暂未就绪（等待后端实现 /api/my/tasks）</td></tr>`;
  }
}

async function loadAgentStatus() {
  const el = document.getElementById("agent-status");
  const actions = document.getElementById("agent-actions");
  if (!el) return;
  try {
    const r = await fetch("/local/status");
    if (!r.ok) throw new Error("no agent");
    const s = await r.json();
    el.innerHTML = `代理在线 · server=${s.server_url}<br>` +
      `角色 ID：${s.client_id || "未设置"} · 数据：${s.data_collected ? s.dataset_id : "未采集"}`;
    if (actions) actions.classList.remove("hidden");
    // 训练状态轮询（3s）
    clearInterval(App.agentPolling);
    App.agentPolling = setInterval(async () => {
      try {
        const tr = await (await fetch("/local/train-status")).json();
        const tel = document.getElementById("train-status");
        if (tel) tel.textContent = tr.alive
          ? `训练中 · round=${tr.round} · loss=${tr.loss ?? "-"}`
          : (tr.running ? "训练结束" : "空闲");
      } catch (e) {}
    }, 3000);
  } catch (e) {
    el.textContent = "代理未连接（客户端模式需通过 run_client.bat 打开 localhost:9001）";
  }
}

function showCollect() {
  // 数据源清单：与 agent 本地兜底清单保持一致（server /api/datasets 可能未就绪）
  const datasets = [
    { id: "steel_ind_0", name: "钢厂用电（整份）", client_id: "steel_ind_0", desc: "30 分钟粒度用电负荷" },
    { id: "tetouan_0", name: "城市用电 · 区域 1", client_id: "tetouan_city_0", desc: "Tetouan 区域 1 序列" },
    { id: "tetouan_1", name: "城市用电 · 区域 2", client_id: "tetouan_city_1", desc: "Tetouan 区域 2 序列" },
    { id: "tetouan_2", name: "城市用电 · 区域 3", client_id: "tetouan_city_2", desc: "Tetouan 区域 3 序列" },
  ];
  openModal(`
    <h3>采集数据</h3>
    <p class="page-sub" style="margin-bottom:12px;">选择一个数据源下载到本地</p>
    ${datasets.map(d => `
      <div style="padding:12px 0;border-bottom:1px solid var(--line);">
        <b>${d.name}</b>
        <span style="color:var(--muted);font-size:12px;font-family:var(--mono);margin-left:6px;">${d.client_id}</span><br>
        <span style="color:var(--muted);font-size:12.5px;">${d.desc}</span><br>
        <button style="margin-top:8px;" onclick="doCollect('${d.id}')">采集</button>
      </div>`).join("")}
    <div style="margin-top:14px;text-align:right;">
      <button class="secondary" onclick="closeModal()">关闭</button>
    </div>`);
}

async function doCollect(datasetId) {
  try {
    const r = await fetch("/local/collect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "采集失败");
    closeModal();
    showToast(`采集完成：${d.rows} 条 · ${d.time_range[0]} ~ ${d.time_range[1]} · 缺失率 ${d.missing_rate * 100}%`);
    loadAgentStatus();
  } catch (e) { showToast(e.message, true); }
}

async function doStartTrain() {
  if (!App.grpcAddr) return showToast("请先在广场加入任务（会获得训练通道地址）", true);
  try {
    const r = await fetch("/local/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grpc_addr: App.grpcAddr }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "启动失败");
    showToast(d.message || "训练已启动");
    loadAgentStatus();
  } catch (e) { showToast(e.message, true); }
}

function renderAdmin() {
  document.getElementById("view").innerHTML = `
    <div class="card">
      <h3 style="margin-bottom:12px;">管理大屏 - 全部任务</h3>
      <table><thead><tr>
        <th>ID</th><th>任务名</th><th>发起者</th><th>状态</th><th>轮次</th>
        <th>参与</th><th>操作</th>
      </tr></thead><tbody id="admin-rows"></tbody></table>
    </div>`;
  api("/api/tasks").then(tasks => {
    const rows = document.getElementById("admin-rows");
    if (!tasks.length) {
      rows.innerHTML = `<tr><td colspan="7" class="empty-row">还没有任务</td></tr>`;
      return;
    }
    rows.innerHTML = tasks.map(t => `
      <tr>
        <td>${t.id}</td><td>${t.name}</td><td>${t.creator}</td><td>${statusBadge(t.status)}</td>
        <td>${t.current_round || 0}/${t.rounds}</td><td>${t.participant_count}</td>
        <td><button class="secondary" onclick="location.hash='#/task/${t.id}'">进入大屏</button></td>
      </tr>`).join("");
  }).catch(e => showToast(e.message, true));
}

// ===== 任务详情大屏（Task 8：6 图 + 轮询刷新）=====
const charts = {};

function renderTaskDetail(id) {
  document.getElementById("view").innerHTML = `
    <button class="secondary" style="margin-bottom:12px;" onclick="location.hash='#/plaza'">返回</button>
    <div id="detail-body">加载中...</div>`;
  loadTaskDetail(id);
  clearInterval(App.polling);
  App.polling = setInterval(() => loadTaskDetail(id), 2500);
}

async function loadTaskDetail(id) {
  try {
    const [task, audit, rc] = await Promise.all([
      api(`/api/tasks/${id}`),
      api(`/api/tasks/${id}/audit`),
      api(`/api/tasks/${id}/rc-results`).catch(() => []),
    ]);
    const body = document.getElementById("detail-body");
    if (!body) return;
    // 首次进入才重建 DOM，后续轮询只更新 stat + 图表（避免闪烁）
    if (body.getAttribute("data-built") !== "1") {
      body.setAttribute("data-built", "1");
      body.innerHTML = `
        <div class="stat-cards">
          <div class="stat"><div class="num" id="st-status">${task.status}</div><div class="lbl">状态</div></div>
          <div class="stat"><div class="num" id="st-round">${task.current_round || 0}/${task.rounds}</div><div class="lbl">轮次</div></div>
          <div class="stat"><div class="num" id="st-part">${task.participant_count}</div><div class="lbl">参与人数</div></div>
          <div class="stat"><div class="num" id="st-eps">${task.dp_epsilon ?? "无"}</div><div class="lbl">DP ε</div></div>
        </div>
        ${App.user && (App.user.role === "admin" || task.creator === App.user.username) ? `
        <div style="margin:14px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
          <button onclick="doStartTask(${task.id})">开始训练</button>
          <span style="color:var(--muted);font-size:12.5px;">启动联邦训练，需所有参与方已加入并在线</span>
        </div>` : ""}
        ${App.mode === "client" ? `
        <div style="margin:14px 0;">
          <button onclick="doTrainRc(${task.id})">训练残差修正器（二阶段）</button>
          <span style="color:var(--muted);font-size:12.5px;margin-left:10px;">在本地用 RC 残差修正器微调，上传 WAPE 与对比图</span>
        </div>` : ""}
        <div class="grid2">
          <div class="card"><h4>每轮参与人数</h4><div id="ch-participants" class="chart"></div></div>
          <div class="card"><h4>参与热力图（绿=参与 红=掉线）</h4><div id="ch-heatmap" class="chart"></div></div>
          <div class="card"><h4>每客户端累计 ε</h4><div id="ch-eps" class="chart"></div></div>
          <div class="card"><h4>全局 loss</h4><div id="ch-loss" class="chart"></div></div>
          <div class="card"><h4>自适应裁剪阈值 C</h4><div id="ch-clip" class="chart"></div></div>
          <div class="card"><h4>RC 结果（WAPE %）</h4><div id="ch-rc" class="chart"></div><div id="rc-imgs"></div></div>
        </div>`;
    } else {
      const set = (nid, v) => { const el = document.getElementById(nid); if (el) el.textContent = v; };
      set("st-status", task.status);
      set("st-round", `${task.current_round || 0}/${task.rounds}`);
      set("st-part", task.participant_count);
      set("st-eps", task.dp_epsilon ?? "无");
    }
    renderCharts(audit, rc);
  } catch (e) { showToast(e.message, true); }
}

function renderCharts(audit, rc) {
  const rounds = audit.map(a => a.round);
  // 1. 每轮参与人数
  setChart("ch-participants", {
    xAxis: { type: "category", data: rounds },
    yAxis: { type: "value", minInterval: 1 },
    series: [{ type: "line", data: audit.map(a => a.joined.length),
               name: "实际参与", areaStyle: {} },
             { type: "line", data: audit.map(a => a.expected.length),
               name: "应参与", lineStyle: { type: "dashed" } }],
  });
  // 2. 参与热力图（绿=1 参与 / 红=0 掉线）
  const clients = [...new Set(audit.flatMap(a => a.expected))];
  const heat = [];
  audit.forEach((a, ri) => {
    clients.forEach((cid, ci) => {
      if (a.expected.includes(cid)) {
        heat.push([ri, ci, a.joined.includes(cid) ? 1 : 0]);
      }
    });
  });
  setChart("ch-heatmap", {
    tooltip: {},
    xAxis: { type: "category", data: rounds },
    yAxis: { type: "category", data: clients },
    visualMap: { min: 0, max: 1, show: false,
                 inRange: { color: ["#e74c3c", "#2ca02c"] } },
    series: [{ type: "heatmap", data: heat }],
  });
  // 3. 每客户端累计 ε（A 已改为每轮增量上报，前端逐轮累加得到累计）
  const epsSeries = clients.map(cid => ({
    name: cid, type: "line",
    data: audit.map((a, i) => Number((audit.slice(0, i + 1)
      .reduce((s, x) => s + ((x.client_epsilons || {})[cid] || 0), 0)).toFixed(3))),
  }));
  setChart("ch-eps", {
    xAxis: { type: "category", data: rounds },
    yAxis: { type: "value", name: "累计 ε" },
    series: epsSeries,
  });
  // 4. 全局 loss
  setChart("ch-loss", {
    xAxis: { type: "category", data: rounds },
    yAxis: { type: "value" },
    series: [{ type: "line", data: audit.map(a => a.loss),
               name: "全局 loss", areaStyle: {} }],
  });
  // 5. 自适应裁剪阈值 C
  setChart("ch-clip", {
    xAxis: { type: "category", data: rounds },
    yAxis: { type: "value" },
    series: [{ type: "line", data: audit.map(a => a.clip_norm),
               name: "C", step: "end" }],
  });
  // 6. RC WAPE 对比（全局 vs 全局+RC）
  setChart("ch-rc", {
    xAxis: { type: "category", data: rc.map(r => r.client_id) },
    yAxis: { type: "value" },
    series: [{ type: "bar", name: "全局模型", data: rc.map(r => r.wape_global) },
             { type: "bar", name: "全局+RC", data: rc.map(r => r.wape_rc) }],
  });
  document.getElementById("rc-imgs").innerHTML = rc.map(r =>
    r.png_url ? `<div style="margin-top:10px;"><b>${r.client_id}</b><br>
      <img class="rc-img" src="${r.png_url}" alt="${r.client_id} 对比图"></div>` : ""
  ).join("");
}

function setChart(id, option) {
  const el = document.getElementById(id);
  if (!el) return;
  let chart = charts[id];
  if (!chart || chart.getDom() !== el) {
    if (chart) chart.dispose();
    chart = echarts.init(el);
    charts[id] = chart;
    window.addEventListener("resize", () => chart.resize());
  }
  chart.setOption({ ...option, tooltip: { trigger: "axis" },
                    legend: { show: true, top: 0 } }, true);
}

// ===== 二阶段 RC 触发（Task 13，仅客户端模式）=====
async function doTrainRc(taskId) {
  try {
    const r = await fetch("/local/rc", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "RC 训练失败");
    const has = (v) => v !== null && v !== undefined;
    const hint = (has(d.wape_global) && has(d.wape_rc))
      ? `RC 完成：WAPE ${d.wape_global}% → ${d.wape_rc}%`
      : "RC 已提交，结果已上传到服务端";
    showToast(hint);
    loadTaskDetail(taskId);
  } catch (e) { showToast(e.message, true); }
}

// ===== Server 端开始训练（创建者/管理员触发）=====
async function doStartTask(taskId) {
  try {
    const r = await api(`/api/tasks/${taskId}/start`, { method: "POST" });
    showToast(r.message || "训练已开始");
    loadTaskDetail(taskId);
  } catch (e) { showToast(e.message, true); }
}
