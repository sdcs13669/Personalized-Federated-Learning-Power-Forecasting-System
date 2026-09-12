// ===== 登录/注册 =====
function renderLogin() {
  document.body.classList.add("login-page");
  document.getElementById("topbar").classList.add("hidden");
  document.getElementById("view").innerHTML = `
    <div class="login-page">
      <canvas id="bg-canvas" class="bg-canvas" aria-hidden="true"></canvas>
      <canvas id="net-canvas" class="net-canvas" aria-hidden="true"></canvas>
      <div class="login-layout">
        <div class="login-hero anim-in">
          <div class="brand-line">
            <span class="brand-en">NEXUSGRID</span>
            <span class="brand-cn">云枢启电</span>
          </div>
          <h1 class="hero-title">数据留在本地，<br>智能协同生长。</h1>
          <p class="hero-desc">联邦学习 · 差分隐私 · 电力负荷预测<br>隐私不出的多机协同训练平台</p>
          <div class="hero-foot">
            <span class="hero-dot" aria-hidden="true"></span>
            <span id="ns-text">FL-NET v2.13 · 联邦节点 04 · ε 审计开启</span>
          </div>
        </div>
        <div class="auth-card anim-in anim-d1">
          <div class="auth-head">
            <span class="status-dot" aria-hidden="true"></span>
            <h2 class="auth-title">登录</h2>
            <p class="auth-sub">进入联邦学习平台</p>
          </div>
          <div class="form-row"><label for="login-user">用户名</label><input id="login-user" autocomplete="username" placeholder="请输入用户名"></div>
          <div class="form-row"><label for="login-pass">密码</label><input id="login-pass" type="password" autocomplete="current-password" placeholder="请输入密码"></div>
          <button class="btn-block" onclick="doLogin()">登 录</button>
          <button class="secondary btn-block" onclick="doRegister()">注册新账号</button>
        </div>
      </div>
    </div>`;
  initParticleNet();
  initBgFlow();
}

// ===== 月之暗面式流动光斑背景（Canvas 绘制，随时间漂移+呼吸） =====
let _bgFlowHandle = null;
function initBgFlow() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  if (_bgFlowHandle) { cancelAnimationFrame(_bgFlowHandle); _bgFlowHandle = null; }
  const ctx = canvas.getContext("2d");
  let w, h;
  const blobs = [
    { bx: .12, by: .16, r: .50, c: [20, 184, 166], s: .50, p: 0, ph: 0 },
    { bx: .85, by: .12, r: .55, c: [59, 130, 246], s: .40, p: 1.3, ph: 2 },
    { bx: .50, by: .90, r: .60, c: [167, 139, 250], s: .60, p: 2.6, ph: 4 },
    { bx: .92, by: .70, r: .40, c: [217, 119, 6], s: .35, p: 3.9, ph: 1 },
    { bx: .28, by: .78, r: .34, c: [20, 184, 166], s: .70, p: 5, ph: 3 },
    { bx: .72, by: .50, r: .44, c: [59, 130, 246], s: .55, p: 6, ph: 5 },
  ];
  function resize() {
    w = canvas.width = canvas.clientWidth;
    h = canvas.height = canvas.clientHeight;
  }
  resize();
  window.addEventListener("resize", resize);
  const t0 = performance.now();
  function tick(now) {
    const t = (now - t0) / 1000;
    ctx.clearRect(0, 0, w, h);
    ctx.globalCompositeOperation = "lighter";
    for (const b of blobs) {
      // 双正弦组合轨迹 → 像光影缓慢流动
      const x = (b.bx + Math.sin(t * b.s + b.p) * .09 + Math.sin(t * b.s * .6 + b.ph) * .04) * w;
      const y = (b.by + Math.cos(t * b.s * .8 + b.p) * .09 + Math.cos(t * b.s * .5 + b.ph) * .04) * h;
      const r = b.r * Math.min(w, h) * (1 + Math.sin(t * .35 + b.ph) * .15);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, `rgba(${b.c[0]},${b.c[1]},${b.c[2]},.32)`);
      g.addColorStop(.55, `rgba(${b.c[0]},${b.c[1]},${b.c[2]},.10)`);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.globalCompositeOperation = "source-over";
    _bgFlowHandle = requestAnimationFrame(tick);
  }
  _bgFlowHandle = requestAnimationFrame(tick);
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
          ctx.strokeStyle = "rgba(20,184,166," + alpha.toFixed(3) + ")";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    for (const p of parts) {
      ctx.fillStyle = "rgba(20,184,166,.55)";
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
  if (_bgFlowHandle) { cancelAnimationFrame(_bgFlowHandle); _bgFlowHandle = null; }
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
  // 任务发起入口统一收敛到 Server 端：客户端模式不提供"发起新任务"，只能凭密钥加入
  const isServer = App.mode !== "client";
  document.getElementById("view").innerHTML = `
    <div class="page-head anim-in">
      <div>
        <h2>任务广场</h2>
        <p class="page-sub">${isServer
          ? "浏览可加入的联邦学习任务，或发起新任务"
          : "浏览可加入的联邦学习任务，凭服务端下发的密钥加入"}</p>
      </div>
      ${isServer ? `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        ${App.user && App.user.role === "admin"
          ? `<button class="danger" onclick="doPurgeTasks()">清理历史任务</button>` : ""}
        <button onclick="showCreateTask()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>
        发起新任务
      </button></div>` : ""}
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
        `<tr><td colspan="7" class="empty-row">${App.mode === "client"
          ? "还没有任务，请等待服务端发起后凭密钥加入"
          : "还没有任务，点右上角“发起新任务”创建第一个吧"}</td></tr>`;
      return;
    }
    document.getElementById("task-rows").innerHTML = tasks.map((t, i) => `
      <tr class="anim-in" style="animation-delay:${Math.min(i, 8) * 45}ms">
        <td>${t.name}</td><td>${t.creator}</td><td>${t.rounds}</td>
        <td>${t.dp_epsilon ?? "未启用"}</td><td>${statusBadge(t.status)}</td><td>${t.participant_count}</td>
        <td>${t.status === "recruiting" ? `<button class="secondary" onclick="showJoinTask(${t.id},'${t.name}')">加入</button>` : ""}
            ${App.user && (App.user.role === "admin" || t.creator === App.user.username) ? `
              <button class="secondary" onclick="location.hash='#/task/${t.id}'">详情</button>
              <button class="danger" style="margin-left:6px;" onclick="doDeleteTask(${t.id}, '${t.name}')">删除</button>` : ""}</td>
      </tr>`).join("");
  } catch (e) { showToast(e.message, true); }
}

function showCreateTask() {
  // 统一入口：任务只由 Server 端发起，客户端模式直接拦截（双保险）
  if (App.mode === "client") {
    showToast("任务由服务端统一发起，客户端请凭密钥加入", true);
    return;
  }
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
    <div class="form-row"><label>角色 ID（client_id，如 steel_ind_0 / tetouan_0）</label><input id="j-cid" autocomplete="off" readonly></div>
    <p class="page-sub" id="j-cid-hint">正在读取本机客户端身份…</p>
    <div style="display:flex;gap:10px;justify-content:flex-end;">
      <button class="secondary" onclick="closeModal()">取消</button>
      <button onclick="doJoinTask(${id})">加入</button>
    </div>`);
  // 客户端模式自动填入本机 client_id：多机多开时手打 ID 极易打错，
  // 一旦 client_id 与 agent 配置不符，采集/训练的数据目录就对不上。
  const cidEl = document.getElementById("j-cid");
  const hintEl = document.getElementById("j-cid-hint");
  if (!cidEl || !hintEl) return;
  if (App.mode !== "client") {
    cidEl.removeAttribute("readonly");
    hintEl.textContent = "服务端模式：请填写本次加入所使用的角色 ID";
    return;
  }
  fetch("/local/status").then(r => r.json()).then(s => {
    if (s && s.client_id) {
      cidEl.value = s.client_id;
      const ds = s.dataset_id ? `，本机已采集数据集 ${s.dataset_id}`
                              : "，本机尚未采集数据";
      hintEl.textContent = `已按本机配置自动填入，请勿修改${ds}`;
    } else {
      cidEl.removeAttribute("readonly");
      hintEl.textContent = "未能读取本机 client_id，请手动填写";
    }
  }).catch(() => {
    cidEl.removeAttribute("readonly");
    hintEl.textContent = "未能读取本机 client_id，请手动填写";
  });
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
    stopped: ["已停止", "badge-gray"],
    failed: ["失败", "badge-red"],
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
        if (tel) {
          if (tr.alive) {
            // 已连上但还没跑完：显示轮次与 loss；若中途有过连接告警也带上
            tel.textContent = `训练中 · round=${tr.round} · loss=${tr.loss ?? "-"}`;
            tel.style.color = "";
          } else if (tr.error) {
            // 连不上训练通道（例如服务端还没点"开始训练"）—— 以前这里完全
            // 静默，界面上只表现为"这个客户端掉线了"，无从排查
            tel.textContent = `未连接训练通道：${tr.error}`;
            tel.style.color = "var(--danger)";
          } else {
            tel.textContent = tr.running ? "训练结束" : "空闲";
            tel.style.color = "";
          }
        }
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
    { id: "lcl_res_0", name: "居民社区用电 · 早期投运", client_id: "lcl_res_0", desc: "6 个社区聚合负荷（相邻 30 户聚合）" },
    { id: "lcl_res_1", name: "居民社区用电 · 后期投运", client_id: "lcl_res_1", desc: "5 个社区聚合负荷（相邻 30 户聚合）" },
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
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px;">
        <h3 style="margin:0;">管理大屏 - 全部任务</h3>
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
          <button class="danger" onclick="doPurgeTasks()">清理历史任务</button>
          <span style="color:var(--muted);font-size:12.5px;">录制/演示前用：清掉已结束的任务（训练中的自动跳过）</span>
        </div>
      </div>
      <table><thead><tr>
        <th>ID</th><th>任务名</th><th>发起者</th><th>状态</th><th>轮次</th>
        <th>参与</th><th>操作</th>
      </tr></thead><tbody id="admin-rows"></tbody></table>
    </div>`;
  loadAdminRows();
}

function loadAdminRows() {
  api("/api/tasks").then(tasks => {
    const rows = document.getElementById("admin-rows");
    if (!rows) return;
    if (!tasks.length) {
      rows.innerHTML = `<tr><td colspan="7" class="empty-row">还没有任务</td></tr>`;
      return;
    }
    rows.innerHTML = tasks.map(t => `
      <tr>
        <td>${t.id}</td><td>${t.name}</td><td>${t.creator}</td><td>${statusBadge(t.status)}</td>
        <td>${t.current_round || 0}/${t.rounds}</td><td>${t.participant_count}</td>
        <td>
          <button class="secondary" onclick="location.hash='#/task/${t.id}'">进入大屏</button>
          <button class="danger" style="margin-left:6px;" onclick="doDeleteTask(${t.id}, '${t.name}')">删除</button>
        </td>
      </tr>`).join("");
  }).catch(e => showToast(e.message, true));
}

// 删除/清理后按"当前所在页面"刷新对应列表（广场 or 管理大屏）
function refreshTaskList() {
  if ((location.hash || "").includes("/admin")) loadAdminRows();
  else loadPlaza();
}

// ===== 删除单个任务（创建者/管理员）=====
async function doDeleteTask(taskId, taskName) {
  if (!confirm(`确定删除任务「${taskName}」(ID ${taskId})？\n\n` +
               `· 该任务的审计记录、参与记录、RC 结果会一并删除\n` +
               `· 已保存的模型文件与训练日志也会删除\n` +
               `· 训练中的任务需要先「强制停止训练」`)) return;
  try {
    const r = await api(`/api/tasks/${taskId}`, { method: "DELETE" });
    showToast(r.message || "任务已删除");
    refreshTaskList();
  } catch (e) { showToast(e.message, true); }
}

// ===== 一键清理历史任务（录制前把界面清干净）=====
async function doPurgeTasks() {
  if (!confirm("清理所有已结束的历史任务？\n\n" +
               "· 范围：已完成 / 失败 / 已停止 / 已取消 / 卡死的训练残留\n" +
               "· 「招募中」的任务会保留\n" +
               "· 真正在训练的任务自动跳过\n\n" +
               "该操作不可撤销，确定继续？")) return;
  try {
    const r = await api("/api/tasks/purge",
                        { method: "POST", body: JSON.stringify({ include_recruiting: false }) });
    showToast(r.message || "已清理历史任务");
    refreshTaskList();
  } catch (e) { showToast(e.message, true); }
}

// ===== 任务详情大屏（Task 8：6 图 + 轮询刷新）=====
const charts = {};

function renderTaskDetail(id) {
  document.getElementById("view").innerHTML = `
    <button class="secondary" style="margin-bottom:12px;" onclick="location.hash='#/plaza'">返回</button>
    <div id="detail-body">加载中...</div>`;
  clearInterval(App.polling);
  if (App.mode === "client") {
    // ===== 客户端三阶段页（need.md）：训练过程 → 二阶段 → 预测展示 =====
    loadClientTaskDetail(id);
    // 训练中每 5s 刷新阶段1；二阶段/预测阶段由按钮/状态切换，不轮询
    App.polling = setInterval(() => loadClientTaskDetail(id, true), 5000);
  } else {
    // ===== 管理端大屏（6 图）=====
    loadTaskDetail(id);
    App.polling = setInterval(() => loadTaskDetail(id), 4000);
  }
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
        <div style="margin:14px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;" id="task-actions"></div>` : ""}
        ${App.mode === "client" ? `
        <div style="margin:14px 0;">
          <button onclick="doTrainRc(${task.id})">训练残差修正器（二阶段）</button>
          <span style="color:var(--muted);font-size:12.5px;margin-left:10px;">在本地用 RC 残差修正器微调，上传 WAPE 与对比图</span>
        </div>` : ""}
        <div class="grid2">
          <div class="card"><h4>每轮参与人数</h4><div id="ch-participants" class="chart"></div></div>
          <div class="card"><h4>全局 loss</h4><div id="ch-loss" class="chart"></div></div>
        </div>
        <div class="card">
          <h4>审计明细（逐轮留痕 · 可离线复核）</h4>
          <p class="page-sub" style="margin:6px 0;">补两条曲线看不到的明细：本轮究竟是谁掉线、自适应裁剪阈值、各参与方累计隐私预算。逐轮落库、第三方可离线复核 —— 这就是"可溯可审计"。</p>
          <div style="overflow-x:auto;"><table id="audit-table"></table></div>
        </div>
        <div class="card"><h4>RC 客户端对比图</h4><div id="rc-imgs"></div></div>`;
    } else {
      const set = (nid, v) => { const el = document.getElementById(nid); if (el) el.textContent = v; };
      set("st-status", task.status);
      set("st-round", `${task.current_round || 0}/${task.rounds}`);
      set("st-part", task.participant_count);
      set("st-eps", task.dp_epsilon ?? "无");
    }
    renderTaskActions(task);
    renderCharts(audit, rc);
  } catch (e) { showToast(e.message, true); }
}

// 统一样式：所有图表必须标明横纵坐标含义（评审要求，明确表达训练轮次/参与人数等）
const CHART_GRID = { left: 76, right: 76, top: 46, bottom: 54, containLabel: true };
const AXIS_NAME_TEXT = { fontSize: 12, color: "#475569" };
const axisX = (name) => ({ name, nameLocation: "middle", nameGap: 30,
                           nameTextStyle: AXIS_NAME_TEXT });
const axisY = (name) => ({ name, nameLocation: "middle", nameGap: 56,
                           nameRotate: 90, nameTextStyle: AXIS_NAME_TEXT });

function renderCharts(audit, rc) {
  const rounds = audit.map(a => a.round);
  // 1. 每轮参与人数
  setChart("ch-participants", {
    grid: CHART_GRID,
    xAxis: { type: "category", data: rounds, ...axisX("训练轮次") },
    yAxis: { type: "value", minInterval: 1, ...axisY("参与人数") },
    series: [{ type: "line", data: audit.map(a => a.joined.length),
               name: "实际参与", areaStyle: {} },
             { type: "line", data: audit.map(a => a.expected.length),
               name: "应参与", lineStyle: { type: "dashed" } }],
  });
  // 2. 全局 loss
  setChart("ch-loss", {
    grid: CHART_GRID,
    xAxis: { type: "category", data: rounds, ...axisX("训练轮次") },
    yAxis: { type: "value", ...axisY("全局模型损失") },
    series: [{ type: "line", data: audit.map(a => a.loss),
               name: "全局 loss", areaStyle: {} }],
  });
  // 3. 客户端上传的二阶段 RC 对比图（属图片证据，非图表，予以保留）
  const imgs = document.getElementById("rc-imgs");
  if (imgs) {
    imgs.innerHTML = rc.map(r =>
      r.png_url ? `<div style="margin-top:10px;"><b>${r.client_id}</b><br>
        <img class="rc-img" src="${r.png_url}" alt="${r.client_id} 对比图"></div>` : ""
    ).join("") || `<p class="page-sub">暂无客户端上传的对比图。</p>`;
  }
  // 4. 逐轮审计明细（表格，非图表）
  renderAuditTable("audit-table", audit);
}

// ===== 审计明细表（管理端全量视角）=====
// 只承载「两条曲线看不到的明细」，不与图表重复：
//   · 掉线名单（图只给个数，给不出是谁）
//   · 自适应裁剪阈值 C（该曲线已删，此处是唯一入口）
//   · 各参与方累计 ε（服务端 ε 曲线已删，此处是唯一入口）
function renderAuditTable(elId, audit) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!audit || !audit.length) {
    el.innerHTML = `<tr><td colspan="3" class="empty-row">暂无审计记录，训练开始后逐轮写入。</td></tr>`;
    return;
  }
  const clients = [...new Set(audit.flatMap(a => a.expected || []))];
  const cum = {};
  const head = `<tr>
    <th>轮次</th><th>掉线名单</th><th>裁剪阈值 C</th>
    ${clients.map(c => `<th>累计 ε<br><span style="font-weight:400;color:var(--muted);">${c}</span></th>`).join("")}
  </tr>`;
  const rows = audit.map(a => {
    const eps = a.client_epsilons || {};
    clients.forEach(c => { cum[c] = (cum[c] || 0) + (eps[c] || 0); });
    const dropped = a.dropped || [];
    const droppedCell = dropped.length
      ? `<span class="badge badge-red">${dropped.join("、")}</span>`
      : `<span class="badge badge-green">无</span>`;
    const c = (a.clip_norm === null || a.clip_norm === undefined)
      ? "—" : fmtNum(a.clip_norm);
    return `<tr>
      <td>${a.round}</td>
      <td>${droppedCell}</td>
      <td>${c}</td>
      ${clients.map(cid => `<td>${(cum[cid] || 0).toFixed(4)}</td>`).join("")}
    </tr>`;
  }).join("");
  el.innerHTML = head + rows;
}

// ===== 审计明细表（客户端自身视角，仅本客户端）=====
// 与阶段1两条曲线不重复：只给「本方逐轮参与状态」与逐轮精确的累计 ε
// （曲线给不出 4 位小数的精确值，也无法说明某轮本方是否在列）。
function renderMyAuditTable(elId, audit, myCid) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!audit || !audit.length || !myCid) {
    el.innerHTML = `<tr><td colspan="3" class="empty-row">暂无审计记录，训练开始后逐轮写入。</td></tr>`;
    return;
  }
  let cum = 0;
  const rows = audit.map(a => {
    const eps = (a.client_epsilons || {})[myCid] || 0;
    cum += eps;
    const joined = (a.joined || []).includes(myCid);
    const expected = (a.expected || []).includes(myCid);
    const state = joined
      ? `<span class="badge badge-green">本轮已参与</span>`
      : (expected ? `<span class="badge badge-red">本轮掉线</span>`
                  : `<span class="badge badge-gray">未在本轮</span>`);
    return `<tr>
      <td>${a.round}</td>
      <td>${state}</td>
      <td>${cum.toFixed(4)}</td>
    </tr>`;
  }).join("");
  el.innerHTML = `<tr><th>轮次</th><th>本方参与状态</th><th>我的累计 ε</th></tr>` + rows;
}

function setChart(id, option) {
  const el = document.getElementById(id);
  if (!el) return;
  let chart = charts[id];
  const first = (!chart || chart.getDom() !== el);
  if (first) {
    if (chart) chart.dispose();
    chart = echarts.init(el);
    charts[id] = chart;
    window.addEventListener("resize", () => chart.resize());
  }
  // 只有首次绘制带动画；轮询刷新时关闭动画，避免每几秒重放动画导致页面卡顿
  chart.setOption({ ...option, animation: first,
                    tooltip: { trigger: "axis" },
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

// ===== Server 端强制停止训练（创建者/管理员触发）=====
async function doStopTask(taskId) {
  if (!confirm("确定强制停止该任务的训练？\n\n" +
               "· 当前轮次进度会丢失\n" +
               "· worker 进程将被终止，8089 端口释放\n" +
               "· 任务状态将变为「已停止」")) return;
  try {
    const r = await api(`/api/tasks/${taskId}/stop`, { method: "POST" });
    showToast(r.message || "训练已停止");
    loadTaskDetail(taskId);
  } catch (e) { showToast(e.message, true); }
}

// 按任务最新状态渲染操作按钮（每次轮询调用，状态变化自动切换）
function renderTaskActions(task) {
  const box = document.getElementById("task-actions");
  if (!box) return; // 非创建者/管理员，无操作区
  if (task.status === "recruiting") {
    box.innerHTML = `
      <button onclick="doStartTask(${task.id})">开始训练</button>
      <span style="color:var(--muted);font-size:12.5px;">启动后服务端会自动等待已加入的参与方接入并开始训练，无需人工盯守；每轮参与与掉线情况均写入审计日志，可追溯</span>`;
  } else if (task.status === "training") {
    box.innerHTML = `
      <button class="danger" onclick="doStopTask(${task.id})">强制停止训练</button>
      <span style="color:var(--danger);font-size:12.5px;">终止当前训练进程并释放 8089 端口，任务状态将变为「已停止」</span>`;
  } else {
    box.innerHTML = ""; // 终态任务不显示操作按钮
  }
}

// =====================================================================
// 客户端三阶段页（need.md）
// 阶段1 训练过程：只看自身 ε 累计 + 本地 TCN loss
// 阶段2 二阶段：一阶段完成后出现"开启二阶段"，纯本地训练，展示本地修正器 loss
// 阶段3 预测展示：本地测试集预测，动态图 + 静态图 + 文本
// =====================================================================
const _clientPhase = { stage2: "idle", stage3: "idle" }; // 本客户端对当前任务的本地阶段

async function loadClientTaskDetail(id, isRefresh) {
  // 1) 探测本客户端身份 client_id
  let myCid = "";
  try { const s = await (await fetch("/local/status")).json(); myCid = s.client_id || ""; }
  catch (e) {}
  // 2) 拉任务与审计
  try {
    const task = await api(`/api/tasks/${id}`);
    const audit = await api(`/api/tasks/${id}/audit`).catch(() => []);
    // 3) 拉 agent 本地阶段状态（stage2 是否已开启、stage3 预测数据是否就绪）
    let st = {};
    try { st = await (await fetch("/local/stage-status?task_id=" + id)).json(); }
    catch (e) { st = {}; }
    const s2 = st.stage2 || "idle";   // idle | running | done
    const s3 = st.stage3 || "idle";   // idle | ready
    const body = document.getElementById("detail-body");
    if (!body) return;
    const built = body.getAttribute("data-built") === "1";
    if (!built) body.setAttribute("data-built", "1");
    renderClientTask(body, built, { task, audit, myCid, s2, s3, st });
  } catch (e) {
    const body = document.getElementById("detail-body");
    if (body) body.innerHTML = `<div class="card" style="color:var(--danger);">加载失败：${e.message}</div>`;
  }
}

function renderClientTask(body, built, ctx) {
  const { task, audit, myCid, s2, s3 } = ctx;
  const rounds = audit.map(a => a.round);
  // 自己的 ε 累计 + 本地 loss
  const cumEps = [], myLoss = [];
  let acc = 0;
  for (const a of audit) {
    const e = (a.client_epsilons && a.client_epsilons[myCid]) || 0;
    acc += e;
    cumEps.push(Number(acc.toFixed(4)));
    const l = a.client_losses && a.client_losses[myCid];
    myLoss.push(l === undefined || l === null ? null : Number(l.toFixed(6)));
  }
  // 一阶段是否结束（进入 recruiting/training 之外的终态即认为一阶段结束）
  const phase1Done = (task.status === "completed" || task.status === "cancelled");
  // 当前阶段：按 一阶段 → 二阶段 → 三阶段 的真实状态推导（随轮询自动更新）
  const phaseLabel = !phase1Done ? "阶段1·训练中"
    : s2 === "running" ? "阶段2·训练中"
    : s2 === "failed" ? "阶段2失败（可重试）"
    : s2 === "done" ? (s3 === "running" ? "阶段3·生成中"
                       : s3 === "ready" ? "阶段3完成" : "阶段2完成")
    : "阶段1完成（待开启二阶段）";

  if (!built) {
    body.innerHTML = `
      <div class="stat-cards">
        <div class="stat"><div class="num" id="ct-status">${task.status}</div><div class="lbl">状态</div></div>
        <div class="stat"><div class="num" id="ct-round">${task.current_round || 0}/${task.rounds}</div><div class="lbl">轮次</div></div>
        <div class="stat"><div class="num" id="ct-cid" style="font-size:16px;">${myCid || "未知"}</div><div class="lbl">本客户端</div></div>
        <div class="stat"><div class="num" id="ct-phase" style="font-size:18px;">${phaseLabel}</div><div class="lbl">当前阶段</div></div>
      </div>
      <div class="grid2">
        <div class="card"><h4>我的隐私预算 ε 累计</h4><p class="page-sub" style="margin-bottom:8px;">仅本客户端 ${myCid || ""}</p><div id="ch-self-eps" class="chart"></div></div>
        <div class="card"><h4>本地 TCN 模型 loss</h4><p class="page-sub" style="margin-bottom:8px;">仅本客户端 ${myCid || ""}</p><div id="ch-self-loss" class="chart"></div></div>
      </div>
      <div class="card">
        <h4>我的审计记录</h4>
        <p class="page-sub" style="margin:6px 0;">本方逐轮的参与状态与累计隐私预算，均可在本机复核（仅本客户端 ${myCid || ""}）。</p>
        <div style="overflow-x:auto;"><table id="my-audit-table"></table></div>
      </div>
      <div id="ct-stage2"></div>
      <div id="ct-stage3"></div>`;
  }
  // 状态卡片统一更新（"当前阶段"随二/三阶段推进自动变化）
  const setStat = (n, v) => { const el = document.getElementById(n); if (el) el.textContent = v; };
  setStat("ct-status", task.status);
  setStat("ct-round", `${task.current_round || 0}/${task.rounds}`);
  setStat("ct-phase", phaseLabel);
  // 阶段1：本方审计明细表 + 两条曲线
  renderMyAuditTable("my-audit-table", audit, myCid);
  setChart("ch-self-eps", {
    grid: CHART_GRID,
    xAxis: { type: "category", data: rounds.length ? rounds : [0], ...axisX("训练轮次") },
    yAxis: { type: "value", ...axisY("累计隐私预算 ε") },
    series: [{ type: "line", data: rounds.length ? cumEps : [],
               name: myCid, areaStyle: {}, smooth: true }],
  });
  setChart("ch-self-loss", {
    grid: CHART_GRID,
    xAxis: { type: "category", data: rounds.length ? rounds : [0], ...axisX("训练轮次") },
    yAxis: { type: "value", ...axisY("本地模型损失") },
    series: [{ type: "line", data: rounds.length ? myLoss : [],
               name: "本地 loss", areaStyle: {}, smooth: true, connectNulls: true }],
  });
  // 阶段2 / 阶段3 容器由下述函数根据本地状态填充
  renderStage2Area(body, ctx);
  renderStage3Area(body, ctx);
}

function renderStage2Area(body, ctx) {
  const el = document.getElementById("ct-stage2");
  if (!el) return;
  const { task, s2, s3, st } = ctx;
  const done1 = (task.status === "completed" || task.status === "cancelled");
  if (!done1) { el.innerHTML = ""; el.dataset.key = ""; return; }
  if (s2 === "running") {
    const key = "running";
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      el.innerHTML = `<div class="card"><h4>二阶段 · 本地个性化修正器训练中…</h4>
        <p class="page-sub" style="margin-top:6px;">纯本地训练（不上传服务器），完成后自动刷新。</p></div>`;
    }
    return;
  }
  if (s2 === "failed") {
    const key = "failed";
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      el.innerHTML = `<div class="card"><h4 style="color:var(--danger);">二阶段失败</h4>
        <p class="page-sub">${(st && st.error) || "未知错误，查看 agent 窗口日志"}</p>
        <button style="margin-top:10px;" onclick="startStage2(${task.id})">重试</button></div>`;
    }
    return;
  }
  if (s2 === "done") {
    const arch = (st && st.corrector_arch) || "";
    const key = "done";
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      el.innerHTML = `<div class="card"><h4>二阶段 · 本地个性化修正器${arch ? "（" + arch + "）" : ""}</h4>
        <div id="ch-rc-loss" class="chart" style="height:240px;"></div></div>`;
    }
    const losses = (st && st.stage2_epoch_losses) || [];
    if (document.getElementById("ch-rc-loss")) {
      setChart("ch-rc-loss", {
        grid: CHART_GRID,
        xAxis: { type: "category", data: losses.map((_, i) => i + 1),
                 ...axisX("本地训练轮次") },
        yAxis: { type: "value", ...axisY("修正器 pinball loss") },
        series: [{ type: "line", data: losses, name: "本地修正器 loss",
                   smooth: true, areaStyle: {} }],
      });
    }
    return;
  }
  // idle：一阶段完成但二阶段未开启
  const key = "idle";
  if (el.dataset.key !== key) {
    el.dataset.key = key;
    el.innerHTML = `<div class="card">
      <h4>二阶段 · 本地个性化修正器</h4>
      <p class="page-sub" style="margin:6px 0 12px;">一阶段训练已结束。二阶段在本地训练残差修正器（纯本地，不上传服务器），完成后可查看本地 loss 与预测。</p>
      <button onclick="startStage2(${task.id})">开启二阶段</button>
    </div>`;
  }
}

function renderStage3Area(body, ctx) {
  const el = document.getElementById("ct-stage3");
  if (!el) return;
  const { task, s2, s3 } = ctx;
  const done1 = (task.status === "completed" || task.status === "cancelled");
  if (!done1 || s2 !== "done") { el.innerHTML = ""; el.dataset.key = ""; return; }
  if (s3 === "running") {
    const key = "running";
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      el.innerHTML = `<div class="card"><h4>预测展示</h4>
        <p class="page-sub" style="margin-top:6px;">正在用本地测试集生成预测…</p></div>`;
    }
    return;
  }
  if (s3 === "ready") {
    const key = "ready";
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      el.innerHTML = `<div class="card"><h4>预测展示</h4><div id="ct-predict">加载预测数据…</div></div>`;
    }
    if (el.dataset.loaded !== "1") {
      el.dataset.loaded = "1";
      loadStage3Data(ctx);
    }
    return;
  }
  // idle：二阶段完成但预测未生成 → 提供按钮
  const key = "idle";
  if (el.dataset.key !== key) {
    el.dataset.key = key;
    el.dataset.loaded = "0";
    el.innerHTML = `<div class="card">
      <h4>预测展示</h4>
      <p class="page-sub" style="margin:6px 0 12px;">本地用测试集跑预测并生成图表：真实 / 全局TCN / 全局+RC P50 / 预测区间，以及 WAPE、PINAW、区间覆盖率。</p>
      <button onclick="startStage3(${task.id})">生成本地预测</button>
    </div>`;
  }
}

// ===== 二阶段：开启（触发 agent 本地训练 RC，返回后轮询阶段状态）=====
async function startStage2(taskId) {
  try {
    const r = await fetch("/local/stage2", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "二阶段启动失败");
    showToast("二阶段已在本地开始训练");
    // 触发后立刻重建（显示训练中），随轮询更新 loss
    loadClientTaskDetail(taskId);
  } catch (e) { showToast(e.message, true); }
}

// ===== 阶段3：触发本地预测生成 =====
async function startStage3(taskId) {
  try {
    const r = await fetch("/local/stage3", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "预测启动失败");
    showToast("本地预测生成中…");
    loadClientTaskDetail(taskId);
  } catch (e) { showToast(e.message, true); }
}

// ===== 阶段3：加载本地预测结果 =====
async function loadStage3Data(ctx) {
  const el = document.getElementById("ct-predict");
  if (!el) return;
  try {
    const taskId = ctx.task.id;
    const r = await fetch("/local/stage3-data?task_id=" + taskId);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "预测数据获取失败");
    renderStage3Charts(d, taskId);
  } catch (e) {
    el.innerHTML = `<span style="color:var(--danger);">预测数据加载失败：${e.message}</span>`;
  }
}

function renderStage3Charts(d, taskId) {
  const el = document.getElementById("ct-predict");
  if (!el) return;
  // d: { arch, wape_global, wape_rc, wape_drop_pct, pinaw, coverage,
  //      series:{ real:[], global:[], rc:[], upper:[], lower:[] }, lastWindow:{...} }
  el.innerHTML = `
    <div class="card" style="margin-bottom:12px;">
      <h4>性能对比</h4>
      <p class="page-sub" style="margin:6px 0;">修正器架构：${d.arch || "未知"}</p>
      <p class="page-sub">WAPE：全局TCN <b>${fmtNum(d.wape_global)}%</b> → 全局+RC <b>${fmtNum(d.wape_rc)}%</b>（<b style="color:var(--ok);">↓ ${fmtNum(d.wape_drop_pct)}%</b>）· PINAW ${fmtNum(d.pinaw)} · 区间覆盖率 ${fmtNum(d.coverage)}%</p>
    </div>
    <div class="grid2">
      <div class="card"><h4>WAPE 对比</h4><div id="ch-wape" style="height:240px;"></div></div>
      <div class="card"><h4>区间覆盖率</h4><div id="ch-coverage" style="height:240px;"></div></div>
    </div>
    <div class="card">
      <h4>预测动态演示（真实 / 全局TCN / 全局+RC P50 ± 区间）</h4>
      <div style="margin:8px 0;">
        <button onclick="window._fc && window._fc.play()">播放</button>
        <button class="secondary" onclick="window._fc && window._fc.pause()">暂停</button>
        <button class="secondary" onclick="window._fc && window._fc.reset()">重置</button>
      </div>
      <div id="ch-forecast" style="height:300px;"></div>
    </div>`;
  // 静态图1：柱状 = 全局TCN WAPE / 全局+RC WAPE / PINAW（need.md b）
  setChart("ch-wape", {
    grid: CHART_GRID,
    legend: { top: 0 },
    xAxis: { type: "category", data: ["全局TCN WAPE", "全局+RC WAPE", "PINAW"],
             ...axisX("评估指标") },
    yAxis: [
      { type: "value", ...axisY("WAPE（%）") },
      { type: "value", ...axisY("区间平均宽度 PINAW（%）"), splitLine: { show: false } },
    ],
    series: [
      { type: "bar", name: "WAPE", yAxisIndex: 0,
        data: [d.wape_global, d.wape_rc, null],
        itemStyle: { color: p => p.dataIndex === 1 ? "#0d9488" : "#64748b" } },
      { type: "bar", name: "PINAW", yAxisIndex: 1,
        data: [null, null, fmtNum(d.pinaw)],
        itemStyle: { color: "#d97706" } },
    ],
  });
  // 静态图2：饼图覆盖率（覆盖率 vs 未覆盖）
  setChart("ch-coverage", {
    series: [{ type: "pie", radius: ["42%", "68%"],
               data: [
                 { value: d.coverage, name: "区间覆盖率", itemStyle: { color: "#0d9488" } },
                 { value: Math.max(0, 100 - d.coverage), name: "未覆盖", itemStyle: { color: "#e2e8f0" } },
               ] }],
  });
  setupForecastPlayer("ch-forecast", d.series || null);
}

function fmtNum(v) {
  if (v === null || v === undefined || isNaN(v)) return "-";
  return (Math.round(v * 100) / 100);
}

// ===== 动态预测播放器 =====
function setupForecastPlayer(chartId, series) {
  const el = document.getElementById(chartId);
  if (!el) return;
  let chart = charts[chartId];
  if (!chart || chart.getDom() !== el) {
    if (chart) chart.dispose();
    chart = echarts.init(el);
    charts[chartId] = chart;
  }
  if (!series || !series.real || !series.real.length) {
    chart.setOption({ title: { text: "暂无预测数据", left: "center", top: "middle" } });
    return;
  }
  const n = series.real.length;
  let idx = 0, timer = null;
  function draw() {
    // 未开始（初始/重置后）：只画 1 个点 = 空图 + 提示；点「播放」后从第 1 步逐段推进
    const upto = Math.min(idx + 1, n);
    const x = Array.from({ length: upto }, (_, i) => i);
    const sl = (arr) => arr.slice(0, upto);
    chart.setOption({
      grid: CHART_GRID,
      title: {
        show: upto <= 1,
        text: "点击「播放」开始演示",
        left: "center", top: "middle",
        textStyle: { color: "#94a3b8", fontSize: 14, fontWeight: "normal" },
      },
      xAxis: { type: "category", data: x, ...axisX("预测步（30 分钟/步）") },
      yAxis: { type: "value", scale: true, ...axisY("用电负荷") },
      legend: { top: 0 },
      series: [
        { type: "line", data: sl(series.real), name: "真实", smooth: true },
        { type: "line", data: sl(series.global), name: "全局TCN预测", smooth: true },
        { type: "line", data: sl(series.rc), name: "全局+RC P50", smooth: true },
        { type: "line", data: sl(series.upper), name: "区间上界", smooth: true, lineStyle: { opacity: .3 } },
        { type: "line", data: sl(series.lower), name: "区间下界", smooth: true, lineStyle: { opacity: .3 } },
      ],
    }, true);
  }
  const play = () => {
    if (timer) return;
    timer = setInterval(() => { idx++; if (idx >= n) { idx = n - 1; pause(); } draw(); }, 200);
  };
  const pause = () => { if (timer) { clearInterval(timer); timer = null; } };
  const reset = () => { pause(); idx = 0; draw(); };
  window._fc = { play, pause, reset };
  draw();
}
