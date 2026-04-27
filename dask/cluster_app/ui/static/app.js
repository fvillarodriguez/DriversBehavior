let token = localStorage.getItem("clusterAppToken") || "";
let currentUser = JSON.parse(localStorage.getItem("clusterAppUser") || "null");
let fileDialogMode = "folder";
let fileDialogPath = "";
let selectedJobFolder = "";

function setNotice(id, message, kind = "") {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = message;
  element.className = `notice ${kind}`.trim();
}

function friendlyError(error) {
  try {
    const payload = JSON.parse(error.message);
    if (typeof payload.detail === "string") return payload.detail;
  } catch (_) {
    // Fall through to raw message.
  }
  return error.message || "Unexpected error";
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function rememberAuth(result) {
  token = result.token;
  currentUser = result.user;
  localStorage.setItem("clusterAppToken", token);
  localStorage.setItem("clusterAppUser", JSON.stringify(currentUser));
  setNotice("auth-status", `User saved: ${currentUser.email}`, "success");
}

function clearAuth() {
  token = "";
  currentUser = null;
  localStorage.removeItem("clusterAppToken");
  localStorage.removeItem("clusterAppUser");
  setNotice("auth-status", "Session expired. Please login again.", "error");
}

async function api(path, options = {}) {
  const headers = {"Content-Type": "application/json", ...(options.headers || {})};
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    if (response.status === 401) clearAuth();
    throw new Error(await response.text());
  }
  return response.json();
}

document.getElementById("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  setNotice("auth-status", "Saving user...");
  try {
    const result = await api("/api/auth/register", {method: "POST", body: JSON.stringify(data)});
    rememberAuth(result);
    await loadOnlineUsers();
  } catch (error) {
    setNotice("auth-status", friendlyError(error), "error");
  }
});

document.getElementById("refresh-online-users").addEventListener("click", () => {
  loadOnlineUsers().catch((error) => setNotice("auth-status", friendlyError(error), "error"));
});

document.getElementById("job-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  delete data.entrypoint_select;
  data.args = data.args ? data.args.split(" ").filter(Boolean) : [];
  if (!data.entrypoint) data.entrypoint = null;
  setNotice("job-status", "Submitting job...");
  try {
    await api("/api/jobs", {method: "POST", body: JSON.stringify(data)});
    setNotice("job-status", "Job queued", "success");
    await refresh();
  } catch (error) {
    setNotice("job-status", friendlyError(error), "error");
  }
});

document.getElementById("clear-queue-records").addEventListener("click", async () => {
  if (!token) {
    setNotice("queue-status", "Save a user before clearing records.", "error");
    return;
  }
  if (!confirm("Clear finished, failed, canceled, and interrupted job records?")) return;
  setNotice("queue-status", "Clearing records...");
  try {
    const result = await api("/api/jobs/records", {method: "DELETE"});
    setNotice("queue-status", `Removed ${result.removed} records`, "success");
    await refresh();
  } catch (error) {
    setNotice("queue-status", friendlyError(error), "error");
  }
});

document.getElementById("manual-node-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  data.port = Number(data.port || 18080);
  setNotice("node-status", "Checking node...");
  try {
    const node = await api("/api/nodes/manual", {method: "POST", body: JSON.stringify(data)});
    setNotice("node-status", `Added ${node.name}`, "success");
    await refresh();
  } catch (error) {
    setNotice("node-status", friendlyError(error), "error");
  }
});

document.getElementById("cleanup-old-nodes").addEventListener("click", async () => {
  if (!token) {
    setNotice("node-status", "Save a user before removing old nodes.", "error");
    return;
  }
  if (!confirm("Remove offline and revoked node records? Online nodes will stay listed.")) return;
  setNotice("node-status", "Removing old nodes...");
  try {
    const result = await api("/api/nodes/cleanup-old", {method: "POST"});
    setNotice("node-status", `Removed ${result.removed} old nodes`, "success");
    await refresh();
  } catch (error) {
    setNotice("node-status", friendlyError(error), "error");
  }
});

document.getElementById("start-scheduler").addEventListener("click", async () => {
  const button = document.getElementById("start-scheduler");
  button.disabled = true;
  if (!token) {
    setNotice("scheduler-status", "Please login or register before starting the scheduler.", "error");
    setNotice("auth-status", "Login required for scheduler control.", "error");
    button.disabled = false;
    return;
  }
  setNotice("scheduler-status", "Starting scheduler...");
  try {
    const status = await api("/api/admin/scheduler/start", {method: "POST"});
    updateSchedulerStatus(status);
    const action = status.local === false ? "Using scheduler" : "Scheduler starting";
    setNotice("scheduler-status", `${action} at ${status.address}`, "success");
    await refresh();
  } catch (error) {
    setNotice("scheduler-status", friendlyError(error), "error");
  } finally {
    button.disabled = false;
  }
});

document.getElementById("start-worker").addEventListener("click", async () => {
  const button = document.getElementById("start-worker");
  button.disabled = true;
  if (!token) {
    setNotice("worker-status", "Please login or register before starting a worker.", "error");
    setNotice("auth-status", "Login required for worker control.", "error");
    button.disabled = false;
    return;
  }
  setNotice("worker-status", "Starting worker...");
  try {
    const status = await api("/api/admin/worker/start", {method: "POST"});
    updateWorkerStatus(status);
    await refresh();
  } catch (error) {
    setNotice("worker-status", friendlyError(error), "error");
  } finally {
    button.disabled = false;
  }
});

document.getElementById("browse-folder").addEventListener("click", async () => {
  await openFileDialog("folder", document.getElementById("source-dir").value || "");
});

document.getElementById("browse-file").addEventListener("click", async () => {
  const sourceDir = document.getElementById("source-dir").value;
  if (!sourceDir) {
    setNotice("job-status", "Choose a job folder first", "error");
    return;
  }
  await openFileDialog("file", sourceDir);
});

document.getElementById("entrypoint-select").addEventListener("change", (event) => {
  document.getElementById("entrypoint").value = event.target.value;
});

document.getElementById("source-dir").addEventListener("change", async (event) => {
  selectedJobFolder = event.target.value;
  await loadEntrypoints(selectedJobFolder);
});

document.getElementById("close-file-dialog").addEventListener("click", closeFileDialog);
document.getElementById("file-home").addEventListener("click", async () => openFileDialog(fileDialogMode, ""));
document.getElementById("file-up").addEventListener("click", async () => {
  const parent = document.getElementById("file-up").dataset.parent;
  if (parent) await renderFileDialog(parent);
});
document.getElementById("choose-current-folder").addEventListener("click", async () => {
  await chooseFolder(fileDialogPath);
});

async function openFileDialog(mode, path) {
  fileDialogMode = mode;
  document.getElementById("file-dialog-title").textContent =
    mode === "folder" ? "Select Job Folder" : "Select Python Entrypoint";
  document.getElementById("choose-current-folder").style.display =
    mode === "folder" ? "inline-flex" : "none";
  document.getElementById("file-dialog").setAttribute("aria-hidden", "false");
  await renderFileDialog(path);
}

function closeFileDialog() {
  document.getElementById("file-dialog").setAttribute("aria-hidden", "true");
}

async function renderFileDialog(path) {
  setNotice("job-status", "");
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  const payload = await api(`/api/filesystem/list${query}`);
  fileDialogPath = payload.path;
  document.getElementById("file-dialog-path").textContent = payload.path;
  document.getElementById("file-up").dataset.parent = payload.parent || "";
  document.getElementById("file-up").disabled = !payload.parent;
  const list = document.getElementById("file-list");
  list.innerHTML = "";
  if (payload.entries.length === 0) {
    list.innerHTML = `<div class="item muted">No folders or Python files here</div>`;
    return;
  }
  for (const entry of payload.entries) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "file-row";
    row.innerHTML = `
      <strong>${entry.kind === "directory" ? "dir" : "py"}</strong>
      <span>${entry.name}</span>
      <span class="badge">${entry.kind === "directory" ? "Open" : "Choose"}</span>
    `;
    row.addEventListener("click", async () => {
      if (entry.kind === "directory") {
        await renderFileDialog(entry.path);
      } else if (fileDialogMode === "file") {
        chooseEntrypoint(entry.path);
      }
    });
    list.appendChild(row);
  }
}

async function chooseFolder(path) {
  selectedJobFolder = path;
  document.getElementById("source-dir").value = path;
  closeFileDialog();
  await loadEntrypoints(path);
  setNotice("job-status", "Folder selected", "success");
}

async function loadEntrypoints(path) {
  const select = document.getElementById("entrypoint-select");
  const input = document.getElementById("entrypoint");
  select.innerHTML = `<option value="">Scanning Python files...</option>`;
  if (!path) {
    select.innerHTML = `<option value="">Select folder first</option>`;
    input.value = "";
    return;
  }
  try {
    const payload = await api(`/api/filesystem/python-files?path=${encodeURIComponent(path)}`);
    select.innerHTML = "";
    if (payload.files.length === 0) {
      select.innerHTML = `<option value="">No .py files found</option>`;
      input.value = "";
      return;
    }
    for (const file of payload.files) {
      const option = document.createElement("option");
      option.value = file.relative_path;
      option.textContent = file.relative_path;
      select.appendChild(option);
    }
    const preferred =
      payload.files.find((file) => file.relative_path === "main.py") ||
      payload.files.find((file) => file.relative_path.endsWith("/main.py")) ||
      payload.files[0];
    select.value = preferred.relative_path;
    input.value = preferred.relative_path;
  } catch (error) {
    select.innerHTML = `<option value="">Could not scan folder</option>`;
    setNotice("job-status", friendlyError(error), "error");
  }
}

function chooseEntrypoint(path) {
  const sourceDir = document.getElementById("source-dir").value;
  const normalizedSource = sourceDir.replace(/[\\/]+$/, "");
  const insideSource =
    path === normalizedSource ||
    path.startsWith(`${normalizedSource}/`) ||
    path.startsWith(`${normalizedSource}\\`);
  if (!sourceDir || !insideSource) {
    setNotice("job-status", "Choose a file inside the selected job folder", "error");
    return;
  }
  const relative = path.slice(normalizedSource.length).replace(/^[/\\]/, "");
  document.getElementById("entrypoint").value = relative;
  closeFileDialog();
  loadEntrypoints(sourceDir).then(() => {
    const select = document.getElementById("entrypoint-select");
    const input = document.getElementById("entrypoint");
    select.value = relative;
    input.value = relative;
    setNotice("job-status", "Entrypoint selected", "success");
  });
}

async function refresh() {
  const status = await api("/api/metrics/status");
  document.getElementById("cluster-status").textContent =
    `${status.nodes} nodes, scheduler: ${status.scheduler_running ? "online" : "offline"}, ${status.queue_depth} queued, active: ${status.active_job || "none"}`;
  updateSchedulerStatus({
    running: status.scheduler_running,
    address: status.scheduler_address,
    dashboard: status.dask_dashboard,
    dashboard_reachable: status.dask_dashboard_reachable,
  });
  try {
    updateWorkerStatus(await api("/api/admin/worker/status"));
  } catch (_) {
    setNotice("worker-status", "Worker status unavailable");
  }
  const jobs = await api("/api/jobs");
  document.getElementById("jobs").innerHTML = jobs.map(job =>
    `<div class="item"><strong>${job.name}</strong><div class="muted">${job.status} · ${job.id}</div></div>`
  ).join("") || `<div class="muted">No jobs yet</div>`;
  if (token) {
    await loadOnlineUsers();
  }
}

function updateSchedulerStatus(status) {
  const link = document.getElementById("native-dashboard-main");
  if (link) {
    link.href = status.dashboard_reachable ? status.dashboard : "#";
    link.setAttribute("aria-disabled", status.dashboard_reachable ? "false" : "true");
  }
  const message = status.running
    ? `Scheduler online: ${status.address}`
    : "Scheduler offline";
  const kind = status.running ? "success" : "";
  setNotice("scheduler-status", message, kind);
}

function updateWorkerStatus(status) {
  const message = status.running
    ? `Worker online: ${status.scheduler_address || "scheduler connected"}`
    : "Worker offline";
  setNotice("worker-status", message, status.running ? "success" : "");
}

async function loadOnlineUsers() {
  const container = document.getElementById("online-users");
  if (!token) {
    container.innerHTML = `<div class="muted">Save a user to see online users</div>`;
    return;
  }
  const result = await api("/api/admin/users/online");
  container.innerHTML = result.users.map((user) => `
    <div class="item presence-user">
      <span class="presence-dot" aria-hidden="true"></span>
      <div>
        <strong>${esc(user.name)}</strong>
        <div class="muted">${esc(user.email)}</div>
      </div>
    </div>
  `).join("") || `<div class="muted">No users online</div>`;
}

const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/events`);
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type !== "status") return;
  document.getElementById("nodes").innerHTML = data.nodes.map(node =>
    `<div class="item"><strong>${node.name}</strong><div class="muted">${node.status} · ${node.host}</div></div>`
  ).join("") || `<div class="muted">No nodes connected</div>`;
};

if (currentUser) {
  setNotice("auth-status", `User saved: ${currentUser.email}`, "success");
}
refresh().catch(() => {});
setInterval(() => refresh().catch(() => {}), 5000);
