async function api(path, options = {}) {
  const token = localStorage.getItem("clusterAppToken") || "";
  const headers = {"Content-Type": "application/json", ...(options.headers || {})};
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem("clusterAppToken");
      localStorage.removeItem("clusterAppUser");
    }
    throw new Error(await response.text());
  }
  return response.json();
}

function friendlyError(error) {
  try {
    const payload = JSON.parse(error.message);
    if (typeof payload.detail === "string") return payload.detail;
  } catch (_) {}
  return error.message || "Unexpected error";
}

function text(id, value, kind = null) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  if (kind !== null) el.className = `notice ${kind}`.trim();
}

function html(id, markup) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = markup;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function formatDuration(seconds) {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function statusBadge(status) {
  const colors = {
    running: "#2f6fae",
    succeeded: "#27633a",
    failed: "#a33434",
    canceled: "#667487",
    queued: "#b8871a",
  };
  const color = colors[status] || "#667487";
  return `<span style="display:inline-block;background:${color};color:#fff;border-radius:4px;padding:0 8px;font-size:12px;line-height:20px">${status}</span>`;
}

async function refreshDashboard() {
  const [status, jobs, nodeDetails] = await Promise.all([
    api("/api/metrics/status"),
    api("/api/jobs"),
    api("/api/metrics/node-details"),
  ]);

  text("dashboard-status", `${status.cluster} · ${status.nodes} nodes · ${status.queue_depth} queued`);
  text("metric-nodes", String(status.nodes));
  text("metric-queued", String(status.queue_depth));
  text("metric-active", status.active_job || "none");
  text("metric-dask", status.scheduler_running ? "online" : "offline");

  const failedCount = jobs.filter(j => j.status === "failed").length;
  text("metric-failed", String(failedCount));

  renderNodeDetails(nodeDetails.nodes);
  renderActiveJob(jobs);
  renderJobsList(jobs);
  renderFailedJobs(jobs);
  await loadUsers();
}

function renderNodeDetails(nodes) {
  if (!nodes.length) {
    html("node-details", '<div class="muted">No nodes registered</div>');
    return;
  }
  const rows = nodes.map(n => {
    const gpu = n.gpu_backends && n.gpu_backends.length ? n.gpu_backends.join(", ") : "none";
    const ram = n.ram_gb != null ? `${n.ram_gb} GB` : "—";
    const lastSeen = n.last_seen_at ? new Date(n.last_seen_at).toLocaleString() : "—";
    const statusColor = n.status === "online" ? "#27633a" : n.status === "offline" ? "#a33434" : "#667487";
    return `<tr>
      <td><strong>${esc(n.name)}</strong></td>
      <td>${esc(n.host)}</td>
      <td>${n.cpu}</td>
      <td>${ram}</td>
      <td>${gpu}</td>
      <td style="color:${statusColor}">${n.status}</td>
      <td style="font-size:12px">${lastSeen}</td>
    </tr>`;
  }).join("");
  html("node-details", `<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr style="text-align:left;border-bottom:2px solid #d7dce3">
      <th style="padding:6px 8px">Name</th>
      <th style="padding:6px 8px">Host</th>
      <th style="padding:6px 8px">CPU</th>
      <th style="padding:6px 8px">RAM</th>
      <th style="padding:6px 8px">GPU</th>
      <th style="padding:6px 8px">Status</th>
      <th style="padding:6px 8px">Last Seen</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`);
}

function renderActiveJob(jobs) {
  const active = jobs.find(j => j.status === "running");
  if (!active) {
    html("active-job-detail", '<div class="muted">No active job</div>');
    return;
  }
  const duration = active.started_at
    ? formatDuration((Date.now() - new Date(active.started_at).getTime()) / 1000)
    : "—";
  html("active-job-detail", `<div class="item">
    <strong>${esc(active.name)}</strong> ${statusBadge(active.status)}
    <div class="muted" style="margin-top:4px">entrypoint: ${esc(active.entrypoint)}</div>
    <div class="muted">running for: ${duration}</div>
    <div class="muted" style="font-size:12px">${active.id}</div>
    <div style="margin-top:6px">
      <button type="button" class="secondary" style="font-size:12px;min-height:30px"
        onclick="showJobDetail('${active.id}')">View Logs</button>
      <button type="button" class="danger" style="font-size:12px;min-height:30px;margin-left:6px"
        onclick="stopJob('${active.id}')">Stop</button>
    </div>
  </div>`);
}

function renderJobsList(jobs) {
  if (!jobs.length) {
    html("jobs", '<div class="muted">No jobs yet</div>');
    return;
  }
  html("jobs", jobs.map(j => `<div class="item" style="cursor:pointer" onclick="showJobDetail('${j.id}')">
    <strong>${esc(j.name)}</strong> ${statusBadge(j.status)}
    <div class="muted" style="font-size:12px">${j.id}</div>
  </div>`).join(""));
}

function renderFailedJobs(jobs) {
  const failed = jobs.filter(j => j.status === "failed");
  if (!failed.length) {
    html("failed-jobs", '<div class="muted">No failures</div>');
    return;
  }
  html("failed-jobs", failed.map(j => {
    const meta = j.metadata || {};
    const rc = meta.return_code != null ? `exit code ${meta.return_code}` : "";
    return `<div class="item" style="cursor:pointer" onclick="showJobDetail('${j.id}')">
      <strong>${esc(j.name)}</strong>
      <span style="color:#a33434;font-size:12px">FAILED ${rc}</span>
      <div class="muted" style="font-size:12px">${j.id}</div>
    </div>`;
  }).join(""));
}

async function loadUsers() {
  const container = document.getElementById("registered-users");
  if (!container) return;
  if (!localStorage.getItem("clusterAppToken")) {
    container.innerHTML = '<div class="muted">Sign in to manage users</div>';
    text("users-status", "");
    return;
  }
  try {
    const result = await api("/api/admin/users");
    if (!result.users.length) {
      container.innerHTML = '<div class="muted">No users registered</div>';
      text("users-status", "");
      return;
    }
    container.innerHTML = result.users.map(user => `
      <div class="item user-row">
        <div class="user-copy">
          <strong>${esc(user.name)}</strong>
          <div class="muted">${esc(user.email)} · ${esc(user.role)}</div>
        </div>
        <button
          type="button"
          class="icon-button danger"
          title="Delete user"
          aria-label="Delete ${esc(user.name)}"
          data-delete-user="${user.id}"
          data-user-name="${esc(user.name)}"
        >X</button>
      </div>
    `).join("");
    text("users-status", "");
  } catch (error) {
    container.innerHTML = '<div class="muted">Could not load users</div>';
    text("users-status", friendlyError(error), "error");
  }
}

async function deleteUser(userId, userName) {
  const label = userName || "this user";
  if (!confirm(`Delete ${label}? Finished job records owned by this user will also be removed.`)) {
    return;
  }
  text("users-status", "Deleting user...");
  try {
    const result = await api(`/api/admin/users/${encodeURIComponent(userId)}`, {method: "DELETE"});
    if (result.self_deleted) {
      localStorage.removeItem("clusterAppToken");
      localStorage.removeItem("clusterAppUser");
      document.getElementById("registered-users").innerHTML =
        '<div class="muted">Sign in to manage users</div>';
      text("users-status", "User deleted. Sign in again to manage users.", "success");
      return;
    } else {
      text("users-status", "User deleted", "success");
    }
    await loadUsers();
  } catch (error) {
    text("users-status", friendlyError(error), "error");
  }
}

async function showJobDetail(jobId) {
  const modal = document.getElementById("job-detail-modal");
  modal.setAttribute("aria-hidden", "false");
  text("modal-title", "Job Detail");
  text("modal-subtitle", "Loading...");
  html("modal-body", '<div class="muted">Loading job details...</div>');
  try {
    const detail = await api(`/api/jobs/${jobId}/detail`);
    const job = detail.job;
    text("modal-subtitle", `${esc(job.name)} · ${job.id}`);

    let stopButton = "";
    if (job.status === "running") {
      stopButton = `<button type="button" class="danger" style="font-size:13px;min-height:34px;margin-top:12px"
        onclick="stopJob('${job.id}')">Stop Job</button>`;
    }

    let failureHtml = "";
    if (detail.failure) {
      failureHtml = `<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:12px;margin-top:12px">
        <strong style="color:#a33434">Failure Details</strong>
        <div style="margin-top:6px;font-size:13px">
          <div>Return code: <code>${detail.failure.return_code ?? "—"}</code></div>
          ${detail.failure.error_message ? `<div style="margin-top:4px">Last error: <code>${esc(detail.failure.error_message)}</code></div>` : ""}
          ${detail.failure.traceback ? `<pre style="background:#1c2430;color:#e4e8ee;padding:10px;border-radius:4px;margin-top:6px;font-size:12px;overflow:auto;max-height:200px">${esc(detail.failure.traceback)}</pre>` : ""}
        </div>
      </div>`;
    }

    const logs = detail.logs || [];
    const logHtml = logs.length
      ? logs.map(log => {
          const color = log.stream === "stderr" ? "#a33434" : log.stream === "system" ? "#b8871a" : "#1c2430";
          return `<div style="color:${color};font-family:monospace;font-size:12px;white-space:pre-wrap;padding:1px 0">${esc(log.message)}</div>`;
        }).join("")
      : '<div class="muted">No log output</div>';

    const meta = job.metadata || {};

    html("modal-body", `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px">
        <div><strong>Status</strong><br>${statusBadge(job.status)}</div>
        <div><strong>Entrypoint</strong><br>${esc(job.entrypoint)}</div>
        <div><strong>Duration</strong><br>${formatDuration(detail.duration_seconds)}</div>
        <div><strong>Queue Wait</strong><br>${formatDuration(detail.queue_wait_seconds)}</div>
        <div><strong>Retries Left</strong><br>${job.retries_left}</div>
        <div><strong>Dask Scheduler</strong><br>${job.dask_scheduler_url || "—"}</div>
      </div>
      <div style="margin-top:12px;font-size:13px"><strong>Args:</strong> ${(job.args || []).join(" ") || "—"}</div>
      ${failureHtml}
      ${stopButton}
      <div style="margin-top:12px">
        <strong>Logs (${logs.length} lines)</strong>
        <div style="background:#f2f6fa;border:1px solid #d7dce3;border-radius:6px;padding:10px;margin-top:6px;max-height:300px;overflow:auto">${logHtml}</div>
      </div>
    `);
  } catch (error) {
    html("modal-body", `<div class="notice error">${esc(friendlyError(error))}</div>`);
  }
}

function closeJobDetail() {
  document.getElementById("job-detail-modal").setAttribute("aria-hidden", "true");
}

async function stopJob(jobId) {
  if (!confirm("Stop this job?")) return;
  try {
    await api(`/api/jobs/${jobId}/cancel`, {method: "POST"});
    await refreshDashboard();
  } catch (error) {
    alert(friendlyError(error));
  }
}

async function testAllNodes() {
  const button = document.getElementById("test-nodes");
  button.disabled = true;
  text("test-status", "Testing node connectivity...");
  html("test-results", "");
  try {
    const result = await api("/api/nodes/health");
    const rows = result.nodes.map(n => {
      const icon = n.reachable ? "\u2705" : "\u274c";
      const gpu = n.resources && (n.resources.GPU || Object.keys(n.resources).filter(k => k.startsWith("GPU_")).length)
        ? "yes" : "no";
      const cpu = n.resources?.CPU ?? "?";
      return `<tr>
        <td><strong>${esc(n.name)}</strong></td>
        <td>${esc(n.host)}:${n.port || "?"}</td>
        <td>${icon} ${n.reachable ? "OK" : "UNREACHABLE"}</td>
        <td>${n.error ? esc(n.error) : "—"}</td>
        <td>${cpu}</td>
        <td>${gpu}</td>
      </tr>`;
    }).join("");
    const reachable = result.nodes.filter(n => n.reachable).length;
    html("test-results", `<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="text-align:left;border-bottom:2px solid #d7dce3">
        <th style="padding:6px 8px">Name</th>
        <th style="padding:6px 8px">Host:Port</th>
        <th style="padding:6px 8px">Connectivity</th>
        <th style="padding:6px 8px">Error</th>
        <th style="padding:6px 8px">CPU</th>
        <th style="padding:6px 8px">GPU</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <div style="margin-top:6px;font-size:13px">${reachable}/${result.nodes.length} nodes reachable</div>`);
    text("test-status", reachable === result.nodes.length ? "All nodes healthy" : "Some nodes unreachable", "success");
  } catch (error) {
    text("test-status", friendlyError(error), "error");
  } finally {
    button.disabled = false;
  }
}

document.getElementById("test-nodes").addEventListener("click", testAllNodes);
document.getElementById("close-modal").addEventListener("click", closeJobDetail);
document.getElementById("job-detail-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) closeJobDetail();
});
document.getElementById("refresh-users").addEventListener("click", loadUsers);
document.getElementById("registered-users").addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete-user]");
  if (!button) return;
  deleteUser(button.dataset.deleteUser, button.dataset.userName);
});

refreshDashboard().catch(error => {
  text("dashboard-status", error.message || "Dashboard failed to load");
});
setInterval(() => refreshDashboard().catch(() => {}), 5000);
