async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  let data = {};
  try { data = await res.json(); } catch (_) {}
  if (!res.ok) {
    throw new Error(data.detail || res.statusText || "Request failed");
  }
  return data;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function appendChat(role, text) {
  const log = document.getElementById("chatLog");
  if (!log) return;
  const div = document.createElement("div");
  div.className = "chat-entry";
  div.innerHTML = `<strong>${role}</strong>\n${text}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function getAdminPassword() {
  return localStorage.getItem("avacore_admin_password") || "";
}

function setAdminPassword(value) {
  localStorage.setItem("avacore_admin_password", value || "");
}

function clearAdminPassword() {
  localStorage.removeItem("avacore_admin_password");
}

async function loadDocuments() {
  const q = document.getElementById("docSearch")?.value || "";
  const list = document.getElementById("docList");
  if (!list) return;
  list.textContent = "Loading...";
  try {
    const data = await fetchJson(`/knowledge/documents?q=${encodeURIComponent(q)}&limit=20`);
    list.innerHTML = "";
    const items = data.items || [];
    if (!items.length) {
      list.textContent = "No documents found.";
      return;
    }
    for (const item of items) {
      const div = document.createElement("div");
      div.className = "doc-item";
      const title = item.title || "";
      const type = item.doc_type || "";
      div.innerHTML = `<strong>${title}</strong><br><span class="muted">${type}</span>`;
      div.onclick = () => {
        const target = document.getElementById("pageDocument");
        if (target) target.value = title;
      };
      list.appendChild(div);
    }
  } catch (err) {
    list.textContent = String(err);
  }
}

async function sendChat() {
  const text = document.getElementById("prompt")?.value?.trim();
  if (!text) return;
  const payload = {
    channel: document.getElementById("channel")?.value || "web",
    user_id: document.getElementById("userId")?.value || "web-user",
    chat_id: document.getElementById("chatId")?.value || "web-chat",
    text,
    timestamp: Math.floor(Date.now() / 1000)
  };
  appendChat("user", text);
  document.getElementById("prompt").value = "";
  try {
    const data = await fetchJson("/reply", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    appendChat("ava", data.reply || "");
  } catch (err) {
    appendChat("error", String(err));
  }
}

async function resetChat() {
  const chatId = document.getElementById("chatId")?.value || "web-chat";
  const channel = document.getElementById("channel")?.value || "web";
  try {
    await fetchJson("/reply", {
      method: "DELETE",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        channel,
        chat_id: chatId
      })
    });
    setText("chatLog", "");
  } catch (err) {
    appendChat("error", String(err));
  }
}

async function explainPage() {
  const documentName = document.getElementById("pageDocument")?.value?.trim();
  const page = Number(document.getElementById("pageNumber")?.value || "1");
  if (!documentName) return;
  try {
    const data = await fetchJson("/knowledge/explain_page", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({document: documentName, page})
    });
    setText("pageOutput", data.answer || "");
  } catch (err) {
    setText("pageOutput", String(err));
  }
}

async function refreshStatus() {
  try {
    const health = await fetchJson("/health");
    const model = await fetchJson("/model");
    setText("statusOutput", { health, model });
  } catch (err) {
    setText("statusOutput", String(err));
  }
}

async function unlockAdmin() {
  const password =
    document.getElementById("adminPassword")?.value ||
    document.getElementById("reviewAdminPassword")?.value ||
    "";
  setAdminPassword(password);

  if (document.getElementById("adminOutput")) {
    await loadAdminConfig();
  }
  if (document.getElementById("candidateList")) {
    await loadReviewData();
  }
}

function clearAdminAccess() {
  clearAdminPassword();
  if (document.getElementById("adminOutput")) {
    setText("adminOutput", "Password cleared.");
  }
  if (document.getElementById("candidateList")) {
    setText("candidateList", "Password cleared.");
    setText("verifiedList", "");
    setText("rejectedList", "");
  }
}

async function loadAdminConfig() {
  const password = getAdminPassword();
  try {
    const data = await fetchJson("/admin/runtime", {
      headers: { "X-Admin-Password": password }
    });
    setText("adminOutput", data);
  } catch (err) {
    setText("adminOutput", String(err));
  }
}

function renderMemoryCard(item, mode = "candidate") {
  const div = document.createElement("div");
  div.className = "review-item";

  const title = item.title || "";
  const content = item.content || "";
  const memoryType = item.memory_type || "";
  const status = item.status || "";
  const sourceType = item.source_type || "";
  const sourceRef = item.source_ref || "";
  const confidence = item.confidence ?? 0;
  const tags = item.tags || "";
  const id = item.id;

  let buttons = "";
  if (mode === "candidate") {
    buttons = `
      <button data-action="verify" data-id="${id}">Verify</button>
      <button data-action="reject" data-id="${id}" class="secondary">Reject</button>
      <button data-action="delete" data-id="${id}" class="danger">Delete</button>
    `;
  } else if (mode === "verified") {
    buttons = `
      <button data-action="reject" data-id="${id}" class="secondary">Reject</button>
      <button data-action="delete" data-id="${id}" class="danger">Delete</button>
    `;
  } else if (mode === "rejected") {
    buttons = `
      <button data-action="verify" data-id="${id}">Verify</button>
      <button data-action="delete" data-id="${id}" class="danger">Delete</button>
    `;
  }

  div.innerHTML = `
    <div class="review-header">
      <strong>${title}</strong>
      <span class="badge">${status}</span>
    </div>
    <div class="review-meta">
      <span>type: ${memoryType}</span>
      <span>source: ${sourceType}</span>
      <span>ref: ${sourceRef}</span>
      <span>confidence: ${Number(confidence).toFixed(2)}</span>
    </div>
    <div class="review-content">${content}</div>
    <div class="review-tags muted">${tags}</div>
    <div class="review-actions">${buttons}</div>
  `;
  return div;
}

async function verifyMemoryItem(id) {
  const password = getAdminPassword();
  await fetchJson(`/memories/items/${id}/verify`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Password": password
    },
    body: JSON.stringify({ actor: "roger" })
  });
}

async function rejectMemoryItem(id) {
  const password = getAdminPassword();
  await fetchJson(`/memories/items/${id}/reject`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Password": password
    },
    body: JSON.stringify({ actor: "roger" })
  });
}

async function deleteMemoryItem(id) {
  const password = getAdminPassword();
  await fetchJson(`/memories/items/${id}`, {
    method: "DELETE",
    headers: {
      "X-Admin-Password": password
    }
  });
}

async function loadReviewData() {
  const password = getAdminPassword();
  const candidateList = document.getElementById("candidateList");
  const verifiedList = document.getElementById("verifiedList");
  const rejectedList = document.getElementById("rejectedList");
  if (!candidateList || !verifiedList || !rejectedList) return;

  candidateList.textContent = "Loading...";
  verifiedList.textContent = "Loading...";
  rejectedList.textContent = "Loading...";

  try {
    const [candidates, verified, rejected] = await Promise.all([
      fetchJson("/memories/candidates?limit=50", {
        headers: { "X-Admin-Password": password }
      }),
      fetchJson("/memories/verified?limit=50", {
        headers: { "X-Admin-Password": password }
      }),
      fetchJson("/memories/rejected?limit=50", {
        headers: { "X-Admin-Password": password }
      }),
    ]);

    candidateList.innerHTML = "";
    verifiedList.innerHTML = "";
    rejectedList.innerHTML = "";

    const cItems = candidates.items || [];
    const vItems = verified.items || [];
    const rItems = rejected.items || [];

    if (!cItems.length) {
      candidateList.textContent = "No candidate memories.";
    } else {
      for (const item of cItems) {
        candidateList.appendChild(renderMemoryCard(item, "candidate"));
      }
    }

    if (!vItems.length) {
      verifiedList.textContent = "No verified memories.";
    } else {
      for (const item of vItems) {
        verifiedList.appendChild(renderMemoryCard(item, "verified"));
      }
    }

    if (!rItems.length) {
      rejectedList.textContent = "No rejected memories.";
    } else {
      for (const item of rItems) {
        rejectedList.appendChild(renderMemoryCard(item, "rejected"));
      }
    }

    document.querySelectorAll("[data-action]").forEach(btn => {
      btn.addEventListener("click", async () => {
        const action = btn.getAttribute("data-action");
        const id = btn.getAttribute("data-id");
        if (!id) return;

        try {
          if (action === "verify") {
            await verifyMemoryItem(id);
          } else if (action === "reject") {
            await rejectMemoryItem(id);
          } else if (action === "delete") {
            await deleteMemoryItem(id);
          }
          await loadReviewData();
        } catch (err) {
          alert(String(err));
        }
      });
    });
  } catch (err) {
    candidateList.textContent = String(err);
    verifiedList.textContent = "";
    rejectedList.textContent = "";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("sendBtn")?.addEventListener("click", sendChat);
  document.getElementById("resetBtn")?.addEventListener("click", resetChat);
  document.getElementById("pageExplainBtn")?.addEventListener("click", explainPage);
  document.getElementById("docSearchBtn")?.addEventListener("click", loadDocuments);
  document.getElementById("refreshStatusBtn")?.addEventListener("click", refreshStatus);

  document.getElementById("adminLoginBtn")?.addEventListener("click", unlockAdmin);
  document.getElementById("adminClearBtn")?.addEventListener("click", clearAdminAccess);

  document.getElementById("reviewLoginBtn")?.addEventListener("click", unlockAdmin);
  document.getElementById("reviewClearBtn")?.addEventListener("click", clearAdminAccess);
  document.getElementById("reviewRefreshBtn")?.addEventListener("click", loadReviewData);

  if (document.getElementById("docList")) loadDocuments();
  if (document.getElementById("statusOutput")) refreshStatus();
  if (document.getElementById("adminOutput")) loadAdminConfig();
  if (document.getElementById("candidateList")) loadReviewData();
});