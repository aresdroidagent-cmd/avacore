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
  const password = document.getElementById("adminPassword")?.value || "";
  localStorage.setItem("avacore_admin_password", password);
  await loadAdminConfig();
}

async function loadAdminConfig() {
  const password = localStorage.getItem("avacore_admin_password") || "";
  try {
    const data = await fetchJson("/admin/runtime", {
      headers: { "X-Admin-Password": password }
    });
    setText("adminOutput", data);
  } catch (err) {
    setText("adminOutput", String(err));
  }
}

function clearAdminPassword() {
  localStorage.removeItem("avacore_admin_password");
  setText("adminOutput", "Password cleared.");
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("sendBtn")?.addEventListener("click", sendChat);
  document.getElementById("resetBtn")?.addEventListener("click", resetChat);
  document.getElementById("pageExplainBtn")?.addEventListener("click", explainPage);
  document.getElementById("docSearchBtn")?.addEventListener("click", loadDocuments);
  document.getElementById("refreshStatusBtn")?.addEventListener("click", refreshStatus);
  document.getElementById("adminLoginBtn")?.addEventListener("click", unlockAdmin);
  document.getElementById("adminClearBtn")?.addEventListener("click", clearAdminPassword);

  if (document.getElementById("docList")) loadDocuments();
  if (document.getElementById("statusOutput")) refreshStatus();
  if (document.getElementById("adminOutput")) loadAdminConfig();
});