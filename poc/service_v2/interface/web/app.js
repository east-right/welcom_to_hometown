// 프론트 흐름: 피드(더미) → 채팅 → 검색 완료 시 피드 갱신.
// 서버 API는 /api/chat, /api/reset 두 개만 쓴다.

const SESSION_ID = crypto.randomUUID();

const DOMAIN = {
  restaurant: { label: "맛집", cls: "domain-restaurant" },
  cafe:       { label: "카페", cls: "domain-cafe" },
  festival:   { label: "축제", cls: "domain-festival" },
  spot:       { label: "명소", cls: "domain-spot" },
};

const el = {
  feed:      document.getElementById("feed"),
  feedSub:   document.getElementById("feed-sub"),
  chatView:  document.getElementById("chat-view"),
  messages:  document.getElementById("messages"),
  composer:  document.getElementById("composer"),
  input:     document.getElementById("chat-input"),
  send:      document.querySelector(".send-btn"),
};

// ── 피드 렌더 ────────────────────────────────────────
function renderPlaceholderFeed() {
  const domains = Object.values(DOMAIN);
  el.feed.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const d = domains[i % domains.length];
    el.feed.insertAdjacentHTML("beforeend", `
      <article class="card is-placeholder">
        <div class="card-media ${d.cls}"><span class="badge">준비 중</span></div>
        <div class="card-body">
          <span class="ghost"></span>
          <span class="ghost short"></span>
        </div>
      </article>`);
  }
}

function renderFeed(cards) {
  el.feed.innerHTML = "";
  cards.forEach((c) => el.feed.insertAdjacentHTML("beforeend", cardHtml(c)));
  el.feed.scrollTop = 0;
}

function cardHtml(c) {
  const d = DOMAIN[c.domain] || { label: c.domain, cls: "" };
  const meta = cardMeta(c);
  const src = c.source
    ? `<a class="card-src" href="${esc(c.source)}" target="_blank" rel="noopener">출처 보기 ↗</a>`
    : "";
  return `
    <article class="card">
      <div class="card-media ${d.cls}"><span class="badge">${esc(d.label)}</span></div>
      <div class="card-body">
        <h3 class="card-name">${esc(c.name)}</h3>
        ${meta ? `<div class="card-meta">${meta}</div>` : ""}
        <p class="card-reason">${esc(c.reason)}</p>
        ${src}
      </div>
    </article>`;
}

function cardMeta(c) {
  const parts = [];
  if (c.period) parts.push(`📅 ${esc(c.period)}`);
  if (c.hours)  parts.push(`🕒 ${esc(c.hours)}`);
  if (c.closed) parts.push(`🚫 ${esc(c.closed)}`);
  return parts.join(" · ");
}

// ── 채팅 렌더 ────────────────────────────────────────
function addBubble(text, who) {
  const b = document.createElement("div");
  b.className = `bubble ${who}`;
  b.textContent = text;
  el.messages.appendChild(b);
  el.messages.scrollTop = el.messages.scrollHeight;
  return b;
}

function showTyping() {
  const b = document.createElement("div");
  b.className = "bubble bot typing";
  b.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>
                 <span style="margin-left:4px;color:#9a8f83;font-size:13px">찾는 중…</span>`;
  el.messages.appendChild(b);
  el.messages.scrollTop = el.messages.scrollHeight;
  return b;
}

// ── 뷰 전환 ─────────────────────────────────────────
function openChat() {
  el.chatView.classList.add("is-active");
  if (!el.messages.dataset.greeted) {
    addBubble("어디로, 언제, 누구랑 갈지 편하게 말해주세요. 조건이 모이면 딱 맞는 곳을 찾아드려요!", "bot");
    el.messages.dataset.greeted = "1";
  }
  el.input.focus();
}

function closeChat() {
  el.chatView.classList.remove("is-active");
}

async function resetChat() {
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION_ID }),
  });
  el.messages.innerHTML = "";
  delete el.messages.dataset.greeted;
  openChat();
}

// ── 한 턴: 메시지 전송 ───────────────────────────────
async function sendMessage(text) {
  addBubble(text, "me");
  setBusy(true);
  const typing = showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID, message: text }),
    });
    if (!res.ok) throw new Error(`서버 오류 (${res.status})`);
    const data = await res.json();
    typing.remove();

    if (data.feed) {
      addBubble(data.reply, "bot");
      renderFeed(data.feed);
      el.feedSub.textContent = "추천 결과예요";
      setTimeout(closeChat, 600);   // 봇 답변 잠깐 보여주고 피드로 전환
    } else {
      addBubble(data.reply, "bot");
    }
  } catch (err) {
    typing.remove();
    addBubble(`앗, 문제가 생겼어요: ${err.message}\n잠시 후 다시 시도해 주세요.`, "bot");
  } finally {
    setBusy(false);
  }
}

function feedSubText(slots) {
  const bits = [slots.region, slots.companion, slots.purpose].filter(Boolean);
  return bits.length ? `${bits.join(" · ")} 추천이에요` : "추천 결과예요";
}

function setBusy(busy) {
  el.input.disabled = busy;
  el.send.disabled = busy;
  if (!busy) el.input.focus();
}

// ── 유틸 ────────────────────────────────────────────
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[m]));
}

// ── 바인딩 ──────────────────────────────────────────
document.getElementById("open-chat").addEventListener("click", openChat);
document.getElementById("close-chat").addEventListener("click", closeChat);
document.getElementById("reset-chat").addEventListener("click", resetChat);
el.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = el.input.value.trim();
  if (!text || el.input.disabled) return;
  el.input.value = "";
  sendMessage(text);
});

renderPlaceholderFeed();
