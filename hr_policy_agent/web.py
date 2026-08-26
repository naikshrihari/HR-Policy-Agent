"""Web chat UI for the HR Policy Agent.

A small FastAPI app that serves a single-page chat interface and a JSON endpoint that
runs one turn of the agent. The agent's response is already HTML (answer + referral +
citation cards), so the browser renders it directly.

Run it:
    pip install '.[web]'
    python -m hr_policy_agent.web          # then open http://localhost:8000

Configuration (LLM provider, RAG backend, etc.) is read from the environment / .env
exactly like the CLI — see README and .env.example.
"""

import uuid
from functools import lru_cache
from typing import Optional

from .agent import HRPolicyAgent
from .config import get_settings


@lru_cache(maxsize=1)
def _agent() -> HRPolicyAgent:
    # Built once (graph compile + RAG index load are expensive) and reused across requests.
    return HRPolicyAgent(get_settings())


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The web UI needs FastAPI. Install it with:  pip install '.[web]'") from exc

    app = FastAPI(title="HR Policy Agent")

    class AskRequest(BaseModel):
        message: str
        person_number: Optional[str] = None
        conversation_id: Optional[str] = None

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/config")
    def config() -> dict:
        s = get_settings()
        return {
            "provider": s.llm_provider,
            "model": s.llm_model if s.llm_provider != "fake" else None,
            "fast_mode": s.fast_mode,
            "mock_rag": s.use_mock_rag,
        }

    @app.post("/api/ask")
    def ask(req: AskRequest):
        # Sync def -> FastAPI runs it in a threadpool, so a slow LLM call doesn't block
        # the event loop / other requests.
        conversation_id = req.conversation_id or str(uuid.uuid4())
        try:
            state = _agent().run(req.message, person_number=req.person_number,
                                 conversation_id=conversation_id)
        except Exception as exc:  # noqa: BLE001 - surface a friendly error to the UI
            return JSONResponse(status_code=500, content={"error": str(exc)})
        return JSONResponse({
            "response": state.get("final_response", ""),
            "language": state.get("language", "EN"),
            "routed_to_hr": state.get("routed_to_hr", False),
            "conversation_id": conversation_id,
        })

    return app


# --------------------------------------------------------------------------- HTML page
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HR Policy Assistant</title>
<style>
  :root {
    --bg:#f4f5f7; --panel:#ffffff; --ink:#161513; --muted:#6b6b6b;
    --brand:#0c447c; --brand2:#185fa5; --user:#0c447c; --border:#e2e2df;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family:'Segoe UI',system-ui,-apple-system,Arial,sans-serif;
         background:var(--bg); color:var(--ink); height:100vh; display:flex; flex-direction:column; }
  header { background:var(--panel); border-bottom:1px solid var(--border); padding:14px 20px;
           display:flex; align-items:center; gap:12px; }
  header .logo { width:34px; height:34px; border-radius:8px; background:var(--brand);
                 color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; }
  header h1 { font-size:16px; margin:0; }
  header .status { margin-left:auto; font-size:12px; color:var(--muted); }
  #chat { flex:1; overflow-y:auto; padding:20px; max-width:860px; width:100%; margin:0 auto; }
  .msg { display:flex; margin:12px 0; gap:10px; }
  .msg .avatar { width:30px; height:30px; border-radius:50%; flex:0 0 30px; display:flex;
                 align-items:center; justify-content:center; font-size:13px; font-weight:700; color:#fff; }
  .msg.user { flex-direction:row-reverse; }
  .msg.user .avatar { background:#5f5e5a; }
  .msg.bot .avatar { background:var(--brand); }
  .bubble { padding:12px 14px; border-radius:12px; max-width:78%; line-height:1.5; font-size:14px;
            box-shadow:0 1px 2px rgba(0,0,0,.05); word-wrap:break-word; overflow-wrap:anywhere; }
  .msg.user .bubble { background:var(--user); color:#fff; border-bottom-right-radius:4px; }
  .msg.bot .bubble { background:var(--panel); border:1px solid var(--border); border-bottom-left-radius:4px; }
  .bubble details { margin-top:6px; }
  .bubble hr { border:none; border-top:1px solid var(--border); margin:8px 0 0; }
  .hint { color:var(--muted); font-size:13px; text-align:center; margin:6px 0 16px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-bottom:8px; }
  .chip { background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:6px 12px;
          font-size:13px; cursor:pointer; color:var(--brand2); }
  .chip:hover { background:#eef4fb; }
  .typing span { display:inline-block; width:6px; height:6px; margin:0 1px; border-radius:50%;
                 background:var(--muted); animation:blink 1.2s infinite both; }
  .typing span:nth-child(2){animation-delay:.2s} .typing span:nth-child(3){animation-delay:.4s}
  @keyframes blink { 0%,80%,100%{opacity:.2} 40%{opacity:1} }
  footer { background:var(--panel); border-top:1px solid var(--border); padding:12px 20px; }
  .composer { max-width:860px; margin:0 auto; display:flex; gap:10px; }
  #q { flex:1; resize:none; padding:11px 14px; border:1px solid var(--border); border-radius:10px;
       font-size:14px; font-family:inherit; max-height:120px; }
  #send { background:var(--brand); color:#fff; border:none; border-radius:10px; padding:0 20px;
          font-size:14px; font-weight:600; cursor:pointer; }
  #send:disabled { opacity:.5; cursor:default; }
  @media (max-width:600px){ .bubble{max-width:88%} }
</style>
</head>
<body>
  <header>
    <div class="logo">HR</div>
    <h1>HR Policy Assistant</h1>
    <div class="status" id="status"></div>
  </header>

  <div id="chat">
    <div class="hint">Ask about company HR policy — leave, pay, benefits, conduct, and more.</div>
    <div class="chips" id="chips">
      <div class="chip">How many PL days do I accrue?</div>
      <div class="chip">What is the bereavement leave policy?</div>
      <div class="chip">How do I see my pay stub?</div>
      <div class="chip">¿Cuándo es el día de pago?</div>
    </div>
  </div>

  <footer>
    <div class="composer">
      <textarea id="q" rows="1" placeholder="Type your question and press Enter…"></textarea>
      <button id="send">Send</button>
    </div>
  </footer>

<script>
const chat = document.getElementById('chat');
const q = document.getElementById('q');
const send = document.getElementById('send');
let conversationId = null;
let busy = false;

fetch('/api/config').then(r=>r.json()).then(c=>{
  const parts = ['LLM: '+c.provider + (c.model?(' ('+c.model+')'):'')];
  if (c.mock_rag) parts.push('mock RAG');
  if (c.fast_mode) parts.push('fast');
  document.getElementById('status').textContent = parts.join(' · ');
}).catch(()=>{});

function el(tag, cls, html){ const e=document.createElement(tag); if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e; }

function addMessage(role, html){
  const m = el('div','msg '+role);
  m.appendChild(el('div','avatar', role==='user'?'You':'HR'));
  const b = el('div','bubble', html);
  m.appendChild(b);
  chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight;
  return b;
}

function escapeHtml(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

async function ask(text){
  if(busy || !text.trim()) return;
  busy = true; send.disabled = true;
  const chips = document.getElementById('chips'); if(chips) chips.remove();
  addMessage('user', escapeHtml(text));
  const thinking = addMessage('bot', '<span class="typing"><span></span><span></span><span></span></span>');
  try {
    const res = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message:text, conversation_id:conversationId})
    });
    const data = await res.json();
    if(data.error){ thinking.innerHTML = '⚠️ '+escapeHtml(data.error); }
    else { conversationId = data.conversation_id; thinking.innerHTML = data.response || '(no answer)'; }
  } catch(e){
    thinking.innerHTML = '⚠️ '+escapeHtml(String(e));
  } finally {
    busy = false; send.disabled = false; q.focus();
    chat.scrollTop = chat.scrollHeight;
  }
}

send.addEventListener('click', ()=>{ const t=q.value; q.value=''; ask(t); });
q.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); const t=q.value; q.value=''; ask(t); }
});
q.addEventListener('input', ()=>{ q.style.height='auto'; q.style.height=Math.min(q.scrollHeight,120)+'px'; });
document.addEventListener('click', e=>{ if(e.target.classList.contains('chip')) ask(e.target.textContent); });
q.focus();
</script>
</body>
</html>
"""


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the HR Policy Agent web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("The web UI needs uvicorn. Install it with:  pip install '.[web]'")
        return 1

    settings = get_settings()
    print(f"HR Policy Agent web UI → http://{args.host}:{args.port}")
    print(f"  LLM provider: {settings.llm_provider} | fast={settings.fast_mode} | "
          f"mock RAG={settings.use_mock_rag}")
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
