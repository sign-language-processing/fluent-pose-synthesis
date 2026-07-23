"""Generate the self-contained review artifact HTML from review_data.json."""
import json
import sys

data_path, out_path = sys.argv[1], sys.argv[2]
data = json.load(open(data_path))
payload = json.dumps(data).replace("</", "<\\/")

CSS = r"""
:root{
  --ground:#f4f6f8; --surface:#ffffff; --surface-2:#fbfcfd;
  --ink:#151a21; --muted:#5c6673; --faint:#8a95a3; --hair:#e3e7ec;
  --teal:#0f766e; --slate:#5b6b7f; --amber:#b45309;
  --teal-soft:#e2f3f1; --amber-soft:#f7ecdd; --slate-soft:#eceff3;
  --shadow:0 1px 2px rgba(16,24,40,.04),0 6px 20px rgba(16,24,40,.06);
  --radius:14px;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0b0e12; --surface:#141920; --surface-2:#10151b;
  --ink:#e8ebf0; --muted:#9aa5b2; --faint:#6b7684; --hair:#242c36;
  --teal:#2dd4bf; --slate:#9fb0c3; --amber:#f5b556;
  --teal-soft:#123330; --amber-soft:#33260f; --slate-soft:#1b222b;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 26px rgba(0,0,0,.35);
}}
:root[data-theme="light"]{
  --ground:#f4f6f8; --surface:#ffffff; --surface-2:#fbfcfd;
  --ink:#151a21; --muted:#5c6673; --faint:#8a95a3; --hair:#e3e7ec;
  --teal:#0f766e; --slate:#5b6b7f; --amber:#b45309;
  --teal-soft:#e2f3f1; --amber-soft:#f7ecdd; --slate-soft:#eceff3;
  --shadow:0 1px 2px rgba(16,24,40,.04),0 6px 20px rgba(16,24,40,.06);
}
:root[data-theme="dark"]{
  --ground:#0b0e12; --surface:#141920; --surface-2:#10151b;
  --ink:#e8ebf0; --muted:#9aa5b2; --faint:#6b7684; --hair:#242c36;
  --teal:#2dd4bf; --slate:#9fb0c3; --amber:#f5b556;
  --teal-soft:#123330; --amber-soft:#33260f; --slate-soft:#1b222b;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 26px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px}

/* top bar */
.topbar{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--surface) 88%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--hair)}
.topbar .wrap{display:flex;align-items:center;gap:18px;height:60px;flex-wrap:wrap}
.brand{font-weight:650;letter-spacing:-.01em}
.brand small{display:block;font-weight:450;color:var(--muted);font-size:12px;letter-spacing:.02em}
.legend{display:flex;gap:14px;margin-left:auto;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
.dot{width:9px;height:9px;border-radius:3px;display:inline-block}
.controls{display:flex;gap:8px}
button{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);
  border:1px solid var(--hair);border-radius:9px;padding:7px 12px;cursor:pointer;transition:border-color .15s,background .15s}
button:hover{border-color:var(--teal)}
button:focus-visible{outline:2px solid var(--teal);outline-offset:2px}

/* intro */
.intro{padding:30px 0 6px}
.intro h1{font-size:26px;line-height:1.2;margin:0 0 8px;letter-spacing:-.02em;text-wrap:balance}
.intro p{margin:0;color:var(--muted);max-width:64ch}

/* card */
.card{background:var(--surface);border:1px solid var(--hair);border-radius:var(--radius);
  box-shadow:var(--shadow);margin:20px 0;overflow:hidden}
.card-head{padding:18px 22px;border-bottom:1px solid var(--hair)}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.sentence{font-size:20px;line-height:1.35;margin:6px 0 4px;letter-spacing:-.01em;text-wrap:balance}
.german{font-size:13.5px;color:var(--muted);font-style:italic}
.section-label{font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--faint);
  padding:16px 22px 10px;display:flex;align-items:center;gap:8px}
.section-label .rule{height:1px;background:var(--hair);flex:1}

/* signs strip */
.signs{display:flex;gap:12px;overflow-x:auto;padding:0 22px 20px;scrollbar-width:thin}
.chip{flex:0 0 auto;width:118px}
.chip video{width:118px;height:118px;object-fit:cover;background:var(--surface-2);
  border:1px solid var(--hair);border-radius:10px;display:block}
.chip.crop video{border-style:dashed;border-color:var(--amber)}
.chip .g{font-size:11.5px;margin-top:6px;color:var(--ink);word-break:break-word}
.chip .m{font-size:11px;color:var(--muted);line-height:1.3}
.chip .src{font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;margin-top:3px;color:var(--faint)}
.chip.crop .src{color:var(--amber)}

/* comparison grid */
.cmp{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:0 22px 22px}
.pane{border:1px solid var(--hair);border-radius:12px;overflow:hidden;background:var(--surface-2)}
.pane.fluent{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal)}
.pane .bar{display:flex;align-items:center;justify-content:space-between;
  padding:9px 12px;border-bottom:1px solid var(--hair)}
.pane .name{display:flex;align-items:center;gap:8px;font-size:12.5px;font-weight:600}
.pane .tag{font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:2px 7px;border-radius:999px;
  background:var(--teal);color:#fff}
.pane .frames{font-size:11.5px;color:var(--muted)}
.pane video{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:var(--surface-2)}
.pane.naive .name{color:var(--slate)} .pane.naive .bar{background:var(--slate-soft)}
.pane.fluent .name{color:var(--teal)} .pane.fluent .bar{background:var(--teal-soft)}
.pane.gold .name{color:var(--amber)} .pane.gold .bar{background:var(--amber-soft)}

.foot{color:var(--muted);font-size:12.5px;text-align:center;padding:26px 0 40px}
@media (max-width:760px){.cmp{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('review-data').textContent);
const app = document.getElementById('app');
const vid = (src,cls)=>`<video class="${cls||''}" src="${src}" autoplay loop muted playsinline preload="auto"></video>`;
const esc = s => (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
app.innerHTML = DATA.map((s,i)=>`
  <article class="card">
    <div class="card-head">
      <div class="eyebrow">Sentence ${String(i+1).padStart(2,'0')}</div>
      <div class="sentence">${esc(s.english)}</div>
      <div class="german">${esc(s.german)}</div>
    </div>
    <div class="section-label">Building blocks · dictionary signs + sentence crops (dashed)<span class="rule"></span></div>
    <div class="signs">
      ${s.signs.map(g=>`<div class="chip ${g.source==='sentence'?'crop':''}">${vid(g.video)}
        <div class="g mono">${esc(g.gloss)}</div><div class="m">${esc(g.meaning)}</div>
        <div class="src">${g.source==='sentence'?'from sentence':'dictionary'}</div></div>`).join('')}
    </div>
    <div class="section-label">Stitched sequence · compare<span class="rule"></span></div>
    <div class="cmp">
      <div class="pane naive"><div class="bar"><span class="name"><span class="dot" style="background:var(--slate)"></span>Naive</span>
        <span class="frames mono">${s.frames.naive}f</span></div>${vid(s.naive)}</div>
      <div class="pane fluent"><div class="bar"><span class="name"><span class="dot" style="background:var(--teal)"></span>Fluent <span class="tag">ours</span></span>
        <span class="frames mono">${s.frames.fluent}f</span></div>${vid(s.fluent)}</div>
      <div class="pane gold"><div class="bar"><span class="name"><span class="dot" style="background:var(--amber)"></span>Gold</span>
        <span class="frames mono">${s.frames.gold}f</span></div>${vid(s.gold)}</div>
    </div>
  </article>`).join('');

// theme toggle: auto -> light -> dark
const root=document.documentElement, tb=document.getElementById('theme');
const modes=['auto','light','dark']; let mi=0;
tb.onclick=()=>{mi=(mi+1)%3;const m=modes[mi];
  if(m==='auto')root.removeAttribute('data-theme');else root.setAttribute('data-theme',m);
  tb.textContent='Theme: '+m;};
// replay all
document.getElementById('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play();});
"""

HTML = f"""<title>Stitching review — DGS Corpus</title>
<style>{CSS}</style>
<header class="topbar"><div class="wrap">
  <div class="brand">Pose-stitching review<small>DGS Corpus · 10 sentences</small></div>
  <div class="legend">
    <span><span class="dot" style="background:var(--faint)"></span>signs</span>
    <span><span class="dot" style="background:var(--slate)"></span>naive</span>
    <span><span class="dot" style="background:var(--teal)"></span>fluent (ours)</span>
    <span><span class="dot" style="background:var(--amber)"></span>gold</span>
  </div>
  <div class="controls"><button id="replay">Replay all</button><button id="theme">Theme: auto</button></div>
</div></header>
<main class="wrap">
  <section class="intro">
    <h1>Stitching quality review</h1>
    <p>Ten DGS-Corpus sentences. For each: the isolated dictionary signs it was built
    from, then the <strong>naive</strong> concatenation, our <strong>fluent</strong>
    stitch, and the <strong>gold</strong> real signing — all pose-rendered as one
    anonymized signer so you compare motion, not appearance. Videos loop automatically.</p>
  </section>
  <div id="app"></div>
  <div class="foot">Naive = spoken-to-signed baseline · Fluent = anonymize + segmentation trim + idle-hand removal + Butterworth + duration cap · Gold = corpus reference</div>
</main>
<script id="review-data" type="application/json">{payload}</script>
<script>{JS}</script>
"""

open(out_path,"w").write(HTML)
print(f"wrote {out_path}: {len(HTML)/1e6:.1f} MB")
