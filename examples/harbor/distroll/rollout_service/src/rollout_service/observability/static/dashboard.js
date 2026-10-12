const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "—").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const n = (value) => Number(value || 0).toLocaleString();
const pct = (value) => value == null ? "—" : `${Number(value).toFixed(1)}%`;
const bytes = (value) => { let x=Number(value||0), i=0; const u=["B","KiB","MiB","GiB","TiB"]; while(x>=1024&&i<u.length-1){x/=1024;i++;} return `${x.toFixed(i<3?1:2)} ${u[i]}`; };
const short = (value, size=10) => { const s=String(value||""); return s.length>size ? `${s.slice(0,size)}…` : s || "—"; };
const age = (stamp) => { if(!stamp)return "—"; const s=Math.max(0,(Date.now()-new Date(stamp).getTime())/1000); return s<60?`${s.toFixed(0)}s`:s<3600?`${(s/60).toFixed(1)}m`:`${(s/3600).toFixed(1)}h`; };
const bar = (label, used, detail, physical=false) => `<div><div class="bar-label"><span>${esc(label)}</span><strong>${esc(detail)}</strong></div><div class="bar ${physical?'physical':''}"><i style="width:${Math.max(0,Math.min(100,Number(used||0)))}%"></i></div></div>`;

function renderKpis(data){
  const c=data.cluster||{}, r=c.recent_15m||{}, done=Number(r.SUCCEEDED||0), failed=Number(r.FAILED||0)+Number(r.REJECTED||0), rate=done+failed?`${(done/(done+failed)*100).toFixed(1)}%`:'—';
  const cards=[
    ["CPU Accepted Queue",n(c.queued_attempts),`${n(c.reserved_executions)} reserved (RL pending queue is separate)`,"rgba(217,119,6,.10)"],
    ["Oldest Wait",c.oldest_queue_age_seconds==null?'—':age(new Date(Date.now()-c.oldest_queue_age_seconds*1000).toISOString()),c.oldest_queued_at?'queue is waiting':'queue is empty',"rgba(217,119,6,.10)"],
    ["Active Attempts",n(c.active_attempts),`${n(c.cleanup_pending_attempts)} cleanup pending`,"rgba(37,99,235,.10)"],
    ["Ready Workers",`${n(c.ready_workers)} / ${n(c.total_workers)}`,"Control heartbeat view","rgba(22,163,74,.10)"],
    ["Success attempts · 15m",rate,`all runs · ${done} success · ${failed} failed`,"rgba(8,145,178,.10)"],
    ["Budget-ended success · 15m",n((data.diagnostics||{}).budget_ended_succeeded_15m),"max_turns accepted · reward may be 0", "rgba(8,145,178,.10)"],
  ];
  $("kpis").innerHTML=cards.map(x=>`<article class="kpi" style="--accent:${x[3]}"><span class="kpi-label">${x[0]}</span><strong class="kpi-value">${x[1]}</strong><span class="kpi-sub">${x[2]}</span></article>`).join("");
}

function renderWorkers(data){
  $("workers").innerHTML=(data.workers||[]).map(w=>{
    const control=w.control||{}, node=w.node||{}, cpu=node.cpu||{}, mem=node.memory||{}, docker=node.docker||{}, images=node.images||{}, out=node.outbox||{};
    const state=String(control.state||"NOT_READY"), vc=control.virtual_capacity||{}, ac=control.allocated_charge||{};
    const vcpu=vc.cpu_units?ac.cpu_units/vc.cpu_units*100:0, vmem=vc.memory_mb?ac.memory_mb/vc.memory_mb*100:0;
    const disk=(node.filesystems||[]).find(f=>(f.paths||[]).some(p=>p.purpose==='docker_root'))||{};
    const states=Object.entries(w.state_counts||{}).filter(x=>x[1]).map(x=>`<span class="state-chip">${esc(x[0])} · ${n(x[1])}</span>`).join("")||'<span class="state-chip">IDLE</span>';
    return `<article class="worker-card"><div class="worker-head"><div><h3>${esc(w.worker_id)}</h3><p class="worker-meta">heartbeat ${age((w.database||{}).last_heartbeat)} · observer ${age((w.source||{}).last_success_at)}</p></div><span class="badge ${state.toLowerCase().replace('_','-')}">${esc(state)}</span></div><div class="state-row">${states}</div><div class="bar-block">${bar('Virtual CPU',vcpu,`${n(ac.cpu_units)} / ${n(vc.cpu_units)}`)}${bar('Virtual Memory',vmem,`${n(ac.memory_mb)} / ${n(vc.memory_mb)} MiB`)}${bar('Physical CPU',cpu.used_percent,pct(cpu.used_percent),true)}${bar('Physical Memory',mem.used_percent,`${pct(mem.used_percent)} · ${bytes(mem.available_bytes)} free`,true)}${bar('Docker Disk',disk.used_percent,`${bytes(disk.free_bytes)} free`,true)}</div><div class="metric-grid"><div class="metric"><span>Active Attempts</span><strong>${n(w.active_attempts)}</strong></div><div class="metric"><span>Containers</span><strong>${n(docker.managed_running_containers)}</strong></div><div class="metric"><span>Managed Images</span><strong>${n(images.managed_present)}</strong></div><div class="metric"><span>Load 1m</span><strong>${cpu.load1==null?'—':Number(cpu.load1).toFixed(2)}</strong></div><div class="metric"><span>Pulling / Failed</span><strong>${n(images.pulling)} / ${n(images.failed)}</strong></div><div class="metric"><span>Outbox</span><strong>${n(out.pending)} / ${n(out.rejected)}</strong></div></div></article>`;
  }).join("");
}

function renderAttempts(data){
  const rows=(data.active_attempts||[]); $("attempt-count").textContent=rows.length;
  $("attempts").innerHTML=rows.length?rows.map(a=>`<tr><td>${esc(a.worker_id)}</td><td class="state ${String(a.state).toLowerCase()}">${esc(a.state)}</td><td title="${esc(a.client_scope)}">${short(a.client_scope,14)}</td><td class="task" title="${esc(a.task_ref)}">${esc(a.task_ref)}</td><td title="${esc(a.attempt_id)}">${short(a.attempt_id,12)}</td><td>${age(a.claimed_at||a.assigned_at||a.updated_at)}</td></tr>`).join(""):`<tr><td colspan="6" class="empty">No active attempts</td></tr>`;
}

function renderFailures(data){
  const rows=data.recent_failures||[];
  $("failures").innerHTML=rows.length?rows.slice(0,20).map(a=>`<tr><td>${esc(a.worker_id)}</td><td class="state ${String(a.state).toLowerCase()}">${esc(a.state)}</td><td title="${esc(a.client_scope)}">${esc(String(a.client_scope||'—').slice(-20))}</td><td class="task" title="${esc(a.task_ref)}">${esc(a.task_ref)}</td><td title="${esc(a.failure_class)} · ${esc(a.failure_stage)}">${esc(a.failure_reason)}<br><small>${esc(a.error_code||a.failure_class||'—')}</small></td><td>${age(a.updated_at)}</td></tr>`).join(""):`<tr><td colspan="6" class="empty">No recent failures</td></tr>`;
}

function renderAlerts(data){ const rows=data.alerts||[]; $("alert-count").textContent=rows.length; $("alerts").innerHTML=rows.length?rows.map(a=>`<div class="alert ${a.level}"><strong>${esc(a.code)}${a.worker_id?` · ${esc(a.worker_id)}`:''}</strong>${esc(a.message)}</div>`).join(""):'<div class="empty">No active alerts</div>'; }
function renderSources(data){ const s=data.sources||{}, rows=[["Control capacity",s.capacity],["PostgreSQL",s.database],...Object.entries(s.nodes||{}).map(x=>[`Node · ${x[0]}`,x[1]])]; $("sources").innerHTML=rows.map(([name,x])=>{x=x||{}; const fresh=x.available&&!x.error, label=fresh?`${Number(x.age_seconds||0).toFixed(1)}s`:(x.error||'unavailable'); return `<div class="source"><span>${esc(name)}</span><small class="${fresh?'fresh':'stale'}">${esc(label)}</small></div>`;}).join(""); }

function linePath(values,max,w=900,h=210,pad=16){ if(!values.length)return ''; return values.map((v,i)=>`${i?'L':'M'} ${pad+(w-2*pad)*(i/Math.max(1,values.length-1))} ${h-pad-(h-2*pad)*(v/max)}`).join(' '); }
function renderTrend(history){ const points=history.points||[], q=points.map(x=>Number(x.queued_attempts||0)), a=points.map(x=>Number(x.active_attempts||0)), max=Math.max(1,...q,...a); const grid=[.25,.5,.75].map(y=>`<line class="chart-grid" x1="16" y1="${210*y}" x2="884" y2="${210*y}"/>`).join(''); $("trend-chart").innerHTML=`${grid}<path class="chart-queue" d="${linePath(q,max)}"/><path class="chart-active" d="${linePath(a,max)}"/>`; }

async function refresh(){
  try{
    const [snapshot,history]=await Promise.all([fetch('/api/v1/snapshot',{cache:'no-store'}),fetch('/api/v1/history?window=30m',{cache:'no-store'})]);
    if(!snapshot.ok)throw new Error(`snapshot HTTP ${snapshot.status}`);
    const data=await snapshot.json(); renderKpis(data); renderWorkers(data); renderAttempts(data); renderFailures(data); renderAlerts(data); renderSources(data); if(history.ok)renderTrend(await history.json());
    $("last-refresh").textContent=new Date(data.sampled_at).toLocaleString(); $("live-label").textContent=(data.alerts||[]).some(x=>x.level==='critical')?'ATTENTION':'LIVE'; $("live-dot").className=`live-dot ${(data.alerts||[]).some(x=>x.level==='critical')?'bad':'ok'}`;
  }catch(error){ $("live-label").textContent='DISCONNECTED'; $("last-refresh").textContent=String(error); $("live-dot").className='live-dot bad'; }
}
refresh(); setInterval(refresh,3000);
