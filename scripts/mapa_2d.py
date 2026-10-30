"""Mapa 2D de los pacientes por tipo de cáncer (PCA y t-SNE sobre RP-256 del stream).

    python scripts/mapa_2d.py <carpeta_gdc> [--out results/mapa] [--fig results/figures]
    python scripts/mapa_2d.py --desde results/mapa/proyecciones.npz   # rehace figuras sin el stream

Una pasada del stream de expresión completa (misma selección y consultas que
``scripts/experimento_completo.py``) da la proyección RP-256 de cada paciente.
Sobre ella:

- PCA 2D (distancias globales) y t-SNE 2D (vecindarios locales, lo que usa el k-NN).
- Predicción del tipo de cáncer para todos los pacientes con SimHash-256 +
  re-ranking RP (leave-one-out, voto k = 5); en las consultas fijas, también la
  del k-NN exacto.

Salidas: ``mapa.csv``, ``proyecciones.npz``, ``mapa_tipos.png``, ``mapa_errores.png``
y ``mapa_interactivo.html`` (Plotly desde cdnjs; requiere internet al abrirla).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def leave_one_out_predictions(bits: np.ndarray, P: np.ndarray, y: np.ndarray, *, b: int, f: int, k: int) -> np.ndarray:
    """Tipo predicho para cada paciente con SimHash-b + re-ranking RP, excluyéndolo de su búsqueda."""
    from tcgaknn.algoritmos import SimHashKNN
    from tcgaknn.algoritmos.coseno import pack_bits
    from tcgaknn.algoritmos.evaluacion import knn_vote

    packed = pack_bits(bits[:, :b])
    idx = SimHashKNN.from_signatures(packed, b, base=P, rerank_factor=f)
    nb, _, _ = idx.query_packed(packed, k + 1, Q_base=P)
    own = np.arange(len(y))[:, None]
    keep = np.where(nb == own, -1, nb)  # quita al propio paciente
    rows = []
    for r in keep:
        r = r[r >= 0][:k]
        rows.append(np.pad(r, (0, k - r.size), constant_values=-1))
    return knn_vote(np.stack(rows), y)


def embeddings(P: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=50, random_state=seed).fit(P)
    Z50 = pca.transform(P)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, learning_rate="auto").fit_transform(Z50)
    return Z50[:, :2], tsne, pca.explained_variance_ratio_[:2]


def palette(n: int) -> list[str]:
    import matplotlib

    cmaps = [matplotlib.colormaps[c] for c in ("tab20", "tab20b", "tab20c")]
    cols = [matplotlib.colors.to_hex(cm(i)) for cm in cmaps for i in range(20)]
    return [cols[(i * 7) % len(cols)] if n > 20 else cols[i] for i in range(n)]


def build(root: Path, out: Path, seed: int, b: int, m: int, f: int, k: int, n_queries: int) -> Path:
    from tcgaknn.completo import stream_pass

    data = stream_pass(root, n_queries=n_queries, seed=seed, b=b, m=m, log=lambda s: print(s, flush=True))
    y, P, bits, q = data["y"], data["P"], data["bits"], data["q"]
    print("Predicción leave-one-out para todos los pacientes", flush=True)
    pred = leave_one_out_predictions(bits, P, y, b=b, f=f, k=k)
    # k-NN exacto en las consultas (misma regla que el apéndice 010)
    from tcgaknn.algoritmos import topk
    from tcgaknn.algoritmos.evaluacion import knn_vote

    pred_ex = np.full(len(y), "", dtype=object)
    pred_ex[q] = knn_vote(topk(data["S"], k)[0], y[data["base"]])
    print("PCA y t-SNE", flush=True)
    pca2, tsne2, var = embeddings(P, seed)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "proyecciones.npz"
    np.savez_compressed(npz, P=P, y=y, barcodes=np.asarray(data["barcodes"]), pred_aprox=pred,
                        pred_exacto=pred_ex.astype(str), es_consulta=np.isin(np.arange(len(y)), q),
                        pca=pca2, tsne=tsne2, pca_var=var)
    return npz


def figures(npz: Path, out: Path, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz, allow_pickle=True)
    y, pred, bc = z["y"].astype(str), z["pred_aprox"].astype(str), z["barcodes"].astype(str)
    pca, tsne, var = z["pca"], z["tsne"], z["pca_var"]
    ok = pred == y
    types = sorted(set(y))
    col = dict(zip(types, palette(len(types))))
    fig_dir.mkdir(parents=True, exist_ok=True)

    # CSV para análisis
    with open(out / "mapa.csv", "w", encoding="utf-8") as fh:
        fh.write("barcode,tipo,pred_aprox_loo,acierto,es_consulta,pred_exacto,pca1,pca2,tsne1,tsne2\n")
        for i in range(len(y)):
            fh.write(f"{bc[i]},{y[i]},{pred[i]},{int(ok[i])},{int(z['es_consulta'][i])},{z['pred_exacto'][i]},"
                     f"{pca[i,0]:.4f},{pca[i,1]:.4f},{tsne[i,0]:.3f},{tsne[i,1]:.3f}\n")

    # PNG 1: PCA y t-SNE por tipo
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, E, title in ((axes[0], pca, f"PCA 2D (varianza explicada {100*var[0]:.1f} % + {100*var[1]:.1f} %)"),
                         (axes[1], tsne, "t-SNE 2D (vecindarios locales)")):
        for t in types:
            s = y == t
            ax.scatter(E[s, 0], E[s, 1], s=4, c=col[t], alpha=0.7, linewidths=0, label=t)
        for t in types:
            c = np.median(E[y == t], axis=0)
            ax.text(c[0], c[1], t, fontsize=8.5, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col[t], alpha=0.8))
        ax.set(title=title, xticks=[], yticks=[])
    axes[1].legend(markerscale=4, fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"Pacientes TCGA por tipo de cáncer — expresión completa, RP-256 desde el stream (n = {len(y):,})".replace(",", "."),
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "mapa_tipos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PNG 2: t-SNE con los mal clasificados resaltados
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(tsne[ok, 0], tsne[ok, 1], s=3, c=[col[t] for t in y[ok]], alpha=0.25, linewidths=0)
    ax.scatter(tsne[~ok, 0], tsne[~ok, 1], s=22, c=[col[t] for t in y[~ok]], edgecolors="black", linewidths=0.6,
               label=f"mal clasificados ({(~ok).sum():,} = {100*(~ok).mean():.1f} %)".replace(",", "."))
    for t in types:
        c = np.median(tsne[y == t], axis=0)
        ax.text(c[0], c[1], t, fontsize=8, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col[t], alpha=0.8))
    ax.set(xticks=[], yticks=[], title="t-SNE: pacientes mal clasificados por SimHash-256 + re-ranking (leave-one-out, k = 5)\n"
                                       "color = tipo real; borde negro = error")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "mapa_errores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    html(z, col, fig_dir / "mapa_interactivo.html")


def html(z, col: dict[str, str], path: Path) -> None:
    y, pred, bc = z["y"].astype(str), z["pred_aprox"].astype(str), z["barcodes"].astype(str)
    data = {
        "tipos": sorted(col), "colores": col,
        "y": y.tolist(), "pred": pred.tolist(), "bc": bc.tolist(),
        "ex": z["pred_exacto"].astype(str).tolist(), "q": z["es_consulta"].astype(int).tolist(),
        "pca": np.round(z["pca"], 4).tolist(), "tsne": np.round(z["tsne"], 3).tolist(),
    }
    acc = float(np.mean(y == pred))
    page = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace(
        "__ACC__", f"{100*acc:.1f}".replace(".", ",")).replace("__N__", f"{len(y):,}".replace(",", "."))
    path.write_text(page, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mapa de tipos de cáncer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#ffffff;--fg:#1f2328;--muted:#656d76;--panel:#f6f8fa;--border:#d0d7de}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--panel:#161b22;--border:#30363d}}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:12px 16px;border-bottom:1px solid var(--border)}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px}
.wrap{display:flex;gap:12px;padding:12px 16px;flex-wrap:wrap}
aside{flex:0 0 260px;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;font-size:13px;max-height:calc(100vh - 110px);overflow:auto}
#plot{flex:1 1 600px;min-height:75vh}
label{display:block;margin:4px 0}
.tipos label{display:flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
input[type=text]{width:100%;box-sizing:border-box;padding:6px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--fg)}
button{padding:4px 8px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--fg);cursor:pointer;margin:2px 2px 2px 0}
.sec{margin:10px 0 4px;font-weight:600}
#info{color:var(--muted);margin-top:8px}
@media (max-width:700px){aside{flex:1 1 100%;max-height:none}}
</style></head><body>
<header><h1>Pacientes TCGA por tipo de cáncer</h1>
<div class="sub">Expresión completa (20.531 genes) leída como stream → RP-256 → PCA / t-SNE. n = __N__ pacientes. Predicción: SimHash-256 + re-ranking RP, voto k = 5, leave-one-out (acierto __ACC__ %).</div></header>
<div class="wrap"><aside>
<div class="sec">Proyección</div>
<label><input type="radio" name="emb" value="tsne" checked> t-SNE (vecindarios)</label>
<label><input type="radio" name="emb" value="pca"> PCA (distancias globales)</label>
<div class="sec">Mostrar</div>
<label><input type="checkbox" id="soloErr"> Solo mal clasificados</label>
<label><input type="checkbox" id="soloQ"> Solo las 500 consultas</label>
<div class="sec">Buscar barcode</div>
<input type="text" id="buscar" placeholder="TCGA-XX-XXXX…">
<div class="sec">Tipos <button id="todos">todos</button><button id="ninguno">ninguno</button></div>
<div class="tipos" id="tipos"></div>
<div id="info"></div>
</aside><div id="plot"></div></div>
<script>
const D=__DATA__;
const dark=matchMedia('(prefers-color-scheme: dark)').matches;
const sel=new Set(D.tipos);
const box=document.getElementById('tipos');
D.tipos.forEach(t=>{const n=D.y.filter(v=>v===t).length;const l=document.createElement('label');
 l.innerHTML=`<input type="checkbox" checked data-t="${t}"><span class="sw" style="background:${D.colores[t]}"></span>${t} <span style="color:var(--muted)">(${n})</span>`;box.appendChild(l);});
box.addEventListener('change',e=>{const t=e.target.dataset.t;e.target.checked?sel.add(t):sel.delete(t);draw();});
document.getElementById('todos').onclick=()=>{D.tipos.forEach(t=>sel.add(t));box.querySelectorAll('input').forEach(i=>i.checked=true);draw();};
document.getElementById('ninguno').onclick=()=>{sel.clear();box.querySelectorAll('input').forEach(i=>i.checked=false);draw();};
document.querySelectorAll('input[name=emb]').forEach(r=>r.onchange=draw);
['soloErr','soloQ'].forEach(id=>document.getElementById(id).onchange=draw);
document.getElementById('buscar').oninput=draw;
function draw(){
 const emb=document.querySelector('input[name=emb]:checked').value, E=D[emb];
 const soloErr=document.getElementById('soloErr').checked, soloQ=document.getElementById('soloQ').checked;
 const term=document.getElementById('buscar').value.trim().toUpperCase();
 const traces=[];let shown=0,errs=0;const hits=[];
 const bg={x:[],y:[]};
 for(const t of D.tipos){const tr={x:[],y:[],text:[],mode:'markers',type:'scattergl',name:t,
   marker:{color:D.colores[t],size:[],line:{color:dark?'#fff':'#000',width:[]}},hovertemplate:'%{text}<extra></extra>'};
  D.y.forEach((v,i)=>{if(v!==t)return;const err=D.pred[i]!==v;
   const visible=sel.has(t)&&(!soloErr||err)&&(!soloQ||D.q[i]);
   if(!visible){bg.x.push(E[i][0]);bg.y.push(E[i][1]);return;}
   tr.x.push(E[i][0]);tr.y.push(E[i][1]);shown++;if(err)errs++;
   tr.marker.size.push(err?8:4);tr.marker.line.width.push(err?1:0);
   tr.text.push(`<b>${D.bc[i]}</b><br>tipo real: ${v}<br>predicho (aprox.): ${D.pred[i]} ${err?'✗':'✓'}`+(D.q[i]?`<br>consulta · predicho (exacto): ${D.ex[i]}`:''));
   if(term&&D.bc[i].toUpperCase().includes(term))hits.push(i);});
  if(tr.x.length)traces.push(tr);}
 traces.unshift({x:bg.x,y:bg.y,mode:'markers',type:'scattergl',name:'otros',hoverinfo:'skip',showlegend:false,
  marker:{color:dark?'#30363d':'#d8dde3',size:3}});
 if(hits.length)traces.push({x:hits.map(i=>E[i][0]),y:hits.map(i=>E[i][1]),mode:'markers+text',type:'scatter',name:'búsqueda',
  text:hits.map(i=>D.bc[i]),textposition:'top center',marker:{size:16,color:'rgba(0,0,0,0)',line:{color:'#e3116c',width:2.5}},hoverinfo:'skip'});
 const fg=dark?'#e6edf3':'#1f2328';
 Plotly.react('plot',traces,{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg},
  margin:{l:10,r:10,t:10,b:10},xaxis:{visible:false},yaxis:{visible:false,scaleanchor:'x'},showlegend:false,hovermode:'closest'},{responsive:true});
 document.getElementById('info').textContent=`${shown} pacientes visibles · ${errs} mal clasificados`+(term?` · ${hits.length} coincidencias`:'');
}
draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?")
    ap.add_argument("--desde", type=Path, help="proyecciones.npz ya calculadas (no lee el stream)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "mapa")
    ap.add_argument("--fig", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--queries", type=int, default=500)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.desde is None and a.root is None:
        ap.error("indica la carpeta de datos o --desde")
    npz = a.desde or build(a.root, a.out, a.seed, b=256, m=256, f=10, k=5, n_queries=a.queries)
    a.out.mkdir(parents=True, exist_ok=True)
    figures(npz, a.out, a.fig)
    print(f"Listo: {a.out}, {a.fig}")


if __name__ == "__main__":
    main()
