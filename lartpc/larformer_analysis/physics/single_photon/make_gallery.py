#!/usr/bin/env python3
"""Build the single-photon plot gallery page (published as a claude.ai artifact).
    python3 make_gallery.py <out_dir>   -> <out_dir>/single_photon_gallery.html + gallery_files.json
Sections are listed in SECTIONS (id, title, blurb, glob list, published subfolder)."""
import glob, os, sys, json, re, html
SP = "plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_reserved"
SECTIONS = [
  ("vpass-ccnumu", "Vertex + veto-pass, CC-numu-like (chain BDT_numu <= 0.1), final step", "lar truth, RESERVED halves. Final: signal 13.1 (1g+X 10.8, entering 2.3) | CC numu 31.6 | >=2g 7.8 | entering bkg 3.0.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_ccnumu/final/*.png")), "vpass_ccnumu/final"),
  ("vpass-nonccnumu", "Vertex + veto-pass, non-CC-numu (chain BDT_numu > 0.1), final step", "lar truth, RESERVED halves. Final: signal 88.2 (1g0X 14.9, 1g+X 25.9, entering 47.4) | CC numu 4.9 | >=2g 21.0 | entering bkg 11.4.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu/final/*.png")), "vpass_nonccnumu/final"),
  ("veto2g-study", "Second-photon veto pair-geometry study (vertex stage, before the veto)", "MC prongs by truth class (fragment / pi0 partner / other nu photon / nu non-photon / unmatched) + EXT, data points; 'vetofiring' = photon-PID prongs with E > 10 MeV (the ones the current veto uses). Table + variants in veto2g_study.txt.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/veto2g_study/scatter_*.png")) + sorted(glob.glob("plots_run1_novtx_dep_pe3/veto2g_study/vetofiring/*.png")), "veto2g"),
  ("final-infv-0p", "FINAL in-FV cell, 1g0p (no accepted proton), BDT_enter >= 0.15", "lar truth, RESERVED halves. Signal 16.8 (1g0X 6.3, 1g+X 0.9, entering 9.5) | >=2g 4.3 | entering bkg 1.8.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025_infv_0p/final/*.png")), "final_infv_0p/final"),
  ("final-infv-np", "FINAL in-FV cell, 1gNp (>= 1 accepted proton), BDT_enter >= 0.30", "lar truth, RESERVED halves. Signal 29.5 (1g0X 4.7, 1g+X 22.7, entering 2.1) | CC numu 2.0 | >=2g 12.5.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025_infv_np/final/*.png")), "final_infv_np/final"),
  ("presplit-0p", "0p sample BEFORE the entering split (vertex + veto-pass, lar BDT_numu > 0.25, no proton) + its entering scan", "lar truth, RESERVED halves. In-FV signal 9.5 vs entering 49.8, EXT 8.4. enter_split_scan.png: distFromWall vs BDT_enter curves for 0p; 2-D grid in enter_split_scan.txt.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025_0p/enter_split_scan.png")) + sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025_0p/final/*.png")), "presplit_0p"),
  ("final-infv", "FINAL in-FV cell: vertex + veto-pass, lar BDT_numu > 0.25, BDT_enter >= 0.30 (Np) / 0.15 (0p)", "lar truth, RESERVED halves + EXT half2 (1.202) + data. The MB-target sample.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025_infv/final/*.png")), "final_infv/final"),
  ("final-entering", "FINAL entering sideband: same chain, BDT_enter below the thresholds", "lar truth, RESERVED halves + EXT half2 (1.202) + data. Entering-photon constraint sample.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025_entering/final/*.png")), "final_entering/final"),
  ("split-scan", "In-FV / entering split scan on the adopted cell (vertex + veto-pass, lar BDT_numu > 0.25)", "distFromWall vs BDT_enter (lar-trained and mb-trained): in-FV signal kept vs entering removed, 1g0X alone, and purity vs efficiency. Table in enter_split_scan.txt.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025/enter_split_scan.png")), "vpass_nonccnumu_lar025/scan"),
  ("vpass-ccnumu-lar025", "Vertex + veto-pass, CC-numu-like with the LAR-trained BDT_numu (<= 0.25), final step", "lar truth, RESERVED halves, model noflash_novtx_lar, cut 0.25.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_ccnumu_lar025/final/*.png")), "vpass_ccnumu_lar025/final"),
  ("vpass-nonccnumu-lar025", "Vertex + veto-pass, non-CC-numu with the LAR-trained BDT_numu (> 0.25), final step", "lar truth, RESERVED halves, model noflash_novtx_lar, cut 0.25.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar025/final/*.png")), "vpass_nonccnumu_lar025/final"),
  ("vpass-ccnumu-lar", "Vertex + veto-pass, CC-numu-like with the LAR-trained BDT_numu (<= 0.1), final step", "lar truth, RESERVED halves, model noflash_novtx_lar. Final: signal 6.3 (1g+X 5.0, entering 1.3) | CC numu 28.4 | >=2g 5.0 | entering bkg 2.3.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_ccnumu_lar/final/*.png")), "vpass_ccnumu_lar/final"),
  ("vpass-nonccnumu-lar", "Vertex + veto-pass, non-CC-numu with the LAR-trained BDT_numu (> 0.1), final step", "lar truth, RESERVED halves, model noflash_novtx_lar. Final: signal 94.9 (1g0X 14.9, 1g+X 31.7, entering 48.4) | CC numu 8.1 | >=2g 23.8 | entering bkg 12.2.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_nonccnumu_lar/final/*.png")), "vpass_nonccnumu_lar/final"),
  ("vpass-final", "Vertex + veto-pass cell, final step (S6)", "lar truth, RESERVED overlay half + EXT half2 (scale 1.202) + bnb5e19 data. Chain BDT scores from the noflash_novtx model, flattened (signal-uniform).",
   sorted(glob.glob(f"{SP}/final/*.png")), "vpass/final"),
  ("vpass-prechi2", "Vertex + veto-pass cell, before the flash chi2 cut (preChi2)", "Same cell and samples, one step earlier.",
   sorted(glob.glob(f"{SP}/preChi2/*.png")), "vpass/preChi2"),
  ("chain-novtx", "MB BDT chain, noflash_novtx model", "Scores at pre-selection (P0) and after the previous steps (prev), data/pred ratio panels; final energy spectra (reported, not cut).",
   sorted(glob.glob("plots_run1_mbchain/score_*_noflash_novtx.png")) + sorted(glob.glob("plots_run1_mbchain/final_*_E_noflash_novtx.png")), "chain/noflash_novtx"),
  ("chain-flash", "MB BDT chain, flash model", "Same plots with the flash variables (log chi2, PE ratio) included in the BDTs.",
   sorted(glob.glob("plots_run1_mbchain/score_*_flash.png")) + sorted(glob.glob("plots_run1_mbchain/final_*_E_flash.png")), "chain/flash"),
  ("study-vpass", "Variable study at the vertex + veto-pass stage (lar)", "Diagnostic BDT score distributions, odd/even folds; see the text block for AUCs and top variables.",
   sorted(glob.glob("plots_run1_mbsel_vars_vpass_lar/*.png")), "study/vpass"),
  ("paths6", "Six-path photon energy stacks (lar suite)", "Candidate photon energy per leaf path at the final step.",
   sorted(glob.glob("plots_run1_novtx_dep_pe3/paths6/*.png")), "paths6"),
]
PAGE_HEAD = '''<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Code+Pro:wght@400;500&display=swap">
<style>
/* layout: sticky section nav + filter on top; each section a responsive card grid of figures; text blocks in monospace */
:root { --bg:#f6f5f2; --fg:#1e2126; --muted:#5d6470; --line:#d8d5cc; --card:#ffffff; --accent:#8a3b12; --code:#efede7;
  --sans:"Source Sans 3", "Segoe UI", system-ui, sans-serif; --mono:"Source Code Pro", ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#17191d; --fg:#e8e6e1; --muted:#9aa0a8; --line:#33373d; --card:#1f2227; --accent:#e0905a; --code:#23262b; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#17191d; --fg:#e8e6e1; --muted:#9aa0a8; --line:#33373d; --card:#1f2227; --accent:#e0905a; --code:#23262b; color-scheme: dark }
body { background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.45; margin:0; }
.wrap { max-width:1500px; margin:0 auto; padding-inline:16px; padding-block:12px 48px; }
h1 { font-size:1.5rem; margin:0 0 4px; text-wrap:balance; }
.sub { color:var(--muted); margin:0 0 12px; }
.bar { position:sticky; top:env(safe-area-inset-top, 0px); background:var(--bg); border-bottom:1px solid var(--line); padding-block:8px; z-index:2; display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; }
.bar nav { display:flex; flex-wrap:wrap; gap:4px 14px; }
.bar a { color:var(--accent); text-decoration:none; font-weight:600; font-size:.9rem; }
.bar a:hover, .bar a:focus-visible { text-decoration:underline; outline:none; }
.bar input { font:inherit; padding:4px 8px; border:1px solid var(--line); border-radius:4px; background:var(--card); color:var(--fg); min-width:0; width:220px; max-width:100%; }
section { margin-top:28px; }
h2 { font-size:1.15rem; margin:0 0 2px; text-wrap:balance; }
.blurb { color:var(--muted); margin:0 0 10px; max-width:80ch; }
.count { font-family:var(--mono); font-size:.8rem; color:var(--muted); }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap:14px; }
.card { margin:0; background:var(--card); border:1px solid var(--line); border-radius:6px; padding:8px; min-width:0; }
.card img { display:block; width:100%; height:auto; max-width:100%; }
.card figcaption { font-family:var(--mono); font-size:.78rem; color:var(--muted); margin-top:6px; overflow-wrap:anywhere; }
.card[hidden] { display:none !important; }
pre { font-family:var(--mono); font-size:.76rem; background:var(--code); border:1px solid var(--line); border-radius:6px; padding:10px; overflow-x:auto; line-height:1.4; }
details { margin-top:10px; } summary { cursor:pointer; color:var(--accent); font-weight:600; }
@media (prefers-reduced-motion: no-preference) { .card img { transition: transform .15s ease; } .card a:hover img { transform: scale(1.01); } }
</style>
'''
PAGE_SCRIPT = '''<script>
(function() {
  var inp = document.getElementById("filt");
  var cards = Array.prototype.slice.call(document.querySelectorAll(".card"));
  function apply() {
    var q = inp.value.trim().toLowerCase();
    cards.forEach(function(c) { c.hidden = q !== "" && c.getAttribute("data-name").indexOf(q) === -1; });
    document.querySelectorAll("section").forEach(function(s) {
      s.querySelector(".count").textContent = s.querySelectorAll(".card:not([hidden])").length + " plots";
    });
  }
  inp.addEventListener("input", apply);
  try { var saved = localStorage.getItem("spg-filter"); if (saved) { inp.value = saved; apply(); } } catch (e) {}
  inp.addEventListener("change", function() { try { localStorage.setItem("spg-filter", inp.value); } catch (e) {} });
})();
</script>
'''

ROOT_B = "plots_run1_bdtpaths_pe3"
CELLS_B = [("vertex_fail", "vertex, veto FAIL (sideband)"), ("vertex_pass", "vertex, veto PASS"),
           ("vertex_pass_ccnumu", "vertex_pass, CC-numu-like (BDT_numu <= 0.25)"), ("vertex_pass_nonccnumu", "vertex_pass, non-CC-numu (BDT_numu > 0.25)"),
           ("vertex_pass_nonccnumu_entering", "non-CC-numu, ENTERING sideband (BDT_enter below 0.30 Np / 0.15 0p)"),
           ("vertex_pass_nonccnumu_infv", "non-CC-numu, IN-FV (MB target)"), ("vertex_pass_nonccnumu_infv_0p", "in-FV, 1g0p"),
           ("vertex_pass_nonccnumu_infv_np", "in-FV, 1gNp"), ("novertex_fail", "no-vertex, veto FAIL"), ("novertex_pass", "no-vertex, veto PASS (entering sideband)")]
SECTIONS_B = [("tree", "Tree table stacks: candidate photon energy per cell (geometric veto, lar truth, RESERVED halves)",
               "Veto = photon prong E > 10 MeV AND axes cross within 100 cm of both starts; chain BDT_numu (lar) > 0.25; BDT_enter >= 0.30 Np / 0.15 0p. Table in tree/bdt_paths.txt.",
               sorted(glob.glob(f"{ROOT_B}/tree/*.png")), "tree")] + [
              (f"cell-{c}", f"Diagnostics, final step: {lab}", "", sorted(glob.glob(f"{ROOT_B}/diagnostics_{c}/final/*.png")), f"cells/{c}") for c, lab in CELLS_B]


ROOT_G = "plots_isogamma"
CELLS_G = [("all_selected", "ALL selected (base S1-S6, both paths)")] + CELLS_B
SECTIONS_G = [("cutflow", "Cutflow + BDT-path tree occupancy (unit weights; fractions of truth denominators)",
               "Isotropic 1-gamma gun (mcc9.10, 53,938 events, simulated cosmics). Same selection as the run-1 BDT-path suite. Tables in cutflow/isogamma_cutflow.txt and cutflow/isogamma_losses.txt.",
               sorted(glob.glob(f"{ROOT_G}/cutflow/*.png")), "cutflow"),
              ("epid", "Electron-PID loss study: photon showers tagged as electrons (gun vs held-out run-1 overlay)",
               "One row per lar-visible true photon converting >= 10 cm inside the TPC, matched to its nu-stream shower. Summary in epid/epid_summary.txt.",
               sorted(glob.glob(f"{ROOT_G}/epid/*.png")), "epid")] + [
              (f"cell-{c}", f"Diagnostics, final step: {lab}", "", sorted(glob.glob(f"{ROOT_G}/diagnostics_{c}/final/*.png")), f"cells/{c}") for c, lab in CELLS_G]


def main(out, suite="main"):
    global SECTIONS, ROOT_B
    page_name, title_txt, intro = "single_photon_gallery.html", "Single-photon plot gallery", "Run-1, lar truth. Regenerated from the single_photon folder (make_gallery.py); every figure opens full size in a new tab."
    if suite == "bdtpaths":
        SECTIONS = SECTIONS_B; page_name = "bdtpaths_gallery.html"; title_txt = "BDT-Path Suite Gallery"
        intro = "Geometric second-photon veto + chain-BDT path tree (2026-10-08), lar truth, RESERVED overlay half + EXT half2 (x1.202) + bnb5e19 data. Tree table: " + ROOT_B + "/tree/bdt_paths.txt."
    if suite == "isogamma":
        SECTIONS = SECTIONS_G; page_name = "isogamma_gallery.html"; title_txt = "Photon Gun Suite Gallery"; suite = "bdtpaths"; ROOT_B = ROOT_G
        intro = "Isotropic single-photon gun through the run-1 single-photon selection (2026-10-10): cutflow, electron-PID loss study, and the per-cell diagnostics (unit weights, MC only). Tables: " + ROOT_G + "/cutflow/*.txt, " + ROOT_G + "/epid/epid_summary.txt."
    files, cards = {}, {}
    for sid, title, blurb, paths, pub in SECTIONS:
        items = []
        for p in paths:
            key = p if suite == "bdtpaths" else f"{pub}/{os.path.basename(p)}"; files[key] = p
            items.append((os.path.splitext(os.path.basename(p))[0], key))
        cards[sid] = items
    if suite == "bdtpaths":
        tree_txt = []
        for tf in (f"{ROOT_B}/tree/bdt_paths.txt", f"{ROOT_B}/cutflow/isogamma_cutflow.txt", f"{ROOT_B}/cutflow/isogamma_losses.txt", f"{ROOT_B}/epid/epid_summary.txt"):
            if os.path.exists(tf):
                tree_txt += [f"##### {tf}"] + open(tf).read().splitlines() + [""]
        esc = html.escape
        nav = "".join(f'<a href="#{sid}">{esc(t)}</a>' for sid, t, *_ in SECTIONS)
        body = []
        for sid, title, blurb, paths, pub in SECTIONS:
            figs = "".join(f'<figure class="card" data-name="{esc(n.lower())}"><a href="{k}" target="_blank" rel="noopener"><img loading="lazy" src="{k}" alt="{esc(n)}"></a><figcaption>{esc(n)}</figcaption></figure>' for n, k in cards[sid])
            body.append(f'<section id="{sid}"><header><h2>{esc(title)}</h2><p class="blurb">{esc(blurb)} <span class="count">{len(cards[sid])} plots</span></p></header><div class="grid">{figs}</div></section>')
        page = PAGE_HEAD.replace("__TITLE__", title_txt) + f'<div class="wrap"><h1>{esc(title_txt)}</h1><p class="sub">{esc(intro)}</p>' + \
               f'<div class="bar"><nav>{nav}</nav><label for="filt" class="count">filter</label><input id="filt" type="search" placeholder="variable name, e.g. bdtChain" autocomplete="off"></div>' + \
               f'<details open><summary>Tables</summary><pre>{esc(chr(10).join(tree_txt))}</pre></details>' + "".join(body) + "</div>" + PAGE_SCRIPT
        open(os.path.join(out, page_name), "w").write(page)
        json.dump(files, open(os.path.join(out, "gallery_files_bdtpaths.json"), "w"), indent=0)
        print(len(files), "files;", round(sum(os.path.getsize(v) for v in files.values()) / 1e6, 2), "MB")
        return
    med = open(f"{SP}/medians.txt").read().splitlines()
    i = max(k for k, l in enumerate(med) if l.startswith("== "))
    final_med = [l for l in med[i:] if re.match(r"^  (bdtChain|distFromWall|startDwall|vtxDwall|nProtons|E |mupiKEmax)", l)]
    blocks = []
    for c, lab in (("ccnumu", "CC-numu-like cell (mb-trained BDT)"), ("nonccnumu", "non-CC-numu cell (mb-trained BDT)"), ("ccnumu_lar", "CC-numu-like cell (lar-trained BDT)"), ("nonccnumu_lar", "non-CC-numu cell (lar-trained BDT)")):
        m = open(f"plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_{c}/medians.txt").read().splitlines()
        j = max(k for k, l in enumerate(m) if l.startswith("== "))
        blocks.append((f"{lab}: final-step medians (chain scores, wall distances, protons)",
                       [l for l in m[j:] if re.match(r"^  (bdtChain|distFromWall|startDwall|vtxDwall|nProtons|mupiKEmax)", l)]))
    study = open("plots_run1_mbsel_vars_vpass_lar/mb_variable_study.txt").read().splitlines()
    keep, take = [], 0
    for l in study:
        if l.startswith("MB variable study") or l.startswith("  signal composition") or l.startswith("  nu background composition"):
            keep.append(l)
        elif l.startswith("== "):
            keep += ["", l]; take = 6
        elif take > 0:
            keep.append(l); take -= 1
    esc = html.escape
    nav = "".join(f'<a href="#{sid}">{esc(t)}</a>' for sid, t, *_ in SECTIONS)
    body = []
    for sid, title, blurb, paths, pub in SECTIONS:
        figs = "".join(f'<figure class="card" data-name="{esc(n.lower())}"><a href="{k}" target="_blank" rel="noopener"><img loading="lazy" src="{k}" alt="{esc(n)}"></a><figcaption>{esc(n)}</figcaption></figure>' for n, k in cards[sid])
        body.append(f'<section id="{sid}"><header><h2>{esc(title)}</h2><p class="blurb">{esc(blurb)} <span class="count">{len(cards[sid])} plots</span></p></header><div class="grid">{figs}</div></section>')
    det = "".join(f'<details><summary>{esc(t)}</summary><pre>{esc(chr(10).join(ls))}</pre></details>' for t, ls in blocks)
    page = f'''''' + PAGE_HEAD.replace('__TITLE__', 'Single-Photon Plot Gallery') + f'''<div class="wrap">
<h1>Single-photon plot gallery</h1>
<p class="sub">Run-1, lar truth. Regenerated 2026-10-05 from the single_photon folder (make_gallery.py); every figure opens full size in a new tab.</p>
<div class="bar"><nav>{nav}</nav><label for="filt" class="count">filter</label><input id="filt" type="search" placeholder="variable name, e.g. bdtChain" autocomplete="off"></div>
{det}
<details><summary>Vertex + veto-pass (whole cell), final step: weighted medians per category</summary><pre>{esc(chr(10).join(final_med))}</pre></details>
<details><summary>Variable study at the vertex + veto-pass stage: AUCs and top variables</summary><pre>{esc(chr(10).join(keep))}</pre></details>
{"".join(body)}
</div>
''' + PAGE_SCRIPT + ''''''
    open(os.path.join(out, "single_photon_gallery.html"), "w").write(page)
    json.dump(files, open(os.path.join(out, "gallery_files.json"), "w"), indent=0)
    print(len(files), "files;", round(sum(os.path.getsize(v) for v in files.values()) / 1e6, 2), "MB")
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "main")
