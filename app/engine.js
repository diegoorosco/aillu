/* Calidad que paga - motor offline (sin dependencias).
   Vision: segmentacion clasica + red convolucional pequena (pesos en granos.bin).
   Logica: equivalencias SCA, grado con incertidumbre, precio de referencia.
   Lenguaje: lector de SMS (TF-IDF de caracteres + regresion logistica + reglas). */
(function (root) {
  'use strict';
  const S = 96, FILL = 0.62;
  const CLASSES = ['sano','negro','vinagre','cereza_seca','hongo','broca_severa','broca_leve','concha','inmaduro','pergamino','partido_cortado'];

  // ---------- RNG ----------
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function randn(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function gamma(r, k) { // Marsaglia-Tsang
    if (k < 1) { const u = r(); return gamma(r, k + 1) * Math.pow(u || 1e-12, 1 / k); }
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) { let x, v; do { x = randn(r); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v; if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
  }

  // ---------- CNN ----------
  function loadWeights(buf, manifest) {
    const f = new Float32Array(buf); const W = { S: 96 };
    for (const m of manifest) {
      if (m.name === '_S') { W.S = m.shape[0]; continue; }
      const n = m.shape.reduce((a, b) => a * b, 1); W[m.name] = { d: f.subarray(m.offset, m.offset + n), s: m.shape };
    }
    return W;
  }
  function conv3(x, C, H, Wd, w, b, O) { // densa 3x3, pad 1, + sesgo
    const out = new Float32Array(O * H * Wd), HW = H * Wd;
    for (let o = 0; o < O; o++) {
      const op = out.subarray(o * HW, (o + 1) * HW); op.fill(b ? b[o] : 0);
      for (let c = 0; c < C; c++) {
        const xp = x.subarray(c * HW, (c + 1) * HW), wo = (o * C + c) * 9;
        for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
          const wv = w[wo + ky * 3 + kx]; if (wv === 0) continue; const dy = ky - 1, dx = kx - 1;
          const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(Wd, Wd - dx);
          for (let y = y0; y < y1; y++) { const ro = y * Wd, ri = (y + dy) * Wd + dx; for (let xx = x0; xx < x1; xx++) op[ro + xx] += wv * xp[ri + xx]; }
        }
      }
    }
    return out;
  }
  function dw3(x, C, H, Wd, w) {
    const out = new Float32Array(C * H * Wd), HW = H * Wd;
    for (let c = 0; c < C; c++) {
      const xp = x.subarray(c * HW, (c + 1) * HW), op = out.subarray(c * HW, (c + 1) * HW);
      for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
        const wv = w[c * 9 + ky * 3 + kx], dy = ky - 1, dx = kx - 1;
        const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(Wd, Wd - dx);
        for (let y = y0; y < y1; y++) { const ro = y * Wd, ri = (y + dy) * Wd + dx; for (let xx = x0; xx < x1; xx++) op[ro + xx] += wv * xp[ri + xx]; }
      }
    }
    return out;
  }
  function pw1(x, C, HW, w, b, O) {
    const out = new Float32Array(O * HW);
    for (let o = 0; o < O; o++) { const op = out.subarray(o * HW, (o + 1) * HW); op.fill(b[o]);
      for (let c = 0; c < C; c++) { const wv = w[o * C + c], xp = x.subarray(c * HW, (c + 1) * HW); for (let i = 0; i < HW; i++) op[i] += wv * xp[i]; } }
    return out;
  }
  function relu(a) { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; }
  function maxpool(x, C, H, Wd) {
    const h = H >> 1, w = Wd >> 1, out = new Float32Array(C * h * w);
    for (let c = 0; c < C; c++) for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) {
      const i = c * H * Wd + 2 * y * Wd + 2 * xx; out[c * h * w + y * w + xx] = Math.max(x[i], x[i + 1], x[i + Wd], x[i + Wd + 1]);
    }
    return out;
  }
  function forward(W, x) { // x: Float32Array 3*S*S normalizado; devuelve probabilidades
    let C = 3, H = W.S || S;
    for (let bi = 0; bi < 5; bi++) {
      const O = W['b' + bi + '_c_w'].s[0];
      x = relu(conv3(x, C, H, H, W['b' + bi + '_c_w'].d, W['b' + bi + '_c_b'].d, O));
      x = dw3(x, O, H, H, W['b' + bi + '_dw_w'].d);
      x = relu(pw1(x, O, H * H, W['b' + bi + '_pw_w'].d, W['b' + bi + '_pw_b'].d, O));
      x = maxpool(x, O, H, H); C = O; H = H >> 1;
    }
    const g = new Float32Array(C); for (let c = 0; c < C; c++) { let s = 0; for (let i = 0; i < H * H; i++) s += x[c * H * H + i]; g[c] = s / (H * H); }
    const fw = W.fc_w.d, fb = W.fc_b.d, K = fb.length, z = new Float32Array(K);
    for (let k = 0; k < K; k++) { let s = fb[k]; for (let c = 0; c < C; c++) s += fw[k * C + c] * g[c]; z[k] = s; }
    const m = Math.max(...z); let t = 0; const p = Array.from(z, v => { const e = Math.exp(v - m); t += e; return e; });
    return p.map(v => v / t);
  }

  // ---------- Segmentacion ----------
  function blur5(g, w, h) { // gaussiana 5x5 sigma 1.1 separable, borde reflejado
    const k = [1, 4, 6, 4, 1].map((_, i) => Math.exp(-((i - 2) ** 2) / (2 * 1.1 * 1.1))); const ks = k.reduce((a, b) => a + b); for (let i = 0; i < 5; i++) k[i] /= ks;
    const t = new Float32Array(w * h), o = new Float32Array(w * h), cl = (v, n) => v < 0 ? -v : v >= n ? 2 * n - v - 2 : v;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let s = 0; for (let i = -2; i <= 2; i++) s += k[i + 2] * g[y * w + cl(x + i, w)]; t[y * w + x] = s; }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let s = 0; for (let i = -2; i <= 2; i++) s += k[i + 2] * t[cl(y + i, h) * w + x]; o[y * w + x] = s; }
    return o;
  }
  function otsu(g) {
    const hist = new Float64Array(256); for (let i = 0; i < g.length; i++) hist[Math.max(0, Math.min(255, Math.round(g[i])))]++;
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sB = 0, wB = 0, best = 0, th = 128; const N = g.length;
    for (let t = 0; t < 256; t++) { wB += hist[t]; if (!wB) continue; const wF = N - wB; if (!wF) break; sB += t * hist[t];
      const mB = sB / wB, mF = (sum - sB) / wF, v = wB * wF * (mB - mF) ** 2; if (v > best) { best = v; th = t; } }
    return th;
  }
  function morph(b, w, h, r, isMax) { // rectangulo (2r+1)^2 separable
    const t = new Uint8Array(w * h), o = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = isMax ? 0 : 1; for (let i = -r; i <= r; i++) { const xx = x + i; if (xx < 0 || xx >= w) continue; const q = b[y * w + xx]; if (isMax ? q > v : q < v) v = q; } t[y * w + x] = v; }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = isMax ? 0 : 1; for (let i = -r; i <= r; i++) { const yy = y + i; if (yy < 0 || yy >= h) continue; const q = t[yy * w + x]; if (isMax ? q > v : q < v) v = q; } o[y * w + x] = v; }
    return o;
  }
  function segment(rgba, w, h) {
    const N = w * h, g = new Float32Array(N);
    for (let i = 0; i < N; i++) g[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
    const gb = blur5(g, w, h), th = otsu(gb);
    let b = new Uint8Array(N); for (let i = 0; i < N; i++) b[i] = Math.round(gb[i]) > th ? 0 : 1;
    b = morph(morph(b, w, h, 2, false), w, h, 2, true);   // apertura 5x5
    b = morph(morph(b, w, h, 3, true), w, h, 3, false);   // cierre 7x7
    const lab = new Int32Array(N), comps = []; const st = new Int32Array(N);
    for (let i = 0; i < N; i++) if (b[i] && !lab[i]) {
      const id = comps.length + 1; let sp = 0; st[sp++] = i; lab[i] = id; const px = [];
      while (sp) { const p = st[--sp]; px.push(p); const y = (p / w) | 0, x = p - y * w;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const yy = y + dy, xx = x + dx; if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue; const q = yy * w + xx; if (b[q] && !lab[q]) { lab[q] = id; st[sp++] = q; } } }
      comps.push(px);
    }
    const areas = comps.map(c => c.length).sort((a, b) => a - b);
    const med = (a) => a.length ? (a.length % 2 ? a[a.length >> 1] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2) : 0;
    let m; if (areas.length > 4) { const p50 = med(areas); m = med(areas.filter(a => a > p50)); } else m = med(areas);
    const keep = comps.filter(c => c.length > 0.2 * m && c.length < 3 * m);
    const hs = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)]; let nb = 0;
    for (let i = 0; i < N; i++) if (!b[i]) { nb++; hs[0][rgba[4 * i]]++; hs[1][rgba[4 * i + 1]]++; hs[2][rgba[4 * i + 2]]++; }
    const gain = hs.map(hh => { let c = 0; for (let v = 0; v < 256; v++) { c += hh[v]; if (c >= nb / 2) return 235 / Math.max(v, 1); } return 1; });
    const beans = keep.map(px => {
      let sx = 0, sy = 0; for (const p of px) { sy += (p / w) | 0; sx += p % w; } const cx = sx / px.length, cy = sy / px.length;
      let a = 0, bb = 0, c = 0; for (const p of px) { const y = ((p / w) | 0) - cy, x = (p % w) - cx; a += x * x; bb += x * y; c += y * y; }
      const ang = 0.5 * Math.atan2(2 * bb, a - c), ux = Math.cos(ang), uy = Math.sin(ang);
      let mn1 = 1e9, mx1 = -1e9, mn2 = 1e9, mx2 = -1e9, x0 = 1e9, x1 = -1, y0 = 1e9, y1 = -1;
      for (const p of px) { const y = (p / w) | 0, x = p % w, dx = x - cx, dy = y - cy, t1 = dx * ux + dy * uy, t2 = -dx * uy + dy * ux;
        if (t1 < mn1) mn1 = t1; if (t1 > mx1) mx1 = t1; if (t2 < mn2) mn2 = t2; if (t2 > mx2) mx2 = t2; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      const L = Math.max(mx1 - mn1 + 1, mx2 - mn2 + 1), m1 = (mn1 + mx1) / 2, m2 = (mn2 + mx2) / 2;
      return { cx: cx + m1 * ux - m2 * uy, cy: cy + m1 * uy + m2 * ux, L, area: px.length, box: [x0, y0, x1, y1] };
    });
    return { beans, gain, threshold: th };
  }
  function cropTensor(rgba, w, h, bean, gain, Sz) { // 3xSxS RGB normalizado, bilineal, borde replicado
    Sz = Sz || S;
    const s = FILL * Sz / bean.L, out = new Float32Array(3 * Sz * Sz);
    for (let y = 0; y < Sz; y++) for (let x = 0; x < Sz; x++) {
      let sx = (x - Sz / 2) / s + bean.cx, sy = (y - Sz / 2) / s + bean.cy;
      sx = Math.min(w - 1, Math.max(0, sx)); sy = Math.min(h - 1, Math.max(0, sy));
      const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1), fx = sx - x0, fy = sy - y0;
      for (let ch = 0; ch < 3; ch++) {
        const v = (rgba[4 * (y0 * w + x0) + ch] * (1 - fx) + rgba[4 * (y0 * w + x1) + ch] * fx) * (1 - fy) + (rgba[4 * (y1 * w + x0) + ch] * (1 - fx) + rgba[4 * (y1 * w + x1) + ch] * fx) * fy;
        const vb = Math.min(255, Math.round(v * gain[ch]));
        out[ch * Sz * Sz + y * Sz + x] = (vb / 255 - 0.5) / 0.25;
      }
    }
    return out;
  }

  // ---------- Grado y precio ----------
  function gradeFrom(cfg, full, cat1) { for (const g of cfg.grados) if (full <= g.max_defectos && (g.cat1_permitidos || cat1 < 0.5)) return g.id; }
  function evaluate(cfg, counts, nDud, draws, seed) {
    draws = draws || 4000; seed = seed || 7;
    const r = mulberry32(seed), N = cfg.granos_en_350g, dist = {}; cfg.grados.forEach(g => dist[g.id] = 0);
    const alpha = CLASSES.map(c => (counts[c] || 0) + (c === 'sano' ? 1 : 0.02)); const fulls = [];
    for (let d = 0; d < draws; d++) {
      const gs = alpha.map(a => gamma(r, a)), t = gs.reduce((a, b) => a + b, 0); let full = 0, cat1 = 0;
      CLASSES.forEach((c, i) => { const info = cfg.clases[c]; if (!info.cat) return; const p = gs[i] / t; full += p * N / info.granos_por_defecto; if (info.cat === 1) cat1 += p * N; });
      dist[gradeFrom(cfg, full, cat1)]++; fulls.push(full);
    }
    for (const k in dist) dist[k] /= draws; fulls.sort((a, b) => a - b);
    const best = Object.keys(dist).reduce((a, b) => dist[a] >= dist[b] ? a : b), n = Object.values(counts).reduce((a, b) => a + b, 0);
    let motivo = null, seguro = dist[best] >= cfg.umbral_confianza_grado && nDud / Math.max(n, 1) <= cfg.umbral_dudosos;
    if (nDud / Math.max(n, 1) > cfg.umbral_dudosos) motivo = 'granos_dudosos'; else if (dist[best] < cfg.umbral_confianza_grado) motivo = 'grado_incierto';
    if (best === 'especial') { seguro = false; motivo = 'especial_requiere_catador'; }
    return { grado: best, prob: dist[best], dist, seguro, motivo, n, nDud, full_med: fulls[draws >> 1], full_p10: fulls[Math.floor(draws * 0.1)], full_p90: fulls[Math.floor(draws * 0.9)] };
  }
  function priceRange(cfg, res) {
    const p = cfg.precio, gs = Object.keys(res.dist).filter(g => res.dist[g] >= 0.15); if (!gs.length) gs.push(res.grado);
    const v = gs.map(g => (p.arabica_usd_kg + p.diferencial_usd_kg_oro[g]) * p.rendimiento_oro_pergamino * p.tipo_cambio * p.participacion_productor * 46);
    return [Math.round(Math.min(...v) / 10) * 10, Math.round(Math.max(...v) / 10) * 10];
  }

  // ---------- SMS ----------
  const NUM_ES = { medio: 0.5, un: 1, uno: 1, una: 1, dos: 2, tres: 3, cuatro: 4, cinco: 5, seis: 6, siete: 7, ocho: 8, nueve: 9, diez: 10, once: 11, doce: 12, quince: 15, veinte: 20, treinta: 30, cuarenta: 40, cincuenta: 50, cien: 100, ciento: 100 };
  const NUM_QU = { huk: 1, hoq: 1, iskay: 2, ishkay: 2, kimsa: 3, kinsa: 3, tawa: 4, pichqa: 5, pisqa: 5, suqta: 6, soqta: 6, qanchis: 7, qanchik: 7, pusaq: 8, pusac: 8, isqun: 9, isqon: 9, chunka: 10, chunca: 10, pachak: 100, pachaq: 100 };
  const UNITS = [['quintal', /\b(q\.?q\.?|qq|quint\w*|kintal\w*)\b/], ['arroba', /(\barroba\w*|@)/], ['kg', /\b(kg|kgs|kilo\w*|quilo\w*)\b/], ['saco', /\b(saco\w*|costal\w*)\b/], ['lata', /\b(lata\w*)\b/]];
  const KG = { quintal: 46, arroba: 11.5, kg: 1, lata: 2.5, saco: null };
  const MESES = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','setiembre','octubre','noviembre','diciembre'];
  const norm = t => t.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  const w2n = tok => { if (tok in NUM_ES) return NUM_ES[tok]; if (tok in NUM_QU) return NUM_QU[tok]; const t = tok.replace(/n+$/, ''); return t in NUM_QU ? NUM_QU[t] : null; };
  const TOKRX = /\d+(?:[.,]\d+)?|[a-z\u00f1]+/g, isD = s => /^\d/.test(s);
  function numbers(t) {
    const toks = t.match(TOKRX) || [], out = []; let i = 0;
    while (i < toks.length) { const tk = toks[i]; let v = isD(tk) ? parseFloat(tk.replace(',', '.')) : w2n(tk);
      if (v !== null && v !== undefined) { let j = i;
        if (j + 1 < toks.length && (toks[j + 1] === 'chunka' || toks[j + 1] === 'chunca') && v < 10 && !isD(tk)) { v *= 10; j++; }
        if ((toks[j] === 'chunka' || toks[j] === 'chunca') && j + 1 < toks.length && (w2n(toks[j + 1]) ?? 99) < 10) { v += w2n(toks[j + 1]); j++; }
        if (j + 2 < toks.length && toks[j + 1] === 'y' && (toks[j + 2] === 'medio' || toks[j + 2] === 'media')) { v += 0.5; j += 2; }
        else if (j + 3 < toks.length && toks[j + 2] === 'y' && (toks[j + 3] === 'medio' || toks[j + 3] === 'media')) v += 0.5;
        out.push([i, v, tk]); i = j; }
      i++; }
    return [toks, out];
  }
  function mes(t, hoy) {
    for (let k = 0; k < 12; k++) { const m = MESES[k]; if (new RegExp('\\b' + m.slice(0, 4)).test(t) || (m === 'setiembre' && t.includes('septiem'))) return k + 1; }
    if (/todos ?santos|noviem/.test(t)) return 11;
    if (/fin de mes|esta semana|la otra semana|proxima semana|semana que viene|ya\b|ahora|listo|hoy|manana|kunan|paqarin/.test(t)) return hoy;
    return null;
  }
  function makeIntent(model) {
    const idx = new Map(model.vocab.map((v, i) => [v, i]));
    return function (text) {
      const t = norm(text).replace(/\s\s+/g, ' '), counts = new Map();
      for (const w0 of t.split(/\s+/).filter(Boolean)) { const w = ' ' + w0 + ' ';
        for (let n = model.ngram[0]; n <= model.ngram[1]; n++) { let off = 0; const add = g => { const k = idx.get(g); if (k !== undefined) counts.set(k, (counts.get(k) || 0) + 1); };
          add(w.slice(off, off + n)); while (off + n < w.length) { off++; add(w.slice(off, off + n)); } } }
      let nn = 0; const feats = []; for (const [k, c] of counts) { const v = (1 + Math.log(c)) * model.idf[k]; feats.push([k, v]); nn += v * v; } nn = Math.sqrt(nn) || 1;
      const z = model.intercept.map((b, ci) => { let s = b; for (const [k, v] of feats) s += model.coef[ci][k] * v / nn; return s; });
      const m = Math.max(...z); const e = z.map(v => Math.exp(v - m)); const tot = e.reduce((a, b) => a + b, 0);
      const p = e.map(v => v / tot); const k = p.indexOf(Math.max(...p)); return [model.classes[k], p[k]];
    };
  }
  function parseSMS(text, clf, hoy) {
    hoy = hoy || 10;
    const t = norm(text); const nr = numbers(t), toks = nr[0], nums = nr[1]; let unidad = null;
    for (const [u, rx] of UNITS) if (rx.test(t)) { unidad = u; break; }
    let cand = [];
    for (const [i, v] of nums) { const prev = i > 0 ? toks[i - 1] : '', nxt = toks[i + 1] || '';
      if (['socio','soc','codigo','cod','n','nro'].includes(prev)) continue;
      if (nxt === 'de' && toks[i + 2] && MESES.some(m => toks[i + 2].startsWith(m.slice(0, 4)))) continue;
      if (v >= 1000) continue; cand.push([i, v]); }
    while (cand.length > 1 && cand[0][0] <= 2 && isD(toks[cand[0][0]]) && !/\b(a|y|o|entre)\b/.test(toks.slice(cand[0][0] + 1, cand[1][0]).join(' '))) cand = cand.slice(1);
    if (cand.length > 1 && isD(toks[cand[0][0]]) && cand[0][0] <= 2 && toks[cand[0][0] + 1] === 'a') cand = cand.slice(1);
    let cantidad = null, rango = null;
    if (cand.length) { const m = /\b(mejor|no son|corrijo|digo)\b/.exec(t);
      if (m) { const pos = (t.slice(0, m.index).match(TOKRX) || []).length; const after = cand.filter(c => c[0] >= pos).map(c => c[1]); cantidad = after.length ? after[0] : cand[cand.length - 1][1]; }
      else { cantidad = cand[0][1]; if (cand.length > 1 && /\b(a|y|o|entre)\b/.test(t) && cand[1][0] - cand[0][0] <= 2) { rango = [cand[0][1], cand[1][1]]; cantidad = (cand[0][1] + cand[1][1]) / 2; } } }
    const producto = /cerez/.test(t) ? 'cereza' : (/perga/.test(t) ? 'pergamino' : null);
    const ip = clf ? clf(text) : ['oferta', 1], intencion = ip[0], p = ip[1];
    let kg = null; if (cantidad !== null && unidad && KG[unidad]) kg = cantidad * KG[unidad] * (producto === 'cereza' && unidad !== 'lata' ? 0.2 : 1);
    const falta = []; if (intencion === 'oferta') { if (cantidad === null) falta.push('cantidad'); if (unidad === null) falta.push('unidad'); else if (unidad === 'saco') falta.push('unidad_ambigua'); }
    return { intencion, confianza: p, cantidad, rango, unidad, producto: producto || (intencion === 'oferta' ? 'pergamino' : null), mes: mes(t, hoy), kg_pergamino: kg, falta, requiere_humano: p < 0.6 || falta.length > 0 };
  }

  // ---------- Pronostico de acopio ----------
  function forecast(socios, reportes, draws, seed) {
    draws = draws || 4000; seed = seed || 3;
    const r = mulberry32(seed), cum = Object.fromEntries(socios.map(s => [s.id, s.cumplimiento_hist])), tot = new Float64Array(draws);
    for (const x of reportes) { if (x.grado_ultima_muestra === 'bajo' || x.grado_ultima_muestra === 'fuera') continue;
      const f = 0.7 * (cum[x.socio] ?? 0.88) + 0.3 * 0.88; for (let d = 0; d < draws; d++) tot[d] += x.qq * Math.min(1.3, Math.max(0, f + 0.12 * randn(r))); }
    const s = Array.from(tot).sort((a, b) => a - b);
    return { med: s[draws >> 1], p10: s[Math.floor(draws * 0.1)], p90: s[Math.floor(draws * 0.9)], probAt: L => s.filter(v => v >= L).length / draws };
  }

  const api = { S, CLASSES, loadWeights, forward, segment, cropTensor, evaluate, priceRange, makeIntent, parseSMS, forecast, mulberry32 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.QP = api;
})(typeof self !== 'undefined' ? self : this);
