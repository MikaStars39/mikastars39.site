/* Single configurable chart for a paper figure.
   Two display modes:
     - smooth  : raw light line + EMA-smoothed dark line on top  (good for many points)
     - scatter : X markers + Catmull-Rom fitted curve            (good for sparse points)
   Data is supplied as a CSV string (first row = header; first column = x).
*/

/* ---------- CSV parser ---------- */
function parseCSV(text) {
  const lines = (text || "").split(/\r?\n/).map(l => l.trim()).filter(l => l.length && !l.startsWith("#"));
  if (lines.length < 2) return { xs: [], methodNames: [], series: {}, xKey: "x" };
  const split = (l) => l.split(",").map(s => s.trim());
  const header = split(lines[0]);
  const xKey = header[0] || "x";
  const methodNames = header.slice(1);
  const xs = [];
  const series = {};
  for (const m of methodNames) series[m] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = split(lines[i]);
    if (parts.length < 1) continue;
    const x = parseFloat(parts[0]);
    if (!Number.isFinite(x)) continue;
    xs.push(x);
    for (let j = 0; j < methodNames.length; j++) {
      const v = parseFloat(parts[j + 1]);
      series[methodNames[j]].push(Number.isFinite(v) ? v : null);
    }
  }
  return { xs, methodNames, series, xKey };
}

/* ---------- smoothing helpers ---------- */
function emaSmooth(arr, alpha) {
  if (!arr.length) return arr;
  if (alpha >= 1) return arr.slice();
  const fwd = new Array(arr.length);
  let s = arr[0];
  for (let i = 0; i < arr.length; i++) { s = alpha * arr[i] + (1 - alpha) * s; fwd[i] = s; }
  // mild two-sided pass to reduce lag
  const out = new Array(arr.length);
  let s2 = fwd[fwd.length - 1];
  for (let i = fwd.length - 1; i >= 0; i--) { s2 = 0.55 * fwd[i] + 0.45 * s2; out[i] = s2; }
  return out;
}

/* Catmull-Rom spline through points. Returns dense [{x,y}] list. */
function catmullRom(points, samples = 24) {
  if (points.length < 2) return points.slice();
  const out = [];
  const p = points;
  const n = p.length;
  for (let i = 0; i < n - 1; i++) {
    const p0 = p[i - 1] || p[i];
    const p1 = p[i];
    const p2 = p[i + 1];
    const p3 = p[i + 2] || p[i + 1];
    for (let j = 0; j < samples; j++) {
      const t = j / samples;
      const t2 = t * t, t3 = t2 * t;
      const x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3);
      const y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3);
      out.push({ x, y });
    }
  }
  out.push(p[n - 1]);
  return out;
}

/* ---------- Palette (paper colors → line + light pair) ---------- */
const PAPER_COLORS = [
  { line: "#55A868", light: "#AACDB4" },
  { line: "#3C88BE", light: "#9DC4DF" },
  { line: "#C44D52", light: "#E2A6A9" },
  { line: "#FF7F0F", light: "#FFBF87" },
  { line: "#9467BD", light: "#CAB3DE" },
  { line: "#927860", light: "#C9BCB0" },
];

function colorFor(i) { return PAPER_COLORS[i % PAPER_COLORS.length]; }

/* hex utilities — accept "#abc", "#aabbcc", "aabbcc"; return null if bad */
function hexToRgb(hex) {
  let h = String(hex || "").trim().replace(/^#/, "");
  if (h.length === 3) h = h.split("").map(c => c + c).join("");
  if (h.length !== 6 || !/^[0-9a-fA-F]{6}$/.test(h)) return null;
  const n = parseInt(h, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}
function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("");
}
function lightenHex(hex, amount = 0.55) {
  const rgb = hexToRgb(hex); if (!rgb) return hex;
  return rgbToHex(rgb.r + (255 - rgb.r) * amount, rgb.g + (255 - rgb.g) * amount, rgb.b + (255 - rgb.b) * amount);
}
function normalizeHex(s) {
  const rgb = hexToRgb(s); if (!rgb) return null;
  return rgbToHex(rgb.r, rgb.g, rgb.b);
}

window.normalizeHex = normalizeHex;
window.lightenHex = lightenHex;

/* ---------- "nice" tick generator ---------- */
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const range = hi - lo;
  const raw = range / n;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / pow;
  let step;
  if (norm < 1.5) step = 1 * pow;
  else if (norm < 3) step = 2 * pow;
  else if (norm < 7) step = 5 * pow;
  else step = 10 * pow;
  const start = Math.ceil(lo / step) * step;
  const out = [];
  for (let v = start; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/* ---------- ConfigurableChart ---------- */
function ConfigurableChart(props) {
  const {
    width = 640, height = 400,
    csv = "", mode = "smooth",
    xLabel = "Training steps", yLabel = "Value",
    xMin, xMax, yMin, yMax,             // numbers or "auto"/undefined
    smoothing = 0.10,                   // EMA alpha
    showDots = false,                   // circles on smooth-mode points
    dotRadius = 3,                      // radius of smooth-mode dots
    markerSize = 4,
    crashed = {},                       // { [methodName]: true } — draw X at end
    rawOpacity = 0.55,
    showGrid = true,
    legend = "br",                      // "br" | "tr" | "bl" | "tl" | "none"
    font = "'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif",
    colors: colorOverrides = {},        // { [methodName]: hexString }
  } = props;

  const resolveColor = (name, i) => {
    const override = colorOverrides[name];
    const norm = override ? normalizeHex(override) : null;
    if (norm) return { line: norm, light: lightenHex(norm, 0.6) };
    return colorFor(i);
  };

  const data = React.useMemo(() => parseCSV(csv), [csv]);
  const { xs, methodNames, series } = data;

  // domains
  const autoX = !xs.length ? [0, 1] : [Math.min(...xs), Math.max(...xs)];
  let allVals = [];
  for (const m of methodNames) allVals = allVals.concat(series[m].filter(v => v != null));
  const autoY = allVals.length ? [Math.min(...allVals), Math.max(...allVals)] : [0, 1];
  const yPad = (autoY[1] - autoY[0]) * 0.08 || 0.05;

  const _xMin = Number.isFinite(xMin) ? xMin : autoX[0];
  const _xMax = Number.isFinite(xMax) ? xMax : autoX[1];
  const _yMin = Number.isFinite(yMin) ? yMin : autoY[0] - yPad;
  const _yMax = Number.isFinite(yMax) ? yMax : autoY[1] + yPad;

  // layout
  const m = { top: 14, right: 16, bottom: 46, left: 60 };
  const pw = width - m.left - m.right;
  const ph = height - m.top - m.bottom;
  const xScale = (v) => m.left + ((v - _xMin) / (_xMax - _xMin)) * pw;
  const yScale = (v) => m.top + ph - ((v - _yMin) / (_yMax - _yMin)) * ph;

  // ticks
  const xT = niceTicks(_xMin, _xMax, 5);
  const yT = niceTicks(_yMin, _yMax, 5);
  const xFmt = (v) => {
    if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    if (Math.abs(v) >= 10) return String(Math.round(v));
    return String(+v.toFixed(2));
  };
  const yFmt = (v) => {
    const r = _yMax - _yMin;
    if (r >= 10) return String(Math.round(v));
    if (r >= 1) return v.toFixed(1);
    return v.toFixed(2);
  };

  // build per-method drawing instructions
  const renderSeries = () => {
    return methodNames.map((name, i) => {
      const col = resolveColor(name, i);
      const vals = series[name];
      // filter to in-range points (skip nulls)
      const pts = [];
      for (let k = 0; k < xs.length; k++) {
        if (vals[k] == null) continue;
        pts.push({ x: xs[k], y: vals[k] });
      }
      if (pts.length === 0) return null;

      if (mode === "scatter") {
        const smoothYs = emaSmooth(pts.map(p => p.y), Math.max(0.01, Math.min(1, smoothing)));
        const smoothPts = pts.map((p, j) => ({ x: p.x, y: smoothYs[j] }));
        const fit = catmullRom(smoothPts, 30);
        const fitD = fit.reduce((acc, p, j) => acc + `${j === 0 ? "M" : "L"}${xScale(p.x).toFixed(2)},${yScale(p.y).toFixed(2)}`, "");
        const crashSize = 6;
        const lastPt = pts[pts.length - 1];
        return (
          <g key={name}>
            <path d={fitD} fill="none" stroke={col.line} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
            <g stroke={col.line} strokeWidth="1.5" strokeLinecap="round" opacity={rawOpacity}>
              {pts.map((p, j) => {
                const cx = xScale(p.x), cy = yScale(p.y);
                return (
                  <g key={j}>
                    <line x1={cx - markerSize} y1={cy - markerSize} x2={cx + markerSize} y2={cy + markerSize} />
                    <line x1={cx - markerSize} y1={cy + markerSize} x2={cx + markerSize} y2={cy - markerSize} />
                  </g>
                );
              })}
            </g>
            {crashed[name] && lastPt && (
              <g stroke={col.line} strokeWidth="2.8" strokeLinecap="round">
                <line x1={xScale(lastPt.x) - crashSize} y1={yScale(lastPt.y) - crashSize} x2={xScale(lastPt.x) + crashSize} y2={yScale(lastPt.y) + crashSize} />
                <line x1={xScale(lastPt.x) - crashSize} y1={yScale(lastPt.y) + crashSize} x2={xScale(lastPt.x) + crashSize} y2={yScale(lastPt.y) - crashSize} />
              </g>
            )}
          </g>
        );
      }

      // smooth mode
      const ys = pts.map(p => p.y);
      const smoothYs = emaSmooth(ys, Math.max(0.01, Math.min(1, smoothing)));
      const rawD  = pts.reduce((acc, p, j) => acc + `${j === 0 ? "M" : "L"}${xScale(p.x).toFixed(2)},${yScale(p.y).toFixed(2)}`, "");
      const smD   = pts.reduce((acc, p, j) => acc + `${j === 0 ? "M" : "L"}${xScale(p.x).toFixed(2)},${yScale(smoothYs[j]).toFixed(2)}`, "");
      const crashSize = 6;
      const lastPt = pts[pts.length - 1];
      const lastSmY = smoothYs[smoothYs.length - 1];
      return (
        <g key={name}>
          <path d={rawD} fill="none" stroke={col.light} strokeWidth="0.9" strokeLinejoin="round" strokeLinecap="round" opacity={rawOpacity} />
          <path d={smD}  fill="none" stroke={col.line}  strokeWidth="2.1" strokeLinejoin="round" strokeLinecap="round" />
          {showDots && pts.map((p, j) => (
            <circle key={j} cx={xScale(p.x)} cy={yScale(smoothYs[j])} r={dotRadius} fill={col.line} stroke="none" />
          ))}
          {crashed[name] && lastPt && (
            <g stroke={col.line} strokeWidth="2.8" strokeLinecap="round">
              <line x1={xScale(lastPt.x) - crashSize} y1={yScale(lastSmY) - crashSize} x2={xScale(lastPt.x) + crashSize} y2={yScale(lastSmY) + crashSize} />
              <line x1={xScale(lastPt.x) - crashSize} y1={yScale(lastSmY) + crashSize} x2={xScale(lastPt.x) + crashSize} y2={yScale(lastSmY) - crashSize} />
            </g>
          )}
        </g>
      );
    });
  };

  // legend
  const renderLegend = () => {
    if (legend === "none" || !methodNames.length) return null;
    const lineLen = 22;
    const rowH = 18;
    const padX = 10, padY = 8;
    const labelW = Math.max(...methodNames.map((n) => n.length)) * 7.2;
    const boxW = lineLen + 8 + labelW + padX * 2;
    const boxH = methodNames.length * rowH + padY * 2 - 4;
    let x, y;
    if (legend === "br") { x = width - m.right - boxW - 6; y = m.top + ph - boxH - 6; }
    if (legend === "bl") { x = m.left + 6; y = m.top + ph - boxH - 6; }
    if (legend === "tr") { x = width - m.right - boxW - 6; y = m.top + 6; }
    if (legend === "tl") { x = m.left + 6; y = m.top + 6; }
    return (
      <g transform={`translate(${x},${y})`} fontFamily={font} fontSize="11">
        <rect width={boxW} height={boxH} fill="rgba(255,255,255,0.94)" stroke="#e4e4e4" strokeWidth="0.6" rx="3" />
        {methodNames.map((name, i) => {
          const col = resolveColor(name, i);
          return (
            <g key={name} transform={`translate(${padX},${padY + i * rowH})`}>
              {mode === "scatter" ? (
                <g>
                  <line x1="0" x2={lineLen} y1="6" y2="6" stroke={col.line} strokeWidth="1.8" strokeLinecap="round" />
                  <g transform={`translate(${lineLen / 2},6)`} stroke={col.line} strokeWidth="1.5" strokeLinecap="round">
                    <line x1="-3" y1="-3" x2="3" y2="3" />
                    <line x1="-3" y1="3" x2="3" y2="-3" />
                  </g>
                </g>
              ) : (
                <g>
                  <line x1="0" x2={lineLen} y1="6" y2="6" stroke={col.line} strokeWidth="2.2" strokeLinecap="round" />
                  {showDots && <circle cx={lineLen / 2} cy="6" r={dotRadius} fill={col.line} stroke="none" />}
                </g>
              )}
              <text x={lineLen + 6} y="9.5" fill="#1a1a1a" fontWeight="500">{name}</text>
            </g>
          );
        })}
      </g>
    );
  };

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} xmlns="http://www.w3.org/2000/svg"
         fontFamily={font} style={{ display: "block", background: "#ffffff", fontVariantNumeric: "tabular-nums" }}>
      <defs>
        <clipPath id="plot-area">
          <rect x={m.left} y={m.top} width={pw} height={ph} />
        </clipPath>
      </defs>

      {/* grid */}
      {showGrid && (
        <g>
          {yT.map((t, i) => (<line key={`yg${i}`} x1={m.left} x2={m.left + pw} y1={yScale(t)} y2={yScale(t)} stroke="#eeeeee" strokeWidth="0.7" />))}
          {xT.map((t, i) => (<line key={`xg${i}`} x1={xScale(t)} x2={xScale(t)} y1={m.top} y2={m.top + ph} stroke="#eeeeee" strokeWidth="0.7" />))}
        </g>
      )}

      {/* series */}
      <g clipPath="url(#plot-area)">
        {renderSeries()}
      </g>

      {/* axes */}
      <rect x={m.left} y={m.top} width={pw} height={ph} fill="none" stroke="#2a2a2a" strokeWidth="1" />

      {/* ticks */}
      <g fontSize="11" fill="#333">
        {xT.map((t, i) => (
          <g key={`xt${i}`}>
            <line x1={xScale(t)} x2={xScale(t)} y1={m.top + ph} y2={m.top + ph + 4} stroke="#2a2a2a" strokeWidth="1" />
            <text x={xScale(t)} y={m.top + ph + 17} textAnchor="middle">{xFmt(t)}</text>
          </g>
        ))}
        {yT.map((t, i) => (
          <g key={`yt${i}`}>
            <line x1={m.left - 4} x2={m.left} y1={yScale(t)} y2={yScale(t)} stroke="#2a2a2a" strokeWidth="1" />
            <text x={m.left - 7} y={yScale(t) + 3.5} textAnchor="end">{yFmt(t)}</text>
          </g>
        ))}
      </g>

      {/* axis labels */}
      <text x={m.left + pw / 2} y={height - 10} textAnchor="middle" fontSize="12.5" fontWeight="500" fill="#111">{xLabel}</text>
      <text transform={`translate(${14},${m.top + ph / 2}) rotate(-90)`} textAnchor="middle" fontSize="12.5" fontWeight="500" fill="#111">{yLabel}</text>

      {renderLegend()}
    </svg>
  );
}

window.ConfigurableChart = ConfigurableChart;
window.parseCSV = parseCSV;
