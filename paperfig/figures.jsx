/* Paper figure builder — chart on the left, configuration panel on the right (in-page). */

const { useTweaks } = window;

/* ---------- Default CSV (50 rows, mild jitter) ---------- */
const DEFAULT_CSV = (() => {
  const cfg = [
    { name: "Muon",    asymp: 38.5, rate: 3.6 },
    { name: "Adam",    asymp: 32.0, rate: 2.9 },
    { name: "RMSprop", asymp: 22.5, rate: 2.4 },
    { name: "SGD",     asymp: 13.0, rate: 1.7 },
  ];
  const STEPS = 50;
  const xMax = 2000;
  const rngs = cfg.map((_, i) => {
    let a = 100 + i * 17;
    return () => { a = (a * 1103515245 + 12345) % 2147483648; return a / 2147483648; };
  });
  const walks = cfg.map(() => 0);
  const lines = ["step," + cfg.map(c => c.name).join(",")];
  for (let s = 0; s < STEPS; s++) {
    const x = Math.round((s * xMax) / (STEPS - 1));
    const t = s / (STEPS - 1);
    const vals = cfg.map((c, i) => {
      const base = 4.5 + c.asymp * (1 - Math.exp(-c.rate * t));
      walks[i] = walks[i] * 0.78 + (rngs[i]() - 0.5) * (2.6 * (1 - t * 0.5));
      return Math.max(0, base + walks[i]).toFixed(2);
    });
    lines.push(x + "," + vals.join(","));
  }
  return lines.join("\n");
})();

const SPARSE_SAMPLE =
  "step,Muon,Adam,RMSprop,SGD\n" +
  "0,4.6,4.4,4.5,4.5\n" +
  "200,20.1,15.3,11.0,7.4\n" +
  "400,28.9,22.4,15.8,10.0\n" +
  "600,33.4,27.1,18.7,11.6\n" +
  "800,35.9,29.4,20.4,12.4\n" +
  "1000,37.1,30.7,21.4,12.8\n" +
  "1200,37.8,31.4,21.9,13.0\n" +
  "1400,38.2,31.8,22.2,13.0\n" +
  "1600,38.4,31.9,22.4,13.0\n" +
  "1800,38.5,32.0,22.5,13.0\n" +
  "2000,38.5,32.0,22.5,13.0\n";

const FONT_OPTIONS = {
  inter:     "'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif",
  plex:      "'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif",
  plexserif: "'IBM Plex Serif', 'Source Serif Pro', Georgia, serif",
  serif:     "'Source Serif 4', 'Source Serif Pro', Georgia, serif",
};

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "mode": "smooth",
  "font": "inter",
  "width": 640,
  "height": 400,
  "xLabel": "Training steps",
  "yLabel": "AIME 2024 accuracy (%)",
  "xMin": 0,
  "xMax": 1000,
  "yMin": 0.4,
  "yMax": 0.8,
  "smoothing": 10,
  "showDots": false,
  "dotRadius": 3,
  "markerSize": 4,
  "rawOpacity": 55,
  "showGrid": true,
  "legend": "br",
  "crashed": {},
  "csv": "",
  "colors": {}
}/*EDITMODE-END*/;

/* ---------- Small UI primitives (in-page) ---------- */

function Section({ title, children, defaultOpen = true }) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div className="pf-sect">
      <button type="button" className="pf-sect-hd" onClick={() => setOpen(o => !o)}>
        <span>{title}</span>
        <span className="pf-sect-chev" data-open={open ? "1" : "0"}>›</span>
      </button>
      {open && <div className="pf-sect-body">{children}</div>}
    </div>
  );
}

function Row({ label, children, stack = false }) {
  return (
    <div className={stack ? "pf-row pf-row-stack" : "pf-row"}>
      <label className="pf-lbl">{label}</label>
      <div className="pf-ctrl">{children}</div>
    </div>
  );
}

function TextInput({ value, onChange, placeholder, mono = false, style }) {
  return (
    <input
      type="text"
      className={"pf-input" + (mono ? " pf-mono" : "")}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      spellCheck={false}
      style={style}
    />
  );
}

function NumberInput({ value, onChange, step = 1, min, max, style }) {
  return (
    <input
      type="number"
      className="pf-input"
      value={value}
      step={step} min={min} max={max}
      onChange={(e) => {
        const v = e.target.value === "" ? 0 : Number(e.target.value);
        onChange(Number.isFinite(v) ? v : 0);
      }}
      style={style}
    />
  );
}

function Slider({ value, onChange, min = 0, max = 100, step = 1, unit = "" }) {
  return (
    <div className="pf-sl">
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
      <span className="pf-sl-val">{value}{unit}</span>
    </div>
  );
}

function Toggle({ value, onChange }) {
  return (
    <button type="button" className="pf-toggle" data-on={value ? "1" : "0"}
      onClick={() => onChange(!value)}>
      <span></span>
    </button>
  );
}

function Select({ value, onChange, options }) {
  return (
    <select className="pf-input" value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => {
        const v = typeof o === "object" ? o.value : o;
        const l = typeof o === "object" ? o.label : o;
        return <option key={v} value={v}>{l}</option>;
      })}
    </select>
  );
}

function Segmented({ value, onChange, options }) {
  return (
    <div className="pf-seg">
      {options.map((o) => {
        const v = typeof o === "object" ? o.value : o;
        const l = typeof o === "object" ? o.label : o;
        return (
          <button key={v} type="button" className="pf-seg-btn" data-on={value === v ? "1" : "0"}
            onClick={() => onChange(v)}>{l}</button>
        );
      })}
    </div>
  );
}

/* hex swatch + input */
function ColorInput({ value, onChange, fallback }) {
  const swatch = window.normalizeHex(value) || fallback || "#cccccc";
  const valid = !value || !!window.normalizeHex(value);
  return (
    <div className="pf-color">
      <span className="pf-color-sw" style={{ background: swatch }}></span>
      <input
        type="text"
        className={"pf-input pf-mono" + (valid ? "" : " pf-bad")}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={fallback ? fallback.replace("#", "") : "RRGGBB"}
        spellCheck={false}
      />
    </div>
  );
}

/* ---------- App ---------- */
function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  const csv = t.csv && t.csv.length ? t.csv : DEFAULT_CSV;
  const parsed = window.parseCSV(csv);
  const methodNames = parsed.methodNames;

  const setColor = (name, hex) => {
    const next = Object.assign({}, t.colors || {});
    if (!hex || hex === "") delete next[name]; else next[name] = hex;
    setTweak("colors", next);
  };
  const clearColors = () => setTweak("colors", {});

  const setCrashed = (name, val) => {
    const next = Object.assign({}, t.crashed || {});
    if (val) next[name] = true; else delete next[name];
    setTweak("crashed", next);
  };

  // shared: clone SVG and embed Google Fonts as base64 data URIs
  const cloneWithFonts = async () => {
    const svgEl = document.querySelector('.pf-card svg');
    if (!svgEl) return null;
    const clone = svgEl.cloneNode(true);
    try {
      const fontUrl = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500;600&family=Source+Serif+4:wght@400;500;600&display=swap";
      const cssResp = await fetch(fontUrl);
      let css = await cssResp.text();
      const urlRe = /url\((https:\/\/[^)]+)\)/g;
      const urls = [...new Set([...css.matchAll(urlRe)].map(m => m[1]))];
      await Promise.all(urls.map(async (u) => {
        try {
          const r = await fetch(u);
          const blob = await r.blob();
          const dataUrl = await new Promise(res => {
            const rd = new FileReader();
            rd.onloadend = () => res(rd.result);
            rd.readAsDataURL(blob);
          });
          css = css.split(u).join(dataUrl);
        } catch (_) {}
      }));
      const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
      style.textContent = css;
      clone.insertBefore(style, clone.firstChild);
    } catch (_) {}
    return clone;
  };

  const handleDownloadSVG = async () => {
    const clone = await cloneWithFonts();
    if (!clone) return;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const svgStr = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "figure.svg"; a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPDF = () => {
    const svgEl = document.querySelector('.pf-card svg');
    if (!svgEl) return;
    const w = Number(svgEl.getAttribute('width'));
    const h = Number(svgEl.getAttribute('height'));
    const svgStr = new XMLSerializer().serializeToString(svgEl);
    const win = window.open('', '_blank');
    win.document.write(`<!DOCTYPE html><html><head>
      <link rel="preconnect" href="https://fonts.googleapis.com"/>
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500;600&family=Source+Serif+4:wght@400;500;600&display=swap" rel="stylesheet"/>
      <style>@page{size:${w}px ${h}px;margin:0}body{margin:0}</style>
      </head><body>${svgStr}</body></html>`);
    win.document.close();
    setTimeout(() => { win.print(); }, 500);
  };

  // default-color preview (when user hasn't overridden)
  const PALETTE_FALLBACKS = ["#55A868", "#3C88BE", "#C44D52", "#FF7F0F", "#9467BD", "#927860"];

  return (
    <div className="pf-app">
      <header className="pf-hdr">
        <div className="pf-hdr-title">Paper Figure Builder</div>
        <div className="pf-hdr-sub">RL training curves · per-line color · CSV-driven</div>
      </header>

      <main className="pf-main">
        <div className="pf-preview">
          <div className="pf-card">
            <window.ConfigurableChart
              width={t.width}
              height={t.height}
              csv={csv}
              mode={t.mode}
              xLabel={t.xLabel}
              yLabel={t.yLabel}
              xMin={t.xMin}
              xMax={t.xMax}
              yMin={t.yMin}
              yMax={t.yMax}
              smoothing={t.smoothing / 100}
              showDots={t.showDots}
              dotRadius={t.dotRadius}
              markerSize={t.markerSize}
              crashed={t.crashed || {}}
              rawOpacity={t.rawOpacity / 100}
              showGrid={t.showGrid}
              legend={t.legend}
              font={FONT_OPTIONS[t.font]}
              colors={t.colors || {}}
            />
          </div>
        </div>

        <aside className="pf-panel">
          <Section title="Format">
            <Row label="Mode">
              <Segmented value={t.mode} onChange={(v) => setTweak("mode", v)} options={[
                { value: "smooth",  label: "Smooth" },
                { value: "scatter", label: "Scatter+fit" },
              ]} />
            </Row>
            <Row label="Smoothing">
              <Slider value={t.smoothing} min={2} max={100} step={1} unit="%"
                onChange={(v) => setTweak("smoothing", v)} />
            </Row>
            {t.mode === "smooth" && (
              <Row label="Show dots">
                <Toggle value={t.showDots} onChange={(v) => setTweak("showDots", v)} />
              </Row>
            )}
            {t.mode === "smooth" && t.showDots && (
              <Row label="Dot radius">
                <Slider value={t.dotRadius} min={1.5} max={8} step={0.5} unit="px"
                  onChange={(v) => setTweak("dotRadius", v)} />
              </Row>
            )}
            <Row label="Marker size">
              <Slider value={t.markerSize} min={1} max={10} step={0.5} unit="px"
                onChange={(v) => setTweak("markerSize", v)} />
            </Row>
            <Row label="Raw opacity">
              <Slider value={t.rawOpacity} min={0} max={100} step={5} unit="%"
                onChange={(v) => setTweak("rawOpacity", v)} />
            </Row>
          </Section>

          <Section title="Size">
            <Row label="Width">
              <Slider value={t.width} min={360} max={1100} step={10} unit="px" onChange={(v) => setTweak("width", v)} />
            </Row>
            <Row label="Height">
              <Slider value={t.height} min={240} max={700} step={10} unit="px" onChange={(v) => setTweak("height", v)} />
            </Row>
          </Section>

          <Section title="Axis labels">
            <Row label="X label">
              <TextInput value={t.xLabel} onChange={(v) => setTweak("xLabel", v)} />
            </Row>
            <Row label="Y label">
              <TextInput value={t.yLabel} onChange={(v) => setTweak("yLabel", v)} />
            </Row>
          </Section>

          <Section title="Axis range">
            <Row label="X min / max">
              <div className="pf-pair">
                <NumberInput value={t.xMin} step={50} onChange={(v) => setTweak("xMin", v)} />
                <NumberInput value={t.xMax} step={50} onChange={(v) => setTweak("xMax", v)} />
              </div>
            </Row>
            <Row label="Y min / max">
              <div className="pf-pair">
                <NumberInput value={t.yMin} step={0.01} onChange={(v) => setTweak("yMin", v)} />
                <NumberInput value={t.yMax} step={0.01} onChange={(v) => setTweak("yMax", v)} />
              </div>
            </Row>
          </Section>

          <Section title="Line colors">
            {methodNames.length === 0 && (
              <div className="pf-hint">Add a CSV with a header row to see color inputs.</div>
            )}
            {methodNames.map((name, i) => {
              const fallback = PALETTE_FALLBACKS[i % PALETTE_FALLBACKS.length];
              return (
                <Row key={name} label={name}>
                  <ColorInput
                    value={(t.colors && t.colors[name]) || ""}
                    fallback={fallback}
                    onChange={(v) => setColor(name, v)}
                  />
                </Row>
              );
            })}
            {methodNames.length > 0 && (
              <button type="button" className="pf-btn pf-btn-mini" onClick={clearColors}>Reset to defaults</button>
            )}
          </Section>

          <Section title="Crash markers" defaultOpen={false}>
            {methodNames.length === 0 && (
              <div className="pf-hint">Add a CSV with a header row to see options.</div>
            )}
            {methodNames.map((name) => (
              <Row key={name} label={name}>
                <Toggle value={!!(t.crashed && t.crashed[name])} onChange={(v) => setCrashed(name, v)} />
              </Row>
            ))}
          </Section>

          <Section title="Display">
            <Row label="Gridlines">
              <Toggle value={t.showGrid} onChange={(v) => setTweak("showGrid", v)} />
            </Row>
            <Row label="Legend">
              <Select value={t.legend} onChange={(v) => setTweak("legend", v)} options={[
                { value: "br",   label: "Bottom right" },
                { value: "tr",   label: "Top right" },
                { value: "bl",   label: "Bottom left" },
                { value: "tl",   label: "Top left" },
                { value: "none", label: "Hidden" },
              ]} />
            </Row>
            <Row label="Font">
              <Select value={t.font} onChange={(v) => setTweak("font", v)} options={[
                { value: "inter",     label: "Inter (sans)" },
                { value: "plex",      label: "IBM Plex Sans" },
                { value: "plexserif", label: "IBM Plex Serif" },
                { value: "serif",     label: "Source Serif (LaTeX-feel)" },
              ]} />
            </Row>
          </Section>

          <Section title="Data (CSV)" defaultOpen={false}>
            <div className="pf-hint">First column = x · header row = method names</div>
            <textarea
              className="pf-input pf-mono pf-textarea"
              value={t.csv && t.csv.length ? t.csv : DEFAULT_CSV}
              onChange={(e) => setTweak("csv", e.target.value)}
              rows={14}
              spellCheck={false}
            />
            <div className="pf-pair" style={{ marginTop: 6 }}>
              <button type="button" className="pf-btn" onClick={() => setTweak("csv", DEFAULT_CSV)}>Dense sample</button>
              <button type="button" className="pf-btn" onClick={() => setTweak("csv", SPARSE_SAMPLE)}>Sparse sample</button>
            </div>
          </Section>

          <Section title="Export">
            <button type="button" className="pf-btn" onClick={handleDownloadSVG}
              style={{ fontWeight: 600 }}>
              Download SVG
            </button>
            <div className="pf-hint">矢量格式，字体内嵌</div>
            <button type="button" className="pf-btn" onClick={handleDownloadPDF}
              style={{ fontWeight: 600, marginTop: 8 }}>
              Save as PDF
            </button>
            <div className="pf-hint">矢量 PDF — 在弹出的打印对话框中选「存储为 PDF」</div>
          </Section>
        </aside>
      </main>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
