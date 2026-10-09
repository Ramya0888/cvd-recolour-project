import { useState } from 'react';
import axios from 'axios';
import './App.css';

const API = 'http://localhost:8000';
const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [cvdType, setCvdType] = useState('deuteranomaly');
  const [severity, setSeverity] = useState(100);
  const [k, setK] = useState(16);
  const [targetDeltaE, setTargetDeltaE] = useState(12);
  const [protectMemory, setProtectMemory] = useState(true);

  const onFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
  };

  const run = async () => {
    if (!file) { setError('Choose an image first.'); return; }
    setError(''); setLoading(true); setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const up = await axios.post(`${API}/upload`, form);
      const res = await axios.post(`${API}/process/${up.data.session_id}`, null, {
        params: { cvd_type: cvdType, severity, k, target_delta_e: targetDeltaE, protect_memory: protectMemory },
      });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  const m = result?.metrics;

  return (
    <div className="app">
      <header className="hero">
        <h1>Adaptive Colour-Blindness Recolouring</h1>
        <p>Pair-based remapping with memory-colour protection</p>
      </header>

      <div className="layout">
        {/* ---- control sidebar ---- */}
        <aside className="sidebar">
          <label className="file-drop">
            {preview
              ? <img src={preview} alt="preview" className="preview" />
              : <span>Click to choose an image</span>}
            <input type="file" accept="image/*" onChange={onFile} hidden />
          </label>

          <div className="field">
            <span>CVD type</span>
            <select value={cvdType} onChange={(e) => setCvdType(e.target.value)}>
              <option value="deuteranomaly">Deuteranopia (red-green)</option>
              <option value="protanomaly">Protanopia (red-green)</option>
              <option value="tritanomaly">Tritanopia (blue-yellow)</option>
            </select>
          </div>

          <Slider label="Severity" value={severity} min={0} max={100} set={setSeverity} />
          <Slider label="Clusters (K)" value={k} min={4} max={32} set={setK} />
          <Slider label="Target ΔE" value={targetDeltaE} min={5} max={25} set={setTargetDeltaE} />

          <label className="switch">
            <input type="checkbox" checked={protectMemory} onChange={(e) => setProtectMemory(e.target.checked)} />
            <span>Protect memory colours</span>
          </label>

          <button className="run" onClick={run} disabled={loading}>
            {loading ? <span className="spinner" /> : 'Process Image'}
          </button>
          {error && <p className="error">{error}</p>}
        </aside>

        {/* ---- results ---- */}
        <main className="results">
          {!result && !loading && <div className="empty">Upload an image and press Process to see the comparison.</div>}
          {loading && <div className="empty"><span className="spinner big" /><p>Running pipeline…</p></div>}

          {result && (
            <>
              {m && (
                <section className="kpis">
                  <Kpi big value={`${m.distinguishability.fix_rate_percent}%`} label="pairs fixed" tone="blue" />
                  <Kpi big value={m.naturalness.ssim} label="SSIM naturalness" tone={m.naturalness.ssim >= 0.9 ? 'green' : 'amber'} />
                  <Kpi value={`${m.distinguishability.avg_delta_e_before ?? '-'} → ${m.distinguishability.avg_delta_e_after ?? '-'}`} label="ΔE before → after" />
                  <Kpi value={`${m.naturalness.percent_pixels_changed}%`} label="pixels changed" />
                  <Kpi value={m.memory_protection.protected_clusters} label="protected clusters" />
                </section>
              )}

              {result.confusable_pairs?.length > 0 && (
                <section className="card">
                  <h2>Detected confusable colours</h2>
                  <div className="swatches">
                    {result.confusable_pairs.map((p, i) => (
                      <div className="swatch-pair" key={i}>
                        <span className="chip" style={{ background: rgb(p.colour_a) }} />
                        <span className="chip" style={{ background: rgb(p.colour_b) }} />
                        <small>ΔE {p.delta_e_simulated.toFixed(1)}</small>
                      </div>
                    ))}
                  </div>
                </section>
              )}

<section className="card">
                <h2>Four-panel comparison</h2>
                <div className="grid4">
                  <Panel title="Original" sub="normal vision" src={result.images.original} />
                  <Panel title="Original — CVD view" sub="what they see" src={result.images.sim_original} />
                  <Panel title="Corrected (ours)" sub="normal vision" src={result.images.remapped} />
                  <Panel title="Corrected — CVD view" sub="now distinguishable" src={result.images.sim_remapped} />
                </div>
              </section>

              <section className="card">
                <h2>Baseline vs ours</h2>
                <div className="grid2">
                  <Panel title="Standard Daltonization" sub="global shift" src={result.images.baseline} />
                  <Panel title="Ours (selective)" sub="pair-based + protection" src={result.images.remapped} />
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, set }) {
  return (
    <div className="field">
      <span>{label}<b>{value}</b></span>
      <input type="range" min={min} max={max} value={value} onChange={(e) => set(+e.target.value)} />
    </div>
  );
}

function Kpi({ value, label, tone, big }) {
  return (
    <div className={`kpi ${tone || ''} ${big ? 'big' : ''}`}>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
    </div>
  );
}

function Panel({ title, sub, src, download }) {
  return (
    <figure className="panel">
      <img src={src} alt={title} />
      <figcaption>
        <div>
          <strong>{title}</strong>
          <span>{sub}</span>
        </div>
        {download && <a href={src} download className="dl" title="Download">↓</a>}
      </figcaption>
    </figure>
  );
}

export default App;