import { useRef, useState } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Database01Icon,
  Upload04Icon,
  ShieldCheckIcon,
  ChartLineData02Icon,
  Cancel01Icon,
  File01Icon,
  ArrowRight01Icon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";

type ModelMap = Record<string, string[]>;
type AnyRecord = Record<string, any>;

const modelOptions: ModelMap = {
  classification: [
    "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost",
    "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Naive Bayes", "AdaBoost", "Extra Trees",
  ],
  regression: [
    "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net", "Decision Tree Regressor",
    "Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "Support Vector Regression (SVR)",
    "K-Nearest Neighbors Regressor",
  ],
  clustering: ["K-Means", "Hierarchical Clustering", "DBSCAN", "Gaussian Mixture Model", "Mean Shift", "Spectral Clustering"],
  "time-series": ["ARIMA", "SARIMA", "Exponential Smoothing", "Prophet", "Random Forest (Forecasting)", "XGBoost (Forecasting)", "LSTM"],
  anomaly: ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "DBSCAN", "Autoencoder"],
  "dimensionality-reduction": ["PCA", "Kernel PCA", "Truncated SVD", "t-SNE", "UMAP", "LDA"],
  recommendation: ["Collaborative Filtering", "Content-Based Filtering", "Matrix Factorization", "K-Nearest Neighbors", "Hybrid Recommendation"],
  ranking: ["Learning to Rank", "RankNet", "LambdaMART", "XGBoost Ranking"],
};

const allModels = Array.from(new Set(Object.values(modelOptions).flat())).sort();

function humanize(value: string | undefined | null) {
  if (!value) return "";
  return value.replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function severityClass(severity: string) {
  return `finding finding-${severity || "info"}`;
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [mlTask, setMlTask] = useState("");
  const [modelType, setModelType] = useState("");
  const [intendedUse, setIntendedUse] = useState("");
  const [description, setDescription] = useState("");
  const [auditResult, setAuditResult] = useState<AnyRecord | null>(null);
  const [isAuditing, setIsAuditing] = useState(false);
  const [auditError, setAuditError] = useState("");

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      alert("Please select a CSV file.");
      event.target.value = "";
      return;
    }
    setSelectedFile(file);
    setAuditResult(null);
    setAuditError("");
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setTargetColumn("");
    setMlTask("");
    setModelType("");
    setIntendedUse("");
    setDescription("");
    setAuditResult(null);
    setAuditError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleTaskChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setMlTask(event.target.value);
    setModelType("");
  };

  const handleAudit = async () => {
    if (!selectedFile) return alert("Please select a CSV file first.");

    setIsAuditing(true);
    setAuditError("");
    setAuditResult(null);

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 120000);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("target", targetColumn.trim());
      formData.append("task", mlTask);
      formData.append("model", modelType);
      formData.append("intended_use", intendedUse.trim());
      formData.append("description", description.trim());

      // Use the deployed FastAPI backend URL when VITE_API_URL is configured.
      // If it is not configured (for local development), fall back to /api/audit.
      const apiBaseUrl = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");
      const response = await fetch(`${apiBaseUrl}/api/audit`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      const raw = await response.text();
      let result: AnyRecord = {};

      try {
        result = raw ? JSON.parse(raw) : {};
      } catch {
        throw new Error(raw || "Invalid backend response.");
      }

      if (!response.ok) {
        throw new Error(result.detail || "Dataset audit failed.");
      }

      setAuditResult(result);
    } catch (error) {
      setAuditError(
        error instanceof DOMException && error.name === "AbortError"
          ? "The audit exceeded 120 seconds. Check the backend terminal."
          : error instanceof Error
            ? error.message
            : "Something went wrong while auditing the dataset."
      );
    } finally {
      window.clearTimeout(timeout);
      setIsAuditing(false);
    }
  };

  const handleBackToAudit = () => {
    setAuditResult(null);
    setAuditError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const visibleModels = mlTask ? (modelOptions[mlTask] || []) : allModels;
  const r = auditResult;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><HugeiconsIcon icon={Database01Icon} size={23} /></div>
          <div>
            <div className="brand-name">Data Audit</div>
            <div className="brand-subtitle">Dataset assessment system</div>
          </div>
        </div>
        <div className="topbar-status"><span className="status-dot" />Local analysis</div>
      </header>

      <main className="main-content">
        {!r ? (
          <>
        <section className="intro">
          <p className="eyebrow">DATASET AUDIT</p>
          <h1>Understand your dataset<br />before building your model.</h1>
          <p className="intro-text">
            Upload a CSV to inspect quality, leakage risk, task and model suitability, and prediction context.
          </p>
        </section>

        <section className="dataset-card">
          {!selectedFile ? (
            <div className="empty-upload">
              <div className="upload-icon"><HugeiconsIcon icon={Upload04Icon} size={30} /></div>
              <div className="upload-copy">
                <h2>Upload your dataset</h2>
                <p>CSV is the only required input. Everything else is optional.</p>
                <input ref={fileInputRef} type="file" accept=".csv,text/csv" onChange={handleFileChange} hidden />
                <button className="upload-button" onClick={() => fileInputRef.current?.click()}>
                  <HugeiconsIcon icon={Upload04Icon} size={18} /> Choose CSV file
                </button>
                <span className="file-note">CSV files only</span>
              </div>
            </div>
          ) : (
            <div className="selected-dataset">
              <div className="selected-left">
                <div className="selected-file-icon"><HugeiconsIcon icon={File01Icon} size={30} /></div>
                <div>
                  <p className="small-label">DATASET SELECTED</p>
                  <h2>Your CSV is ready</h2>
                  <p>You can provide additional information below, but none of it is required.</p>
                </div>
              </div>
              <div className="file-info">
                <HugeiconsIcon icon={File01Icon} size={25} />
                <div className="file-details">
                  <strong>{selectedFile.name}</strong>
                  <span>{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB · CSV</span>
                </div>
                <button className="remove-file" onClick={handleRemoveFile} aria-label="Remove file">
                  <HugeiconsIcon icon={Cancel01Icon} size={18} />
                </button>
              </div>
            </div>
          )}
        </section>

        {selectedFile && (
          <section className="audit-options">
            <div className="section-heading">
              <div><p className="section-label">OPTIONAL INFORMATION</p><h2>Add context to your audit</h2></div>
              <span className="optional-badge">All fields optional</span>
            </div>

            <div className="form-grid">
              <div className="form-field">
                <label htmlFor="target"><span className="label-main">Target column</span><span className="optional-text">Optional</span></label>
                <input id="target" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} placeholder="e.g. Churn" />
                <p>Used for target analysis and target-dependent leakage checks.</p>
              </div>

              <div className="form-field">
                <label htmlFor="task"><span className="label-main">Machine-learning task</span><span className="optional-text">Optional</span></label>
                <select id="task" value={mlTask} onChange={handleTaskChange}>
                  <option value="">Select a task</option>
                  {Object.keys(modelOptions).map((t) => <option key={t} value={t}>{humanize(t)}</option>)}
                </select>
                <p>Used for task-specific dataset suitability analysis.</p>
              </div>

              <div className="form-field">
                <label htmlFor="model"><span className="label-main">Model / algorithm</span><span className="optional-text">Optional</span></label>
                <select id="model" value={modelType} onChange={(e) => setModelType(e.target.value)}>
                  <option value="">{mlTask ? "Select a model" : "Select a model (task optional)"}</option>
                  {visibleModels.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
                <p>A model can be assessed even when no target is supplied.</p>
              </div>

              <div className="form-field">
                <label htmlFor="intended-use"><span className="label-main">Intended use</span><span className="optional-text">Optional</span></label>
                <input id="intended-use" value={intendedUse} onChange={(e) => setIntendedUse(e.target.value)} placeholder="e.g. Predict customer churn next month" />
                <p>Used to check whether the stated goal aligns with the available data context.</p>
              </div>

              <div className="form-field form-field-full">
                <label htmlFor="description"><span className="label-main">Dataset description</span><span className="optional-text">Optional</span></label>
                <textarea id="description" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Briefly describe what this dataset represents or what you want to predict..." rows={4} />
                <p>The description is used as context for target, task and intended-use interpretation.</p>
              </div>
            </div>

            <div className="audit-action">
              <div className="ready-state">
                <div className="ready-icon"><HugeiconsIcon icon={InformationCircleIcon} size={19} /></div>
                <div><strong>Ready to audit</strong><span>The CSV is the only required input.</span></div>
              </div>
              <button className="audit-button" onClick={handleAudit} disabled={isAuditing}>
                {isAuditing ? "Analyzing dataset..." : "Start Dataset Audit"}
                {!isAuditing && <HugeiconsIcon icon={ArrowRight01Icon} size={19} />}
              </button>
            </div>
          </section>
        )}

        {auditError && (
          <section className="audit-error">
            <strong>Audit could not be completed</strong>
            <p>{auditError}</p>
          </section>
        )}
        </>
        ) : (
          <>
            <div style={{ marginBottom: "18px" }}>
              <button
                type="button"
                onClick={handleBackToAudit}
                style={{
                  border: "1px solid #cfd5d5",
                  background: "#ffffff",
                  color: "#1d6c68",
                  padding: "10px 14px",
                  fontWeight: 700,
                  fontSize: "13px",
                  cursor: "pointer",
                }}
              >
                ← Back to audit
              </button>
            </div>

          <section className="results-section">
            <div className="results-header">
              <div>
                <p className="section-label">AUDIT RESULTS</p>
                <h2>Dataset assessment</h2>
                <p>Analysis completed for <strong>{r.dataset.filename}</strong></p>
              </div>
              <div className="quality-score">
                <span>QUALITY SCORE</span>
                <strong>{r.quality.score}</strong>
                <small>/ 100</small>
              </div>
            </div>

            <div className="result-cards">
              <div className="result-card"><span>ROWS</span><strong>{r.dataset.rows.toLocaleString()}</strong></div>
              <div className="result-card"><span>COLUMNS</span><strong>{r.dataset.columns}</strong></div>
              <div className="result-card"><span>DUPLICATE ROWS</span><strong>{r.quality.duplicate_rows.toLocaleString()}</strong><small>{r.quality.duplicate_percentage}%</small></div>
              <div className="result-card"><span>MISSING COLUMNS</span><strong>{r.quality.columns_with_missing_values.length}</strong></div>
              <div className="result-card"><span>LEAKAGE SCORE</span><strong>{r.leakage.score}</strong><small>/ 100</small></div>
            </div>

            <div className="result-panel">
              <div className="result-panel-header">
                <div><h3>What was checked</h3><p>The audit scope changes according to the information supplied.</p></div>
                <span>{r.audit_scope.length} checks</span>
              </div>
              <div className="scope-grid">
                {r.audit_scope.map((item: string) => (
                  <div className="scope-item" key={item}>
                    <span className="scope-marker" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="results-grid">
              <div className="result-panel">
                <div className="result-panel-header">
                  <div><h3>Quality score</h3><p>Transparent, reproducible score based only on measured dataset properties.</p></div>
                  <span>{r.quality.score}/100</span>
                </div>
                <p className="analysis-summary">
                  <strong>This is a deterministic, rule-based audit index:</strong> it is calculated from measured properties of the uploaded data. It is not a model accuracy score, a universal data-quality certification, or a guarantee that every data issue has been detected.
                </p>
                <div className="data-table-wrap">
                  <table className="data-table scoring-table">
                    <thead><tr><th>Parameter checked</th><th>Weight</th><th>Observed metric</th><th>Deduction</th><th>Points</th></tr></thead>
                    <tbody>
                      {r.quality.parameters.map((p: AnyRecord) => (
                        <tr key={p.parameter}>
                          <td>{p.parameter}</td><td>{p.weight}</td><td>{typeof p.metric === "object" ? JSON.stringify(p.metric) : p.metric}</td><td>{p.deduction}</td><td>{p.points_awarded}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="result-panel leakage-panel">
                <div className="result-panel-header">
                  <div><h3>Leakage score</h3><p>One score only: higher means greater detected leakage risk.</p></div>
                  <span className="status-badge">{r.leakage.score}/100</span>
                </div>
                <div className="large-score">{r.leakage.score}<small>/100</small></div>
                <p className="analysis-summary">{r.leakage.scale}. {r.leakage.summary}</p>
                <div className="check-list-column">
                  {r.leakage.checks.map((x: string) => (
                    <div className="check-row" key={x}>{x}</div>
                  ))}
                </div>
                <div className="finding-stack">
                  <h4>Detected leakage findings</h4>
                  {r.leakage.findings.length > 0 ? (
                    r.leakage.findings.map((f: AnyRecord, i: number) => (
                      <div className={severityClass(f.severity)} key={`${f.type || "leakage"}-${f.column || "dataset"}-${i}`}>
                        <strong>{humanize(f.severity || "info")}</strong>
                        {f.column && <span> · Column: <strong>{f.column}</strong></span>}
                        {f.metric && <span> · Evidence: {f.metric}</span>}
                        <div>{f.message}</div>
                      </div>
                    ))
                  ) : (
                    <p className="analysis-summary">No findings matched the current screening rules. This is not proof that the dataset is leakage-free.</p>
                  )}
                </div>
                <p className="analysis-summary"><strong>Screening status:</strong> {r.leakage.status}. {r.leakage.scope || ""}</p>
              </div>
            </div>

            <div className="result-panel">
              <div className="result-panel-header">
                <div><h3>Column-level audit</h3><p>Every column is shown with the measured values used by the quality audit.</p></div>
                <span>{r.column_details.length} columns</span>
              </div>
              <div className="data-table-wrap column-table-wrap">
                <table className="data-table column-table">
                  <thead>
                    <tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Repeated values</th><th>Outliers</th><th>Flags</th><th>Sample values</th></tr>
                  </thead>
                  <tbody>
                    {r.column_details.map((c: AnyRecord) => (
                      <tr key={c.column}>
                        <td><strong>{c.column}</strong></td>
                        <td>{c.dtype}</td>
                        <td>{c.missing_count.toLocaleString()} ({c.missing_percentage}%)</td>
                        <td>{c.unique_count.toLocaleString()} ({c.unique_percentage}%)</td>
                        <td>{c.repeated_value_count.toLocaleString()} ({c.repeated_value_percentage}%)</td>
                        <td>{c.outlier_count.toLocaleString()} ({c.outlier_percentage}%)</td>
                        <td>{[c.is_constant && "constant", c.is_id_like && "ID-like", c.is_high_cardinality && "high-cardinality", c.is_date_like && "date-like"].filter(Boolean).join(", ") || "—"}</td>
                        <td>{c.sample_values.join(" | ") || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {r.target.provided && (
              <div className="result-panel">
                <div className="result-panel-header">
                  <div><h3>Target analysis</h3><p>Target-specific checks are shown because a target was supplied.</p></div>
                  <span className="status-badge">{r.target.exists ? "Target found" : "Target not found"}</span>
                </div>
                {r.target.exists ? (
                  <>
                    <p className="analysis-summary">
                      The target <strong>{r.target.analysis.column}</strong> was found. It contains <strong>{r.target.analysis.unique_values}</strong> unique values and <strong>{r.target.analysis.missing_values.toLocaleString()}</strong> missing values ({r.target.analysis.missing_percentage}%). The observed target characteristics suggest <strong>{humanize(r.target.analysis.suggested_task)}</strong> as a possible task.
                    </p>
                    <div className="detail-grid">
                      <div><span>Data type</span><strong>{r.target.analysis.type}</strong></div>
                      <div><span>Inferred type</span><strong>{humanize(r.target.analysis.inferred_type)}</strong></div>
                      <div><span>Suggested task</span><strong>{humanize(r.target.analysis.suggested_task)}</strong></div>
                      <div><span>Unique values</span><strong>{r.target.analysis.unique_values}</strong></div>
                      {r.target.analysis.class_count !== undefined && <div><span>Classes</span><strong>{r.target.analysis.class_count}</strong></div>}
                      {r.target.analysis.imbalance_ratio !== undefined && <div><span>Imbalance ratio</span><strong>{r.target.analysis.imbalance_ratio}</strong></div>}
                    </div>
                  </>
                ) : (
                  <p className="analysis-summary">{r.target.analysis.message}</p>
                )}
              </div>
            )}

            {r.task.provided && (
              <div className="result-panel">
                <div className="result-panel-header">
                  <div><h3>ML task suitability</h3><p>Dataset fit for the selected or inferred machine-learning task.</p></div>
                  <span className="status-badge">{r.task_analysis.score}/100</span>
                </div>
                <p className="analysis-summary">{r.task_analysis.summary}</p>
                {r.task_analysis.findings.map((f: AnyRecord, i: number) => <div className={severityClass(f.severity)} key={`${f.type || "task"}-${i}`}>{f.message}</div>)}
                {r.task_analysis.recommendations.map((x: string) => <p className="recommendation-text" key={x}>{x}</p>)}
              </div>
            )}

            {r.model.provided && (
              <div className="result-panel">
                <div className="result-panel-header">
                  <div><h3>Model suitability</h3><p>Rated against the dataset, selected task and target when available.</p></div>
                  <span className="status-badge">{r.model_analysis.score}/100</span>
                </div>
                <p className="analysis-summary">
                  <strong>{r.model.value}</strong>: {r.model_analysis.summary} The current evidence gives this model a <strong>{r.model_analysis.score}/100</strong> fit score.
                </p>
                <div className="data-table-wrap">
                  <table className="data-table model-table">
                    <thead><tr><th>Model parameter</th><th>Weight</th><th>Observed metric</th><th>Points</th><th>Status</th></tr></thead>
                    <tbody>
                      {r.model_analysis.parameters.map((p: AnyRecord) => (
                        <tr key={p.parameter}><td>{p.parameter}</td><td>{p.weight}</td><td>{p.metric}</td><td>{p.points_awarded}</td><td>{p.status}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {r.model_analysis.findings.map((f: AnyRecord, i: number) => <div className={severityClass(f.severity)} key={`${f.type || "model"}-${i}`}>{f.message}</div>)}
                {r.model_analysis.recommendations.map((x: string) => <p className="recommendation-text" key={x}>{x}</p>)}
              </div>
            )}

            {r.context_analysis.provided && (
              <div className="result-panel">
                <div className="result-panel-header">
                  <div><h3>Intended use and description</h3><p>Both fields were actively used as context rather than simply stored.</p></div>
                  <span className="status-badge">{r.context_analysis.score}/100 context</span>
                </div>
                {r.intended_use && <p className="analysis-summary"><strong>Intended use:</strong> {r.intended_use}</p>}
                {r.description && <p className="analysis-summary"><strong>Dataset description:</strong> {r.description}</p>}
                <p className="analysis-summary">{r.context_analysis.summary}</p>
                {r.context_analysis.findings.map((f: AnyRecord, i: number) => <div className={severityClass(f.severity)} key={`${f.type || "context"}-${i}`}>{f.message}</div>)}
                {r.context_analysis.recommendations.map((x: string) => <p className="recommendation-text" key={x}>{x}</p>)}
              </div>
            )}

            {r.suggested_models.length > 0 && (
              <div className="result-panel">
                <div className="result-panel-header">
                  <div><h3>Suggested models</h3><p>Suggestions are based on the selected task, target characteristics and/or available feature structure.</p></div>
                  <span className="status-badge">Recommendations</span>
                </div>
                <div className="data-table-wrap suggestion-table-wrap">
                  <table className="data-table suggestion-table">
                    <thead>
                      <tr>
                        <th>Task</th>
                        <th>Model</th>
                        <th>Task fit</th>
                        <th>Model fit</th>
                        <th>Combined fit</th>
                        <th>Why it was suggested</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r.suggested_models.map((s: AnyRecord) => (
                        <tr key={`${s.task || "task"}-${s.model}`}>
                          <td>
                            <strong>{humanize(s.task)}</strong>
                            {s.is_best && (
                              <div style={{ marginTop: "4px", fontSize: "11px", fontWeight: 600 }}>
                                Best-supported combination
                              </div>
                            )}
                          </td>
                          <td><strong>{s.model}</strong></td>
                          <td>{s.task_score ?? "—"}/100</td>
                          <td>{s.model_score ?? "—"}/100</td>
                          <td><strong>{s.score}/100</strong></td>
                          <td>{s.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <div className="result-panel final-panel">
              <div className="result-panel-header">
                <div><h3>Final assessment</h3><p>Combined result of the checks that were actually available.</p></div>
                <span className="status-badge">{r.final_assessment.status}</span>
              </div>
              <p className="analysis-summary">{r.final_assessment.summary}</p>
              {r.final_assessment.findings.slice(0, 15).map((f: AnyRecord, i: number) => <div className={severityClass(f.severity)} key={`${f.area || "finding"}-${i}`}>{f.message}</div>)}
              {r.final_assessment.recommendations.length > 0 && (
                <div className="recommendation-list">
                  <h4>Recommended next steps</h4>
                  {r.final_assessment.recommendations.map((x: string) => <p key={x}>{x}</p>)}
                </div>
              )}
            </div>
          </section>
          </>
        )}

        {!r && (
          <section className="feature-row">
          <div className="feature-item"><div className="feature-icon"><HugeiconsIcon icon={ChartLineData02Icon} size={24} /></div><div><h3>Dataset quality</h3><p>Transparent score with every scoring parameter shown.</p></div></div>
          <div className="feature-item"><div className="feature-icon"><HugeiconsIcon icon={ShieldCheckIcon} size={24} /></div><div><h3>Leakage score</h3><p>Single 0–100 detected-risk score with target-specific checks when available.</p></div></div>
          <div className="feature-item"><div className="feature-icon"><HugeiconsIcon icon={Database01Icon} size={24} /></div><div><h3>ML readiness</h3><p>Task, model, target, intended-use and dataset-context analysis.</p></div></div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;


