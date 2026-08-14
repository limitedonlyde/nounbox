import { useCallback, useEffect, useRef, useState } from "react";
import { api, GpuDeployment, GpuStatus, Settings, TASK_TITLES } from "../api";

const STATUS_TEXT: Record<GpuStatus, string> = {
  not_configured: "Not deployed",
  deploying: "Deploying to your Modal account...",
  ready: "Ready",
  failed: "Deploy failed",
};

function GpuStatusBlock({ deployment }: { deployment: GpuDeployment }) {
  const { status } = deployment;
  return (
    <div className={`gpu-status gpu-status-${status}`}>
      <div className="gpu-status-title">
        {status === "deploying" && <span className="spinner" />}
        {STATUS_TEXT[status]}
      </div>
      {status === "deploying" && (
        <p className="muted">
          The first deploy builds an image with the models — this can take a few
          minutes. You can close the page, the process keeps running on the
          server.
        </p>
      )}
      {status === "ready" && deployment.endpoint_url && (
        <p className="muted">
          Endpoint:{" "}
          <a href={deployment.endpoint_url} target="_blank" rel="noreferrer">
            {deployment.endpoint_url}
          </a>
          <br />
          The <code>{deployment.engine}</code> engine is now available in the
          engine picker of a {TASK_TITLES[deployment.task].toLowerCase()}{" "}
          project.
        </p>
      )}
      {status === "failed" && deployment.error && (
        <p className="error">{deployment.error}</p>
      )}
    </div>
  );
}

function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [tokenId, setTokenId] = useState("");
  const [tokenSecret, setTokenSecret] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pollRef = useRef<number | null>(null);
  // deploy jobs still in flight, by engine — each card starts its own
  const jobsRef = useRef<Map<string, string>>(new Map());

  const stopPoll = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    jobsRef.current.clear();
  }, []);

  useEffect(() => stopPoll, [stopPoll]);

  const fail = useCallback((err: unknown) => {
    setError(err instanceof Error ? err.message : String(err));
  }, []);

  const startPoll = useCallback(() => {
    if (pollRef.current !== null) return;
    // a deploy takes minutes: a single network blip must not kill the poll
    let misses = 0;
    pollRef.current = window.setInterval(async () => {
      try {
        for (const [engine, jobId] of [...jobsRef.current]) {
          const job = await api.getJob(jobId);
          if (job.status === "done" || job.status === "failed") {
            jobsRef.current.delete(engine);
          }
        }
        const s = await api.getSettings();
        misses = 0;
        setSettings(s);
        setError(null);
        const deploying = s.gpus.some((g) => g.status === "deploying");
        if (!deploying && jobsRef.current.size === 0) {
          stopPoll();
          const ready = s.gpus.filter((g) => g.status === "ready");
          setMessage(
            ready.length > 0
              ? `GPU engines ready: ${ready.map((g) => g.engine).join(", ")}.`
              : null
          );
        }
      } catch (err) {
        misses += 1;
        if (misses < 3) return;
        stopPoll();
        fail(err);
      }
    }, 3000);
  }, [fail, stopPoll]);

  const load = useCallback(async () => {
    try {
      const s = await api.getSettings();
      setSettings(s);
      setError(null);
      if (s.gpus.some((g) => g.status === "deploying")) {
        startPoll();
      }
    } catch (err) {
      fail(err);
    }
  }, [fail, startPoll]);

  useEffect(() => {
    void load();
  }, [load]);

  const saveToken = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tokenId.trim() || !tokenSecret.trim()) return;
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      setSettings(await api.saveModalToken(tokenId.trim(), tokenSecret.trim()));
      setTokenId("");
      setTokenSecret("");
      setMessage("Token saved.");
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    setBusy(true);
    setMessage(null);
    setError(null);
    stopPoll();
    try {
      setSettings(await api.deleteModalToken());
      setMessage("Token removed, GPU mode is off.");
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const deploy = async (engine: string) => {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const job = await api.deployGpu(engine);
      jobsRef.current.set(engine, job.id);
      setSettings((s) =>
        s
          ? {
              ...s,
              gpus: s.gpus.map((g) =>
                g.engine === engine
                  ? { ...g, status: "deploying", error: null }
                  : g
              ),
            }
          : s
      );
      startPoll();
    } catch (err) {
      fail(err);
    } finally {
      // "deploy in progress" is the deploying status; busy covers only the request
      setBusy(false);
    }
  };

  const wrongPrefix = tokenId.trim().length > 0 && !tokenId.trim().startsWith("ak-");
  const gpus = settings?.gpus ?? [];

  return (
    <div className="settings-page">
      <h1>Settings</h1>

      <section className="settings-section">
        <h2>Labeling mode</h2>
        <p className="muted">
          By default the platform labels on CPU — <code>owlv2</code> draws boxes
          for the classes of a detection project, <code>rapidocr</code> reads
          text in an OCR one. Both work right away, with no accounts and no
          keys. GPU mode is an optional speed-up: the platform deploys the
          matching recipe <strong>into your own Modal account</strong>, you pay
          Modal directly at your own rate, and the platform only starts the
          deploy and calls the endpoint. A GPU engine returns the same boxes as
          its CPU twin, just faster — the two halves of a project stay one
          dataset.
        </p>
      </section>

      {settings && !settings.access_protected && (
        <section className="settings-section">
          <p className="warning">
            Anyone who can reach the API port can change these settings —
            including putting in their own Modal token or starting a deploy at
            your expense. If anyone else can reach the platform, set{" "}
            <code>APP_ACCESS_TOKEN</code> in <code>.env</code> and restart the
            stack.
          </p>
        </section>
      )}

      <section className="settings-section">
        <h2>Modal token</h2>
        <p className="muted">
          You need an API token: install the Modal CLI and run{" "}
          <code>modal token new</code> — it prints a pair of{" "}
          <code>token_id</code> (<code>ak-...</code>) and <code>token_secret</code>{" "}
          (<code>as-...</code>). Dashboard tokens that look like <code>wk-</code>/
          <code>ws-</code> are proxy tokens for calls, they do not work for a
          deploy. The secret is stored encrypted and never comes back to the
          interface.
        </p>

        {settings?.modal_configured && (
          <p>
            Token <code>{settings.modal_token_id_masked}</code> is connected{" "}
            <button onClick={() => void disconnect()} disabled={busy}>
              Disconnect
            </button>
          </p>
        )}

        <form className="settings-form" onSubmit={saveToken}>
          <label>
            token id
            <input
              value={tokenId}
              onChange={(e) => setTokenId(e.target.value)}
              placeholder="ak-..."
              autoComplete="off"
            />
          </label>
          <label>
            token secret
            <input
              type="password"
              value={tokenSecret}
              onChange={(e) => setTokenSecret(e.target.value)}
              placeholder="as-..."
              autoComplete="new-password"
            />
          </label>
          <button
            type="submit"
            disabled={busy || !tokenId.trim() || !tokenSecret.trim()}
          >
            {settings?.modal_configured ? "Replace token" : "Save"}
          </button>
        </form>

        {wrongPrefix && (
          <p className="warning">
            token id usually starts with <code>ak-</code>. If yours starts with{" "}
            <code>wk-</code>, it is a proxy token — a deploy needs the output of{" "}
            <code>modal token new</code>.
          </p>
        )}
      </section>

      <section className="settings-section">
        <h2>GPU engines</h2>
        <p className="muted">
          One app per task: they share no dependency, so each is deployed
          separately and costs nothing while it is not deployed. Deploying one
          leaves the other exactly as it is.
        </p>
        {settings ? (
          gpus.map((deployment) => (
            <div className="gpu-card" key={deployment.engine}>
              <h3>{deployment.title}</h3>
              <GpuStatusBlock deployment={deployment} />
              <button
                className="primary"
                onClick={() => void deploy(deployment.engine)}
                disabled={
                  busy ||
                  !settings.modal_configured ||
                  deployment.status === "deploying"
                }
              >
                {deployment.status === "ready" ? "Redeploy" : "Connect GPU"}
              </button>
              {!settings.modal_configured && (
                <span className="muted"> — save a Modal token first</span>
              )}
            </div>
          ))
        ) : (
          <p className="muted">...</p>
        )}
      </section>

      {message && <p className="muted">{message}</p>}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default SettingsPage;
