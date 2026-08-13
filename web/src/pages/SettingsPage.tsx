import { useCallback, useEffect, useRef, useState } from "react";
import { api, Settings } from "../api";

const STATUS_TEXT: Record<Settings["gpu_status"], string> = {
  not_configured: "GPU engine not connected",
  deploying: "Deploying to your Modal account...",
  ready: "GPU engine ready",
  failed: "Deploy failed",
};

function GpuStatusBlock({ settings }: { settings: Settings }) {
  const { gpu_status: status } = settings;
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
      {status === "ready" && settings.gpu_endpoint_url && (
        <p className="muted">
          Endpoint:{" "}
          <a href={settings.gpu_endpoint_url} target="_blank" rel="noreferrer">
            {settings.gpu_endpoint_url}
          </a>
          <br />
          The <code>modal_gpu</code> engine is now available in the engine picker
          on the project page.
        </p>
      )}
      {status === "failed" && settings.gpu_error && (
        <p className="error">{settings.gpu_error}</p>
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
  const jobRef = useRef<string | null>(null);

  const stopPoll = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    jobRef.current = null;
  }, []);

  useEffect(() => stopPoll, [stopPoll]);

  const fail = useCallback((err: unknown) => {
    setError(err instanceof Error ? err.message : String(err));
  }, []);

  const startPoll = useCallback(() => {
    if (pollRef.current !== null) return;
    // деплой длится минуты: одиночный сетевой сбой не должен убивать опрос
    let misses = 0;
    pollRef.current = window.setInterval(async () => {
      try {
        if (jobRef.current !== null) {
          const job = await api.getJob(jobRef.current);
          if (job.status !== "done" && job.status !== "failed") return;
          jobRef.current = null;
        }
        const s = await api.getSettings();
        misses = 0;
        setSettings(s);
        setError(null);
        if (s.gpu_status !== "deploying") {
          stopPoll();
          setMessage(
            s.gpu_status === "ready"
              ? "GPU engine deployed and ready."
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
      if (s.gpu_status === "deploying") {
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

  const deploy = async () => {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const job = await api.deployGpu();
      jobRef.current = job.id;
      setSettings((s) => (s ? { ...s, gpu_status: "deploying", gpu_error: null } : s));
      startPoll();
    } catch (err) {
      fail(err);
    } finally {
      // «идёт деплой» показывает статус deploying, busy — только сам запрос
      setBusy(false);
    }
  };

  const wrongPrefix = tokenId.trim().length > 0 && !tokenId.trim().startsWith("ak-");
  const deploying = settings?.gpu_status === "deploying";

  return (
    <div className="settings-page">
      <h1>Settings</h1>

      <section className="settings-section">
        <h2>Labeling mode</h2>
        <p className="muted">
          By default the platform labels documents on CPU with the{" "}
          <code>rapidocr</code> engine — it works right away, no accounts and no
          keys. GPU mode is an optional speed-up: the platform deploys the GPU
          recipe <strong>into your own Modal account</strong>, you pay Modal
          directly at your own rate, and the platform only starts the deploy and
          calls the endpoint. Both modes return the same structure — per-line
          polygons with text and confidence.
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
        <h2>GPU engine</h2>
        {settings ? <GpuStatusBlock settings={settings} /> : <p className="muted">...</p>}
        <button
          onClick={() => void deploy()}
          disabled={busy || !settings?.modal_configured || deploying}
        >
          {settings?.gpu_status === "ready" ? "Redeploy" : "Connect GPU"}
        </button>
        {!settings?.modal_configured && (
          <span className="muted"> — save a Modal token first</span>
        )}
      </section>

      {message && <p className="muted">{message}</p>}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default SettingsPage;
