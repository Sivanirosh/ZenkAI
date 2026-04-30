'use client'

import { useCallback, useEffect, useState } from 'react'
import * as api from '@/lib/api'
import type { LlmOptions, OllamaStatus } from '@/lib/api'

type Status = 'loading' | 'ready' | 'error'

export function AiSettingsPanel() {
  const [open, setOpen] = useState(false)
  const [status, setStatus] = useState<Status>('loading')
  const [options, setOptions] = useState<LlmOptions | null>(null)
  const [ollama, setOllama] = useState<OllamaStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const refresh = useCallback(async () => {
    setStatus('loading')
    setError(null)
    try {
      const data = await api.getOllamaStatus()
      setOptions(data.options)
      setOllama(data.ollama)
      setStatus('ready')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStatus('error')
    }
  }, [])

  useEffect(() => {
    if (open) void refresh()
  }, [open, refresh])

  const patch = async (changes: Partial<LlmOptions>) => {
    if (!options) return
    const previous = options
    setOptions({ ...options, ...changes })
    setSaving(true)
    setError(null)
    try {
      const res = await api.updateLlmOptions(changes)
      setOptions(res.options)
    } catch (err) {
      setOptions(previous)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const models = ollama?.models ?? []
  const activeModel = options?.model ?? ''
  const modelInstalled =
    activeModel.length > 0 && models.includes(activeModel)

  return (
    <div className="ai-settings">
      <button
        type="button"
        className="ai-settings-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span aria-hidden>{'\u2699'}</span>
        <span>KI-Einstellungen</span>
        {status === 'ready' && ollama ? (
          <span
            className={
              'ai-status-dot ' +
              (ollama.reachable && modelInstalled
                ? 'ok'
                : ollama.reachable
                ? 'warn'
                : 'bad')
            }
            aria-hidden
          />
        ) : null}
      </button>

      {open ? (
        <div className="ai-settings-panel" role="dialog">
          {status === 'loading' ? (
            <div className="ai-settings-row">{'\u2026 l\u00E4dt'}</div>
          ) : status === 'error' ? (
            <div className="ai-settings-row" style={{ color: '#a32d2d' }}>
              Fehler: {error}
              <button
                type="button"
                className="chip"
                onClick={() => void refresh()}
              >
                Erneut versuchen
              </button>
            </div>
          ) : options && ollama ? (
            <>
              <div className="ai-settings-row">
                <span className="ai-settings-label">Ollama</span>
                <span className="ai-settings-value">
                  {ollama.reachable ? (
                    <>
                      erreichbar{' '}
                      <code style={{ opacity: 0.7 }}>{ollama.base_url}</code>
                    </>
                  ) : (
                    <span style={{ color: '#a32d2d' }}>
                      nicht erreichbar {ollama.error ? `(${ollama.error})` : ''}
                    </span>
                  )}
                </span>
              </div>

              <div className="ai-settings-row">
                <label
                  className="ai-settings-label"
                  htmlFor="ai-model-select"
                >
                  Modell
                </label>
                <select
                  id="ai-model-select"
                  value={activeModel}
                  disabled={saving || !ollama.reachable || models.length === 0}
                  onChange={(e) => void patch({ model: e.target.value })}
                >
                  {models.length === 0 ? (
                    <option value={activeModel}>{activeModel || '—'}</option>
                  ) : (
                    <>
                      {!modelInstalled && activeModel ? (
                        <option value={activeModel}>
                          {activeModel} (nicht installiert)
                        </option>
                      ) : null}
                      {models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </>
                  )}
                </select>
              </div>

              <div className="ai-settings-row">
                <label className="ai-settings-label" htmlFor="ai-think">
                  Thinking-Modus
                </label>
                <label className="ai-toggle">
                  <input
                    id="ai-think"
                    type="checkbox"
                    checked={options.think}
                    disabled={saving}
                    onChange={(e) => void patch({ think: e.target.checked })}
                  />
                  <span>{options.think ? 'an' : 'aus'}</span>
                </label>
              </div>

              <div className="ai-settings-row">
                <label
                  className="ai-settings-label"
                  htmlFor="ai-temperature"
                >
                  Temperatur
                </label>
                <div className="ai-slider">
                  <input
                    id="ai-temperature"
                    type="range"
                    min={0}
                    max={1.5}
                    step={0.05}
                    value={options.temperature}
                    disabled={saving}
                    onChange={(e) =>
                      void patch({ temperature: Number(e.target.value) })
                    }
                  />
                  <span className="ai-slider-value">
                    {options.temperature.toFixed(2)}
                  </span>
                </div>
              </div>

              {!modelInstalled && activeModel ? (
                <div className="ai-settings-hint">
                  {'\u26A0'} Modell {'\u00AB'}
                  {activeModel}
                  {'\u00BB'} ist nicht installiert. F{'\u00FC'}hre{' '}
                  <code>ollama pull {activeModel}</code> aus oder w
                  {'\u00E4'}hle ein anderes Modell.
                </div>
              ) : null}
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
