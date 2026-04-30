// Stimme: streaming literary Q&A with optional mic + auto-readback.

'use client'

import { useEffect, useRef, useState } from 'react'
import * as api from '@/lib/api'
import type { ParagraphWithTokens } from '@/lib/types'
import { AiSettingsPanel } from '@/components/AiSettingsPanel'
import { MicButton } from '@/components/MicButton'
import { useChat } from '@/hooks/useChat'
import { useStt } from '@/hooks/useStt'
import { useTts } from '@/hooks/useTts'
import { useSession } from '@/store/session'

const QUICK_CHIPS = [
  'Was bedeutet das auf Englisch?',
  'Erkl\u00E4re die Grammatik',
  'Historischer Kontext',
  'Schwierige Stellen',
]

export function Voice() {
  const currentParagraphId = useSession((s) => s.currentParagraphId)
  const setTab = useSession((s) => s.setTab)
  const readerPrefs = useSession((s) => s.readerPrefs)
  const setReaderPrefs = useSession((s) => s.setReaderPrefs)

  const [paragraph, setParagraph] = useState<ParagraphWithTokens | null>(null)
  const [draft, setDraft] = useState('')

  const chat = useChat(currentParagraphId)
  const stt = useStt()
  const tts = useTts(readerPrefs.ttsSpeed)

  useEffect(() => {
    tts.setSpeed(readerPrefs.ttsSpeed)
  }, [readerPrefs.ttsSpeed, tts])

  useEffect(() => {
    if (!currentParagraphId) return
    let active = true
    async function load() {
      try {
        const p = await api.getParagraph(currentParagraphId!)
        if (active) setParagraph(p)
      } catch {
        // ignore; placeholder handles the empty case
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [currentParagraphId])

  const lastSpokenRef = useRef<string | null>(null)
  useEffect(() => {
    if (chat.status !== 'done' || !readerPrefs.ttsAutoPlay) return
    if (chat.history.length === 0) return
    const last = chat.history[chat.history.length - 1]
    if (last.role !== 'assistant') return
    if (lastSpokenRef.current === last.content) return
    lastSpokenRef.current = last.content
    tts.play([{ id: `answer-${chat.history.length}`, text: last.content }])
  }, [chat.status, chat.history, readerPrefs.ttsAutoPlay, tts])

  const endRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [chat.history.length, chat.draft])

  const send = async (q: string) => {
    const question = q.trim()
    if (!question || chat.status === 'streaming') return
    setDraft('')
    await chat.ask(question)
  }

  const handleMic = async () => {
    if (stt.status === 'recording') {
      const transcript = await stt.stop()
      if (transcript) setDraft(transcript)
    } else {
      await stt.start()
    }
  }

  return (
    <div>
      <button className="back-btn" type="button" onClick={() => setTab('rd')}>
        {'\u2190'} Zur{'\u00FC'}ck zum Text
      </button>

      <AiSettingsPanel />

      <div className="passage-context">
        <div className="card-label">Aktuelle Passage</div>
        <div
          style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 14,
            color: 'var(--color-text-secondary)',
            lineHeight: 1.75,
          }}
        >
          {paragraph ? (
            <>
              {'\u201E'}
              {paragraph.text}
              {'\u201C'}
            </>
          ) : (
            <span className="skeleton">Keine Passage ausgew{'\u00E4'}hlt.</span>
          )}
        </div>
      </div>

      <div className="chat-history">
        {chat.history.length === 0 && chat.status === 'idle' ? (
          <div className="chat-ai">
            <div className="ai-avatar">KI</div>
            <div className="chat-bubble-ai">
              Stelle eine Frage zum Text {'\u2014'} auf Deutsch oder Englisch.
              Antworten beziehen sich auf die aktuelle Passage und ihre
              Nachbaraussagen.
            </div>
          </div>
        ) : null}
        {chat.history.map((turn, i) =>
          turn.role === 'user' ? (
            <div key={i} className="chat-user">
              <div className="chat-bubble-user">{turn.content}</div>
            </div>
          ) : (
            <div key={i} className="chat-ai">
              <div className="ai-avatar">KI</div>
              <div className="chat-bubble-ai">{turn.content}</div>
            </div>
          ),
        )}
        {chat.status === 'streaming' && chat.draft ? (
          <div className="chat-ai">
            <div className="ai-avatar">KI</div>
            <div className="chat-bubble-ai">
              {chat.draft}
              <span className="chat-caret">{'\u2588'}</span>
            </div>
          </div>
        ) : null}
        {chat.status === 'error' && chat.error ? (
          <div className="chat-ai">
            <div className="ai-avatar">{'!'}</div>
            <div className="chat-bubble-ai" style={{ color: '#a32d2d' }}>
              Fehler: {chat.error}
            </div>
          </div>
        ) : null}
        {(tts.status === 'playing' || tts.status === 'loading') ? (
          <div className="speaking-indicator">
            <div className="waveform" aria-hidden>
              {Array.from({ length: 5 }).map((_, i) => (
                <div
                  key={i}
                  className="waveform-bar"
                  style={{ height: `${30 + i * 12}%`, animationDelay: `${i * 80}ms` }}
                />
              ))}
            </div>
            <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
              {tts.backend === 'piper' ? 'Piper liest vor\u2026' : 'Browser liest vor\u2026'}
            </span>
          </div>
        ) : null}
        <div ref={endRef} />
      </div>

      <div className="voice-input-area">
        <div className="voice-input-row">
          <MicButton
            status={stt.status === 'error' ? 'idle' : stt.status}
            unavailable={stt.unavailable}
            unavailableHint={stt.installHint}
            onToggle={() => void handleMic()}
          />
          <div className="mic-status">
            {stt.status === 'recording' ? (
              <span style={{ color: '#a32d2d' }}>
                {'\u25CF'} Aufnahme l{'\u00E4'}uft {'\u2014'} zum Beenden klicken
              </span>
            ) : stt.status === 'processing' ? (
              'Transkription l\u00E4uft\u2026'
            ) : stt.unavailable ? (
              'STT offline \u2014 tippe deine Frage.'
            ) : (
              'Sprechen oder tippen Sie Ihre Frage \u2014 auf Deutsch oder Englisch'
            )}
          </div>
          <label
            className="chip"
            style={{ cursor: 'pointer', userSelect: 'none' }}
            title="Antwort automatisch vorlesen"
          >
            <input
              type="checkbox"
              checked={readerPrefs.ttsAutoPlay}
              onChange={(e) =>
                setReaderPrefs({ ttsAutoPlay: e.target.checked })
              }
              style={{ marginRight: 6 }}
            />
            Vorlesen
          </label>
        </div>
        <div className="text-input-row">
          <input
            type="text"
            placeholder="Fragen Sie etwas zur Passage..."
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send(draft)
              }
            }}
            disabled={chat.status === 'streaming'}
          />
          <button
            className="send-btn"
            type="button"
            disabled={!draft.trim() || chat.status === 'streaming'}
            onClick={() => void send(draft)}
          >
            {chat.status === 'streaming' ? '\u2026' : 'Senden'}
          </button>
        </div>
        <div className="chips-row">
          {QUICK_CHIPS.map((q) => (
            <button
              className="chip"
              type="button"
              key={q}
              disabled={chat.status === 'streaming'}
              onClick={() => void send(q)}
            >
              {q}
            </button>
          ))}
          {chat.history.length > 0 ? (
            <button
              className="chip"
              type="button"
              onClick={() => chat.clear()}
              style={{ marginLeft: 'auto' }}
            >
              Verlauf l{'\u00F6'}schen
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}
