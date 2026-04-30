// App shell: header + tab router matching the mockup. PIVOT_ROADMAP §B.1
// adds the Onboarding screen with a first-paint redirect when the
// learner has no goal yet, plus a dedicated 'al' tab for the Atlas
// (§B.4).

'use client'

import { useEffect, useRef } from 'react'

import type { TabId } from '@/lib/types'
import { useSession } from '@/store/session'
import { Atlas } from '@/screens/Atlas'
import { Atrium } from '@/screens/Atrium'
import { Capture } from '@/screens/Capture'
import { Konversation } from '@/screens/Konversation'
import { Library } from '@/screens/Library'
import { Onboarding } from '@/screens/Onboarding'
import { Reader } from '@/screens/Reader'
import { WordCard } from '@/screens/WordCard'
import { Voice } from '@/screens/Voice'
import { Export } from '@/screens/Export'
import { getOnboardingState } from '@/lib/api'

const TABS: { id: TabId; label: string }[] = [
  { id: 'on', label: 'Start' },
  { id: 'at', label: 'Mira' },
  { id: 'al', label: 'Atlas' },
  { id: 'kv', label: 'Sprechen' },
  { id: 'cap', label: 'Sehen' },
  { id: 'lib', label: 'Bibliothek' },
  { id: 'rd', label: 'Leseraum' },
  { id: 'wc', label: 'Wortkarte' },
  { id: 'ex', label: 'Export' },
  { id: 'vx', label: 'Stimme' },
]

export default function Page() {
  const activeTab = useSession((s) => s.activeTab)
  const setTab = useSession((s) => s.setTab)
  const hasGoal = useSession((s) => s.hasGoal)
  const setOnboarding = useSession((s) => s.setOnboarding)
  const checked = useRef(false)

  // First-paint reconciliation with the backend. We always trust the
  // server: if the backend says there's no goal yet, we force the
  // Onboarding tab; if there is one, we hydrate the persisted ids so
  // the Atlas screen can rely on them.
  useEffect(() => {
    if (checked.current) return
    checked.current = true
    const ac = new AbortController()
    void getOnboardingState({ signal: ac.signal })
      .then((state) => {
        setOnboarding({
          goalId: state.latest_goal_id,
          planId: state.latest_plan_id,
          hasGoal: state.has_goal,
        })
        if (!state.has_goal) {
          setTab('on')
        }
      })
      .catch(() => {
        // Backend offline: don't yank the user around. Keep them on
        // whatever tab they last had open. The Onboarding tab is still
        // available manually.
      })
    return () => ac.abort()
  }, [setOnboarding, setTab])

  // Belt-and-braces: if for any reason hasGoal flips to false later
  // (e.g. dev tools clear), bounce them back to onboarding.
  const visibleTabs = hasGoal
    ? TABS
    : TABS.filter((t) => t.id === 'on')
  const renderTab = !hasGoal ? 'on' : activeTab

  return (
    <div className="app">
      <div className="header">
        <div className="app-title">ZenkAI</div>
        <div className="tab-bar">
          {visibleTabs.map((t) => (
            <button
              key={t.id}
              className={`tab${renderTab === t.id ? ' active' : ''}`}
              type="button"
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="screen">
        {renderTab === 'on' ? <Onboarding /> : null}
        {renderTab === 'at' ? <Atrium /> : null}
        {renderTab === 'al' ? <Atlas /> : null}
        {renderTab === 'kv' ? <Konversation /> : null}
        {renderTab === 'cap' ? <Capture /> : null}
        {renderTab === 'lib' ? <Library /> : null}
        {renderTab === 'rd' ? <Reader /> : null}
        {renderTab === 'wc' ? <WordCard /> : null}
        {renderTab === 'ex' ? <Export /> : null}
        {renderTab === 'vx' ? <Voice /> : null}
      </div>
    </div>
  )
}
