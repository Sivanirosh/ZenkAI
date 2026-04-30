// App shell: header + 4-tab router matching the mockup.

'use client'

import type { TabId } from '@/lib/types'
import { useSession } from '@/store/session'
import { Library } from '@/screens/Library'
import { Reader } from '@/screens/Reader'
import { WordCard } from '@/screens/WordCard'
import { Voice } from '@/screens/Voice'
import { Export } from '@/screens/Export'

const TABS: { id: TabId; label: string }[] = [
  { id: 'lib', label: 'Bibliothek' },
  { id: 'rd', label: 'Leseraum' },
  { id: 'wc', label: 'Wortkarte' },
  { id: 'ex', label: 'Export' },
  { id: 'vx', label: 'Stimme' },
]

export default function Page() {
  const activeTab = useSession((s) => s.activeTab)
  const setTab = useSession((s) => s.setTab)

  return (
    <div className="app">
      <div className="header">
        <div className="app-title">ZenkAI</div>
        <div className="tab-bar">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab${activeTab === t.id ? ' active' : ''}`}
              type="button"
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="screen">
        {activeTab === 'lib' ? <Library /> : null}
        {activeTab === 'rd' ? <Reader /> : null}
        {activeTab === 'wc' ? <WordCard /> : null}
        {activeTab === 'ex' ? <Export /> : null}
        {activeTab === 'vx' ? <Voice /> : null}
      </div>
    </div>
  )
}
