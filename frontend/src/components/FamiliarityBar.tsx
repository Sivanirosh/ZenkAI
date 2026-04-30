// Reusable 5-segment familiarity bar.

'use client'

interface Props {
  familiarity: number
  showLabels?: boolean
}

export function FamiliarityBar({ familiarity, showLabels = true }: Props) {
  return (
    <div>
      <div className="familiarity-bar">
        {[0, 1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className={`fam-segment ${i < Math.max(familiarity, 1) ? 'filled' : 'empty'}`}
          />
        ))}
      </div>
      {showLabels ? (
        <div className="fam-labels">
          <span>Erstkontakt</span>
          <span>Verinnerlicht</span>
        </div>
      ) : null}
    </div>
  )
}
