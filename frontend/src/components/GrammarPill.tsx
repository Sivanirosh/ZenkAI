// Colour-coded case / role pill used inside the Word Card grammar grid.

'use client'

interface Props {
  label: string
  value: string
}

export function GrammarPill({ label, value }: Props) {
  return (
    <div className="grammar-cell">
      <div className="grammar-cell-label">{label}</div>
      <div className="grammar-cell-value">{value}</div>
    </div>
  )
}
