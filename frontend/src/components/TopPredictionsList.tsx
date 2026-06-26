import type { TopPrediction } from '../api/client'

interface Props {
  predictions: TopPrediction[]
}

export default function TopPredictionsList({ predictions }: Props) {
  return (
    <div className="rounded-lg border border-pc-border bg-pc-surface p-4">
      <h3 className="mb-2 text-sm font-semibold text-pc-muted">Top predictions</h3>
      <ul className="space-y-2">
        {predictions.map((p, i) => (
          <li key={p.label} className="flex items-center justify-between text-sm">
            <span>
              {i + 1}. {p.label}
            </span>
            <span className="text-pc-muted">{(p.confidence * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
