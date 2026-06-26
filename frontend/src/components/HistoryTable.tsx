import { format } from 'date-fns'
import type { PillCheckRecord } from '../api/client'

interface Props {
  checks: PillCheckRecord[]
}

const VERDICT_COLOR: Record<string, string> = {
  match: 'text-pc-match',
  mismatch: 'text-pc-mismatch',
  uncertain: 'text-pc-uncertain',
}

export default function HistoryTable({ checks }: Props) {
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-pc-muted">
        <tr className="border-b border-pc-border">
          <th className="py-2">File</th>
          <th className="py-2">Claimed</th>
          <th className="py-2">Predicted</th>
          <th className="py-2">Confidence</th>
          <th className="py-2">Verdict</th>
          <th className="py-2">When</th>
        </tr>
      </thead>
      <tbody>
        {checks.map((c) => (
          <tr key={c.id} className="border-b border-pc-border/50">
            <td className="py-2">{c.original_filename}</td>
            <td className="py-2">{c.claimed_drug}</td>
            <td className="py-2">{c.predicted_drug}</td>
            <td className="py-2">{(c.confidence * 100).toFixed(1)}%</td>
            <td className={`py-2 font-semibold ${VERDICT_COLOR[c.verdict]}`}>{c.verdict}</td>
            <td className="py-2">{format(new Date(c.created_at), 'MMM d, HH:mm')}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
