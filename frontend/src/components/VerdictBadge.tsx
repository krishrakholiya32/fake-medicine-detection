interface Props {
  verdict: 'match' | 'mismatch' | 'uncertain'
  confidence: number
}

const STYLES = {
  match: 'border-pc-match bg-pc-match/10 text-pc-match',
  mismatch: 'border-pc-mismatch bg-pc-mismatch/10 text-pc-mismatch',
  uncertain: 'border-pc-uncertain bg-pc-uncertain/10 text-pc-uncertain',
}

export default function VerdictBadge({ verdict, confidence }: Props) {
  return (
    <div className={`inline-flex items-center gap-3 rounded-lg border px-5 py-3 ${STYLES[verdict]}`}>
      <span className="text-xl font-bold uppercase tracking-wide">{verdict}</span>
      <span className="text-sm opacity-80">{(confidence * 100).toFixed(1)}% confidence</span>
    </div>
  )
}
