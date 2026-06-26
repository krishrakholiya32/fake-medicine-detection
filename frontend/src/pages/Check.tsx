import { useState } from 'react'
import { checkApi, type PillCheckResult } from '../api/client'
import UploadDropzone from '../components/UploadDropzone'
import VerdictBadge from '../components/VerdictBadge'
import TopPredictionsList from '../components/TopPredictionsList'

export default function Check() {
  const [claimedDrug, setClaimedDrug] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<PillCheckResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit() {
    if (!file || !claimedDrug.trim()) {
      setError('Select a photo and enter the claimed drug name first')
      return
    }
    setError(null)
    setLoading(true)
    try {
      const res = await checkApi.pill(file, claimedDrug.trim())
      setResult(res)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Check failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <h2 className="text-2xl font-semibold">Check a pill</h2>

      <div className="space-y-3">
        <label className="block text-sm text-pc-muted">What is this pill supposed to be?</label>
        <input
          value={claimedDrug}
          onChange={(e) => setClaimedDrug(e.target.value)}
          placeholder="e.g. Paracetamol 500mg"
          className="w-full rounded-md border border-pc-border bg-pc-surface px-3 py-2 text-white outline-none focus:border-pc-accent"
        />
      </div>

      <UploadDropzone onFile={setFile} disabled={loading} />
      {file && <p className="text-sm text-pc-muted">Selected: {file.name}</p>}

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="rounded-md bg-pc-accent px-4 py-2 font-medium text-black disabled:opacity-50"
      >
        {loading ? 'Checking…' : 'Check pill'}
      </button>

      {error && <p className="text-pc-mismatch">{error}</p>}

      {result && (
        <div className="space-y-4">
          <VerdictBadge verdict={result.verdict} confidence={result.confidence} />
          <p className="text-sm text-pc-muted">
            Claimed: <span className="text-white">{result.claimed_drug}</span> · Predicted:{' '}
            <span className="text-white">{result.predicted_drug}</span>
          </p>
          <TopPredictionsList predictions={result.top_predictions} />
        </div>
      )}
    </div>
  )
}
