import { useEffect, useState } from 'react'
import { historyApi, type PillCheckRecord } from '../api/client'
import HistoryTable from '../components/HistoryTable'

export default function History() {
  const [checks, setChecks] = useState<PillCheckRecord[]>([])

  useEffect(() => {
    historyApi.list({ limit: 50 }).then(setChecks)
  }, [])

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">History</h2>
      <div className="rounded-lg border border-pc-border bg-pc-surface p-4">
        <HistoryTable checks={checks} />
      </div>
    </div>
  )
}
