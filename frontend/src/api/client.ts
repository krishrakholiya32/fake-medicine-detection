import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
})

export interface TopPrediction {
  label: string
  confidence: number
}

export interface PillCheckResult {
  id: number
  claimed_drug: string
  predicted_drug: string
  confidence: number
  verdict: 'match' | 'mismatch' | 'uncertain'
  top_predictions: TopPrediction[]
}

export interface PillCheckRecord {
  id: number
  original_filename: string
  claimed_drug: string
  predicted_drug: string
  confidence: number
  verdict: 'match' | 'mismatch' | 'uncertain'
  created_at: string
}

export const checkApi = {
  pill: (file: File, claimedDrug: string) => {
    const form = new FormData()
    form.append('file', file)
    form.append('claimed_drug', claimedDrug)
    return api.post<PillCheckResult>('/check/pill', form).then((r) => r.data)
  },
}

export const historyApi = {
  list: (params?: { verdict?: string; limit?: number; offset?: number }) =>
    api.get<PillCheckRecord[]>('/history', { params }).then((r) => r.data),
}
