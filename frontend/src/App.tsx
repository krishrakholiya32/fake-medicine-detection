import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Check from './pages/Check'
import History from './pages/History'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Check />} />
        <Route path="/history" element={<History />} />
      </Route>
    </Routes>
  )
}
