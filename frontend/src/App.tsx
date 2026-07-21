// src/App.tsx
import { Sidebar } from '@/components/layout/Sidebar'
import { FactSheet } from '@/pages/FactSheet'
import { Logs } from '@/pages/Logs'
import { Register } from '@/pages/Register'
import { Verify } from '@/pages/Verify'
import { Navigate, Route, Routes } from 'react-router-dom'

export function App() {
  return (
    <div className="flex h-full overflow-hidden bg-[hsl(var(--bg))]">
      <Sidebar />
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to="/register" replace />} />
          <Route path="/register" element={<Register />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/factsheet" element={<FactSheet />} />
          <Route path="/logs" element={<Logs />} />
        </Routes>
      </main>
    </div>
  )
}
