import Image from 'next/image'
import { Inter } from 'next/font/google'
import StreamlitPage from './streamlit'

const inter = Inter({ subsets: ['latin'] })

export default function Home() {
  return (
      <StreamlitPage />
  )
}
