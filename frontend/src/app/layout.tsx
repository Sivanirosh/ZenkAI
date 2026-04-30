// Root layout: imports design tokens + global styles and loads the mockup fonts.

import type { Metadata } from 'next'
import { DM_Sans, Lora } from 'next/font/google'
import '../styles/tokens.css'
import '../styles/globals.css'

const lora = Lora({
  subsets: ['latin'],
  variable: '--font-serif',
  display: 'swap',
  style: ['normal', 'italic'],
  weight: ['400', '500'],
})

const dmSans = DM_Sans({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
  weight: ['400', '500'],
})

export const metadata: Metadata = {
  title: 'ZenkAI',
  description:
    'AI-assisted classical German literature reader with voice, inline annotation, and spaced-repetition vocabulary.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="de" className={`${lora.variable} ${dmSans.variable}`}>
      <body>{children}</body>
    </html>
  )
}
