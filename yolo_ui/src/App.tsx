import React from 'react';
import UnifiedDetector from './components/UnifiedDetector';

export default function App(): JSX.Element {
  return (
    <div className="min-h-screen bg-slate-50 font-yekan">
      <header className="max-w-6xl mx-auto p-6">
        <h1 className="w-full text-2xl yekan-fond text-center md:text-3xl font-bold tracking-tight">سامانه تشخیص پسیل پسته</h1>
      </header>

      <main className="max-w-6xl mx-auto p-6 pt-0">
        <UnifiedDetector />
      </main>
    </div>
  );
}
