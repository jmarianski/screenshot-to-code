import { useState, useEffect } from 'react';

export default function DarkModeToggle() {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);

  useEffect(() => {
    // Check if dark mode is already enabled
    const isDark =
      document.documentElement.classList.contains('dark') ||
      localStorage.getItem('darkMode') === 'true';
    setIsDarkMode(isDark);

    // Apply dark mode class if enabled
    if (isDark) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);

    if (newDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('darkMode', 'true');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('darkMode', 'false');
    }
  };

  return (
    <button
      onClick={toggleDarkMode}
      className='relative flex items-center justify-center w-10 h-10 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 transition-colors duration-200 group'
      title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      <div className='relative w-5 h-5'>
        {/* Sun icon */}
        <svg
          className={`absolute inset-0 w-5 h-5 text-slate-600 transition-all duration-300 ${
            isDarkMode
              ? 'opacity-0 scale-0 rotate-90'
              : 'opacity-100 scale-100 rotate-0'
          }`}
          fill='none'
          stroke='currentColor'
          viewBox='0 0 24 24'
        >
          <path
            strokeLinecap='round'
            strokeLinejoin='round'
            strokeWidth={2}
            d='M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z'
          />
        </svg>

        {/* Moon icon */}
        <svg
          className={`absolute inset-0 w-5 h-5 text-slate-400 transition-all duration-300 ${
            isDarkMode
              ? 'opacity-100 scale-100 rotate-0'
              : 'opacity-0 scale-0 -rotate-90'
          }`}
          fill='none'
          stroke='currentColor'
          viewBox='0 0 24 24'
        >
          <path
            strokeLinecap='round'
            strokeLinejoin='round'
            strokeWidth={2}
            d='M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z'
          />
        </svg>
      </div>

      {/* Hover tooltip-like indicator */}
      <div className='absolute -bottom-8 left-1/2 transform -translate-x-1/2 px-2 py-1 text-xs font-medium text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none'>
        {isDarkMode ? 'Light mode' : 'Dark mode'}
      </div>
    </button>
  );
}
