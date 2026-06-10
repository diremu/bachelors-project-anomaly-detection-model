import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const AUTH_USER_KEY = 'console_auth_user';

type AuthProps = {
  onAuthSuccess: () => void;
};

export const Auth: React.FC<AuthProps> = ({ onAuthSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [badge, setBadge] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setMessage('');

    if (!email.trim() || !password.trim()) {
      setError('Email and password are required.');
      return;
    }

    if (!email.includes('@') || email.length < 5) {
      setError('Please enter a valid email address.');
      return;
    }

    if (!isLogin && password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }

    const stored = localStorage.getItem(AUTH_USER_KEY);
    const existingUser = stored ? JSON.parse(stored) : null;

    if (isLogin) {
      if (!existingUser) {
        setError('No account found. Please request access first.');
        return;
      }

      if (existingUser.email !== email || existingUser.password !== password) {
        setError('Invalid email or password.');
        return;
      }

      onAuthSuccess();
      navigate('/overview');
      return;
    }

    if (existingUser && existingUser.email === email) {
      setError('An account with this email already exists. Please log in.');
      return;
    }

    localStorage.setItem(
      AUTH_USER_KEY,
      JSON.stringify({ email, password, badge: badge.trim() || undefined })
    );

    setMessage('Account created successfully. Redirecting...');
    onAuthSuccess();
    navigate('/overview');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0d1117] font-poppins px-4">
      <div className="max-w-md w-full space-y-8 bg-[#161b22] p-8 rounded-xl border border-gray-800 shadow-2xl">
        
        <div className="text-center">
          <div className="mx-auto h-12 w-12 bg-teal-500/10 rounded-full flex items-center justify-center border border-teal-500/30 mb-4">
             <span className="text-xl">🛡️</span>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-wide">
            {isLogin ? 'SECURE LOGIN' : 'REQUEST ACCESS'}
          </h2>
          <p className="mt-2 text-sm text-gray-400">
            Inmate Anomaly Detection System
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {(error || message) && (
            <div className="rounded-md px-4 py-3 text-sm text-left">
              {error ? (
                <p className="text-red-400 bg-red-500/10 border border-red-500/20 rounded p-3">{error}</p>
              ) : (
                <p className="text-emerald-300 bg-emerald-500/10 border border-emerald-500/20 rounded p-3">{message}</p>
              )}
            </div>
          )}
          <div className="space-y-4 rounded-md shadow-sm">
            {!isLogin && (
              <div>
                <label htmlFor="badge" className="sr-only">Badge Number</label>
                <input
                  id="badge"
                  name="badge"
                  type="text"
                  value={badge}
                  onChange={(event) => setBadge(event.target.value)}
                  className="appearance-none relative block w-full px-3 py-3 border border-gray-700 bg-[#0d1117] text-white rounded focus:outline-none focus:ring-teal-500 focus:border-teal-500 focus:z-10 sm:text-sm transition-colors"
                  placeholder="Badge Number"
                />
              </div>
            )}
            <div>
              <label htmlFor="email" className="sr-only">Email address</label>
              <input
                id="email"
                name="email"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="appearance-none relative block w-full px-3 py-3 border border-gray-700 bg-[#0d1117] text-white rounded focus:outline-none focus:ring-teal-500 focus:border-teal-500 focus:z-10 sm:text-sm transition-colors"
                placeholder="Operator Email"
                required
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="appearance-none relative block w-full px-3 py-3 border border-gray-700 bg-[#0d1117] text-white rounded focus:outline-none focus:ring-teal-500 focus:border-teal-500 focus:z-10 sm:text-sm transition-colors"
                placeholder="Passcode"
                required
              />
            </div>
          </div>

          <div>
            <button type="submit"
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded text-white bg-teal-600 hover:bg-teal-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0d1117] focus:ring-teal-500 transition-colors">
              {isLogin ? 'AUTHENTICATE' : 'SUBMIT CREDENTIALS'}
            </button>
          </div>
        </form>

        <div className="text-center mt-4">
          <button 
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
              setMessage('');
            }}
            className="text-xs text-teal-500 hover:text-teal-400 font-medium transition-colors"
          >
            {isLogin ? 'Need an account? Request Access' : 'Already authorized? Log in'}
          </button>
        </div>
      </div>
    </div>
  );
};