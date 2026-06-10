import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';

const AUTH_USER_KEY = 'console_auth_user';

type UserProfile = {
  displayName?: string;
  badge?: string;
  email?: string;
};

export default function Sidebar() {
  const [user, setUser] = useState<UserProfile | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(AUTH_USER_KEY);
    if (!stored) return;

    try {
      setUser(JSON.parse(stored));
    } catch {
      setUser(null);
    }
  }, []);

  // A helper function to apply active styles using Tailwind
  const linkStyles = ({ isActive }: { isActive: boolean }) =>
    `block py-1.5 px-3 rounded transition-colors ${
      isActive ? 'bg-[#1c2128] text-white font-medium border-l-2 border-teal-500' : 'text-gray-400 hover:text-white hover:bg-[#1c2128]/50'
    }`;

  return (
    <aside className="w-80 h-full bg-[#161b22] border-r border-gray-800 flex flex-col justify-between p-4 font-poppins">
      <div>
        <h3 className="text-xs font-semibold text-gray-500 mb-4 uppercase tracking-wider">System Monitoring</h3>
        
        <div className="mb-6">
           <div className="flex items-center gap-2 text-sm mb-2 text-gray-300">
            <span className="text-gray-500">📊</span> Operational Data
          </div>
          <nav className="pl-6 space-y-1 text-sm">
            <NavLink to="/overview" className={linkStyles}>System Overview</NavLink>
            <NavLink to="/live-feeds" className={linkStyles}>Real-time Feeds</NavLink>
            <NavLink to="/reports" className={linkStyles}>Incident Reports</NavLink>
          </nav>
        </div>

        <div>
          <div className="flex items-center gap-2 text-sm mb-2 text-gray-300">
            <span className="text-gray-500">⚙️</span> Configuration
          </div>
          <nav className="pl-6 space-y-1 text-sm">
            <NavLink to="/config" className={linkStyles}>Storage & Integration</NavLink>
          </nav>
        </div>
      </div>

      <div className="bg-[#0d1117] p-4 rounded-lg border border-gray-800">
        <h3 className="text-xs font-semibold text-gray-500 mb-3 uppercase tracking-wider">Current User Profile</h3>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full border border-gray-600 bg-gray-700 flex items-center justify-center text-sm font-bold">
            {user?.displayName ?
              user.displayName
                .split(' ')
                .map((word) => word[0])
                .join('')
                .slice(0, 2)
                .toUpperCase()
              : 'JM'}
          </div>
          <div>
            <p className="text-sm font-medium">{user?.displayName ?? 'Cpt. J. Miller'}</p>
            <p className="text-xs text-gray-500">{user?.badge ? `Badge ${user.badge}` : 'Administrator'}</p>
          </div>
        </div>
        <button className="w-full py-2 bg-teal-600/20 text-teal-500 hover:bg-teal-600 hover:text-white rounded text-sm font-medium transition-colors border border-teal-600/30">
          LOG OUT
        </button>
      </div>
    </aside>
  );
}