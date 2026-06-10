import React from 'react';

export default function Sidebar() {
  return (
    <aside className="w-80 h-full bg-[#161b22] border-r border-gray-800 flex flex-col justify-between p-4">
      <div>
        <h3 className="text-xs font-semibold text-gray-500 mb-4 uppercase tracking-wider">System Monitoring</h3>
        
        <div className="mb-6">
          <div className="flex items-center gap-2 text-sm mb-2 text-gray-300">
            <span className="text-gray-500">⚙️</span> Configuration
          </div>
          <ul className="pl-6 space-y-2 text-sm text-gray-400">
            <li className="flex items-center justify-between hover:text-white cursor-pointer">
              Integration Status <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            </li>
            <li className="hover:text-white cursor-pointer">Surveillance Data Storage</li>
          </ul>
        </div>

        <div>
           <div className="flex items-center gap-2 text-sm mb-2 text-gray-300">
            <span className="text-gray-500">📊</span> Operational Data
          </div>
          <ul className="pl-6 space-y-2 text-sm text-gray-400">
            <li className="hover:text-white cursor-pointer">Real-time Feeds</li>
            <li className="hover:text-white cursor-pointer">Archive Access</li>
            <li className="hover:text-white cursor-pointer">Incident Reports</li>
            <li className="hover:text-white cursor-pointer">Staff Assignments</li>
          </ul>
        </div>
      </div>

      {/* Current User Profile */}
      <div className="bg-[#0d1117] p-4 rounded-lg border border-gray-800">
        <h3 className="text-xs font-semibold text-gray-500 mb-3 uppercase tracking-wider">Current User Profile</h3>
        <div className="flex items-center gap-3 mb-4">
          <img 
            src="/api/placeholder/40/40" 
            alt="Cpt. J. Miller" 
            className="w-10 h-10 rounded-full border border-gray-600"
          />
          <div>
            <p className="text-sm font-medium">Cpt. J. Miller</p>
            <p className="text-xs text-gray-500">Administrator</p>
          </div>
        </div>
        <button className="w-full py-2 bg-teal-600 hover:bg-teal-500 text-white rounded text-sm font-medium transition-colors flex justify-center items-center gap-2">
         LOG OUT
        </button>
      </div>
    </aside>
  );
}