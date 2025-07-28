import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Navbar = () => {
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
    navigate('/');
  };

  return (
    <nav className="bg-surface shadow-lg py-6 px-6 fixed w-full z-50 border-b border-border">
      <div className="container mx-auto flex justify-between items-center">
        <a href="/" className="text-3xl font-black text-text-primary hover:text-primary transition-colors duration-200">이집맛집</a>
        
        <div className="flex items-center space-x-8">
          <div className="space-x-8">
            <a href="/find-house" className="text-lg text-text-secondary hover:text-primary transition-colors duration-200 font-bold hover:scale-105 transform">Find Your Dream Home</a>
            <a href="/room-planner" className="text-lg text-text-secondary hover:text-primary transition-colors duration-200 font-bold hover:scale-105 transform">2D/3D Room Planner</a>
            <a href="/ai-design" className="text-lg text-text-secondary hover:text-primary transition-colors duration-200 font-bold hover:scale-105 transform">AI Interior Design</a>
          </div>
          
          {/* 인증 관련 버튼 */}
          <div className="flex items-center space-x-4">
            {isAuthenticated ? (
              <div className="relative">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-2 text-text-primary hover:text-primary transition-colors"
                >
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                    <span className="text-white text-sm font-bold">
                      {user?.email?.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <span className="font-medium">{user?.email}</span>
                </button>
                
                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-48 bg-surface border border-border rounded-lg shadow-lg z-50">
                    <div className="py-2">
                      <div className="px-4 py-2 text-sm text-text-secondary border-b border-border">
                        {user?.email}
                      </div>
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          navigate('/my-rooms');
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-text-primary hover:bg-background"
                      >
                        내 방 목록
                      </button>
                      <button
                        onClick={handleLogout}
                        className="block w-full text-left px-4 py-2 text-sm text-text-primary hover:bg-background"
                      >
                        로그아웃
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-x-4">
                <button
                  onClick={() => navigate('/login')}
                  className="text-lg text-text-secondary hover:text-primary transition-colors duration-200 font-bold"
                >
                  로그인
                </button>
                <button
                  onClick={() => navigate('/signup')}
                  className="bg-primary hover:bg-primary/90 text-white px-4 py-2 rounded-lg font-bold transition-colors duration-200"
                >
                  회원가입
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
