import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const Navbar = () => {
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
    navigate("/");
  };

  return (
    <nav className="bg-surface shadow-sm fixed w-full z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:pl-8 lg:pr-2">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center">
            <a href="/" className="text-2xl font-bold text-text-primary">
              방측정
            </a>
          </div>
          <div className="hidden md:flex items-center space-x-1">
            <a
              href="/"
              className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-background rounded-md transition-colors font-bold"
            >
              Room Planner
            </a>
            <a
              href="/ai-interior"
              className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-background rounded-md transition-colors font-bold"
            >
              🎨 AI Interior
            </a>

            {/* 인증 관련 버튼 */}
            {isAuthenticated ? (
              <div className="relative">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-2 text-text-secondary hover:text-text-primary transition-colors font-bold"
                >
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                    <span className="text-white text-sm font-bold">
                      {user?.email?.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <span className="font-bold">{user?.email}</span>
                </button>

                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-48 bg-surface border border-border rounded-lg shadow-lg z-50">
                    <div className="py-2">
                      <div className="px-4 py-2 text-sm text-text-secondary border-b border-border font-bold">
                        {user?.email}
                      </div>
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          navigate("/my-rooms");
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-text-secondary hover:bg-background font-bold"
                      >
                        내 방 목록
                      </button>
                      <button
                        onClick={handleLogout}
                        className="block w-full text-left px-4 py-2 text-sm text-text-secondary hover:bg-background font-bold"
                      >
                        로그아웃
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <>
                <button
                  onClick={() => navigate("/login")}
                  className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-background rounded-md transition-colors font-bold"
                >
                  Login
                </button>
                <button
                  onClick={() => navigate("/signup")}
                  className="bg-primary text-white px-4 py-2 rounded-md hover:bg-secondary font-bold transition-colors"
                >
                  Sign Up
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
