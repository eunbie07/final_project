import React, { createContext, useContext, useState, useEffect } from 'react';
import { getCurrentUser } from '../utils/api';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [isLoading, setIsLoading] = useState(true);

  // 토큰이 있으면 사용자 정보 조회
  useEffect(() => {
    const initAuth = async () => {
      if (token) {
        try {
          const userData = await getCurrentUser();
          setUser(userData);
        } catch (error) {
          console.error('토큰 검증 실패:', error);
          // 토큰이 유효하지 않으면 제거
          localStorage.removeItem('token');
          setToken(null);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, [token]);

  const login = (userData, accessToken) => {
    setUser(userData);
    setToken(accessToken);
    localStorage.setItem('token', accessToken);
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('token');
  };

  const value = {
    user,
    token,
    isLoading,
    isAuthenticated: !!user,
    login,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};