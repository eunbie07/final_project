import React, { useState, useEffect } from 'react';

const Toast = ({ message, type = 'info', duration = 3000, onClose }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onClose, 300); // Wait for fade out animation
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const getToastStyles = () => {
    const baseStyles = "fixed top-4 right-4 z-50 px-4 py-3 rounded-lg shadow-lg backdrop-blur-sm transition-all duration-300 max-w-sm";
    
    switch (type) {
      case 'success':
        return `${baseStyles} bg-accent/90 text-white border border-accent`;
      case 'error':
        return `${baseStyles} bg-danger/90 text-white border border-danger`;
      case 'warning':
        return `${baseStyles} bg-primary/90 text-white border border-primary`;
      default:
        return `${baseStyles} bg-secondary/90 text-white border border-secondary`;
    }
  };

  const getIcon = () => {
    switch (type) {
      case 'success':
        return '✅';
      case 'error':
        return '❌';
      case 'warning':
        return '⚠️';
      default:
        return 'ℹ️';
    }
  };

  return (
    <div 
      className={`${getToastStyles()} ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-full'}`}
    >
      <div className="flex items-center space-x-2">
        <span className="text-lg">{getIcon()}</span>
        <span className="text-sm font-medium">{message}</span>
        <button 
          onClick={() => {
            setIsVisible(false);
            setTimeout(onClose, 300);
          }}
          className="ml-2 text-white/80 hover:text-white transition-colors"
        >
          ×
        </button>
      </div>
    </div>
  );
};

// Toast Container Component
const ToastContainer = ({ toasts, removeToast }) => {
  return (
    <div className="fixed top-0 right-0 z-50 pointer-events-none">
      {toasts.map((toast, index) => (
        <div 
          key={toast.id} 
          className="pointer-events-auto"
          style={{ marginTop: `${index * 80}px` }}
        >
          <Toast
            message={toast.message}
            type={toast.type}
            duration={toast.duration}
            onClose={() => removeToast(toast.id)}
          />
        </div>
      ))}
    </div>
  );
};

export default ToastContainer;
export { Toast };