import React from 'react';

const LoadingSpinner = ({ size = 'medium', text = '로딩 중...', overlay = false }) => {
  const getSizeClass = () => {
    switch (size) {
      case 'small':
        return 'w-4 h-4';
      case 'large':
        return 'w-8 h-8';
      default:
        return 'w-6 h-6';
    }
  };

  const spinner = (
    <div className={`animate-spin rounded-full border-2 border-gray-300 border-t-blue-500 ${getSizeClass()}`} />
  );

  if (overlay) {
    return (
      <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 shadow-lg flex flex-col items-center space-y-3">
          <div className="animate-spin rounded-full border-4 border-gray-300 border-t-blue-500 w-8 h-8" />
          <span className="text-gray-700 font-medium">{text}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center space-x-2">
      {spinner}
      {text && <span className="text-sm text-gray-600">{text}</span>}
    </div>
  );
};

// Button with loading state
const LoadingButton = ({ 
  loading = false, 
  children, 
  disabled = false,
  loadingText = '처리 중...',
  className = '',
  ...props 
}) => {
  return (
    <button 
      disabled={disabled || loading}
      className={`relative ${className} ${loading ? 'cursor-not-allowed' : ''}`}
      {...props}
    >
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <LoadingSpinner size="small" text="" />
        </div>
      )}
      <span className={loading ? 'opacity-0' : 'opacity-100'}>
        {loading ? loadingText : children}
      </span>
    </button>
  );
};

export default LoadingSpinner;
export { LoadingButton };