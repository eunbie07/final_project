import React, { useState, useEffect, useCallback } from 'react';

// 메인 앱 컴포넌트
export default function App() {
  return (
    <div className="bg-gray-100 flex items-center justify-center min-h-screen font-sans">
      <Signup />
    </div>
  );
}

// 회원가입 컴포넌트
function Signup() {
  // 상태 변수들 정의 (email 기반으로 수정)
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  // 유효성 검사 관련 상태 (email 기반으로 수정)
  const [isEmailChecked, setIsEmailChecked] = useState(false);
  const [emailMessage, setEmailMessage] = useState({ text: '', color: '' });
  const [passwordMessage, setPasswordMessage] = useState({ text: '', color: '' });
  const [isSubmitDisabled, setIsSubmitDisabled] = useState(true);

  // 이메일 중복 확인 함수 (백엔드 API 호출 시뮬레이션)
  const handleEmailCheck = () => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setEmailMessage({ text: '유효한 이메일 형식이 아닙니다.', color: 'text-red-500' });
      setIsEmailChecked(false);
      return;
    }
    
    setEmailMessage({ text: '확인 중...', color: 'text-gray-500' });
    
    // 실제로는 여기서 fetch를 사용해 백엔드 API를 호출합니다.
    setTimeout(() => {
      if (email === 'admin@test.com' || email === 'user@test.com') {
        setEmailMessage({ text: '이미 사용 중인 이메일입니다.', color: 'text-red-500' });
        setIsEmailChecked(false);
      } else {
        setEmailMessage({ text: '사용 가능한 이메일입니다.', color: 'text-green-500' });
        setIsEmailChecked(true);
      }
    }, 1000);
  };

  // 이메일 입력값이 변경되면, 중복 확인 상태를 초기화
  const handleEmailChange = (e) => {
    setEmail(e.target.value);
    setIsEmailChecked(false);
    setEmailMessage({ text: '', color: '' });
  };

  // 폼 전체 유효성 검사 및 제출 버튼 활성화 로직
  const validateForm = useCallback(() => {
    const isPasswordValid = password.length >= 8;
    const isPasswordMatch = password !== '' && password === confirmPassword;

    // 비밀번호 일치 여부 메시지 업데이트
    if (confirmPassword) {
      if (isPasswordMatch) {
        setPasswordMessage({ text: '비밀번호가 일치합니다.', color: 'text-green-500' });
      } else {
        setPasswordMessage({ text: '비밀번호가 일치하지 않습니다.', color: 'text-red-500' });
      }
    } else {
      setPasswordMessage({ text: '', color: '' });
    }
    
    // 최종적으로 모든 조건이 만족될 때만 가입하기 버튼 활성화
    if (isEmailChecked && isPasswordValid && isPasswordMatch) {
      setIsSubmitDisabled(false);
    } else {
      setIsSubmitDisabled(true);
    }
  }, [isEmailChecked, password, confirmPassword]);

  // 비밀번호 관련 상태가 변경될 때마다 유효성 검사 실행
  useEffect(() => {
    validateForm();
  }, [password, confirmPassword, isEmailChecked, validateForm]);

  // 폼 제출 핸들러
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isSubmitDisabled) {
      console.log('제출 정보:', { email, password });
      // 여기에 실제 백엔드 API로 데이터를 전송하는 로직을 추가합니다.
      alert('회원가입이 완료되었습니다!');
    }
  };

  return (
    <div className="w-full max-w-md mx-auto p-4">
        <div 
          className="bg-white/70 border border-gray-200 rounded-2xl shadow-lg p-8" 
          style={{ backdropFilter: 'blur(10px)' }}
        >
            <div className="text-center mb-8">
                <h1 className="text-3xl font-bold text-gray-800">회원가입</h1>
                <p className="text-gray-500 mt-2">서비스 이용을 위해 가입해주세요.</p>
            </div>
            <form onSubmit={handleSubmit}>
                <div className="mb-4">
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">이메일</label>
                    <div className="flex">
                        <input 
                          type="email" 
                          id="email" 
                          value={email}
                          onChange={handleEmailChange}
                          className="flex-grow w-full px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition" 
                          placeholder="이메일을 입력하세요" 
                          required 
                        />
                        <button 
                          type="button" 
                          onClick={handleEmailCheck}
                          className="px-4 py-2 text-sm font-semibold text-white bg-blue-500 rounded-r-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition whitespace-nowrap"
                        >
                          중복 확인
                        </button>
                    </div>
                    <p className={`text-xs mt-1 h-4 ${emailMessage.color}`}>{emailMessage.text}</p>
                </div>

                <div className="mb-4">
                    <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">비밀번호</label>
                    <input 
                      type="password" 
                      id="password" 
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="w-full px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition" 
                      placeholder="8자 이상 입력하세요" 
                      required 
                    />
                </div>

                {/* 비밀번호 확인 입력 그룹 */}
                <div className="mb-6">
                    <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 mb-1">비밀번호 확인</label>
                    <input 
                      type="password" 
                      id="confirmPassword" 
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="w-full px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition" 
                      placeholder="비밀번호를 다시 입력하세요" 
                      required 
                    />
                    <p className={`text-xs mt-1 h-4 ${passwordMessage.color}`}>{passwordMessage.text}</p>
                </div>
                <div>
                    <button 
                      type="submit" 
                      disabled={isSubmitDisabled}
                      className="w-full px-4 py-3 text-sm font-semibold text-white bg-blue-500 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                        가입하기
                    </button>
                </div>
            </form>
        </div>
    </div>
  );
}
