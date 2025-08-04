import React from 'react';

const RoomPlannerPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-primary mb-6">
            방 측정 서비스
          </h1>
          <div className="bg-white rounded-lg shadow-lg p-8 max-w-2xl mx-auto">
            <div className="mb-6">
              <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-2m-14 0h2m-2 0h-2m14 0V9a2 2 0 00-2-2H9a2 2 0 002 2v10M7 7h10" />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                서비스 준비 중
              </h2>
              <p className="text-gray-600 text-lg leading-relaxed">
                방 크기 측정 및 가구 배치 서비스가 곧 제공됩니다.<br />
                AI 기술을 활용한 정확한 방 측정과<br />
                3D 가구 배치 시뮬레이션을 경험해보세요.
              </p>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">
                주요 기능
              </h3>
              <ul className="text-left text-gray-600 space-y-2">
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-primary rounded-full mr-3"></span>
                  사진으로 방 크기 자동 측정
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-primary rounded-full mr-3"></span>
                  3D 가구 배치 시뮬레이션
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-primary rounded-full mr-3"></span>
                  공간 활용도 분석
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-primary rounded-full mr-3"></span>
                  레이아웃 저장 및 공유
                </li>
              </ul>
            </div>
            
            <div className="mt-6 text-sm text-gray-500">
              * 서비스 오픈 시 알림을 받으시려면 회원가입을 해주세요
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoomPlannerPage;