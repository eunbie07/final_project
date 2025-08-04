import React from "react";
import { Link } from "react-router-dom";

const HomePage = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="relative bg-surface pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:pl-4 lg:pr-8">
          <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:max-w-3xl lg:w-full lg:pb-28 xl:pb-32 lg:pr-8">
            <main className="mt-10 sm:mt-12 md:mt-16 lg:mt-20 xl:mt-28">
              <div className="sm:text-center lg:text-left">
                <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-6 leading-tight">
                  <span className="block">Transform Your Space</span>{" "}
                  <span className="block text-primary">with AI Technology</span>
                </h1>
                <p className="text-lg text-text-secondary mb-8">
                  Measure room dimensions from photos, <br />
                  design with 3D furniture placement,
                  <br />
                  and get AI-powered interior suggestions.
                </p>
                <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                  <div className="rounded-md shadow">
                    <Link
                      to="/room-planner"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-primary hover:bg-secondary md:py-4 md:text-lg md:px-10"
                    >
                      Get Started
                    </Link>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
        <div className="lg:absolute lg:inset-y-0 lg:right-0 lg:w-1/2 lg:pl-1 lg:pr-60">
          <div className="aspect-[4/3] w-full bg-gray-200 sm:aspect-[3/2] lg:aspect-auto lg:h-full overflow-hidden">
            <img
              className="w-full h-full object-cover"
              src="/hero-bg.jpg"
              alt="Modern home interior design"
              loading="eager"
            />
          </div>
        </div>
      </div>

      {/* Main Services Section */}
      <div className="py-20 bg-surface">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-left mb-16">
            <h2 className="text-base text-primary font-semibold tracking-wide uppercase">
              Choose Your Service
            </h2>
            <p className="text-4xl md:text-5xl font-bold text-text-primary mb-6 leading-tight">
              What Would You Like To Do?
            </p>
            <p className="text-lg text-text-secondary mb-8">
              Start your home journey by selecting the service that best fits
              your needs
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {/* Find Your Home Card */}
            <div className="group relative bg-surface rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-border">
              <div className="text-center">
                <div className="mx-auto w-16 h-16 bg-text-primary rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                  <svg
                    className="w-8 h-8 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
                    />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-text-primary mb-4 group-hover:text-text-secondary transition-colors duration-300">
                  Find Your Home
                </h3>
                <p className="text-text-secondary mb-8 leading-relaxed">
                  라이프스타일과 취향에 맞는 완벽한 부동산을 찾아보세요. 고급
                  필터링 옵션으로 엄선된 매물을 탐색하세요.
                </p>
                <div className="space-y-3 mb-8 text-left">
                  <div className="flex items-center text-text-secondary">
                    <svg
                      className="w-5 h-5 text-text-primary mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    고급 부동산 검색 및 필터링
                  </div>
                  <div className="flex items-center text-text-secondary">
                    <svg
                      className="w-5 h-5 text-text-primary mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    실시간 시장 데이터 및 트렌드
                  </div>
                  <div className="flex items-center text-text-secondary">
                    <svg
                      className="w-5 h-5 text-text-primary mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    동네 정보 및 분석
                  </div>
                </div>
                <Link
                  to="/find-house"
                  className="inline-block w-full bg-primary text-white font-semibold py-4 px-8 rounded-xl hover:bg-secondary transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                >
                  Start House Hunting
                </Link>
              </div>
            </div>

            {/* 2D/3D Room Planner Card */}
            <div className="group relative bg-surface rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-border">
              <div className="text-center">
                <div className="mx-auto w-16 h-16 bg-text-primary rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                  <svg
                    className="w-8 h-8 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                    />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-text-primary mb-4 group-hover:text-text-secondary transition-colors duration-300">
                  Room Planner
                </h3>
                <p className="text-text-secondary mb-8 leading-relaxed">
                  AI 기반 방 분석으로 공간을 측정하고 시각화하세요.
                  <br />
                  단일 사진으로 상세한 평면도와 3D 모델을 생성하세요.
                </p>
                <div className="space-y-3 mb-8 text-left">
                  <div className="flex items-center text-text-secondary">
                    <svg
                      className="w-5 h-5 text-text-primary mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    AI 기반 방 측정
                  </div>
                  <div className="flex items-center text-text-secondary">
                    <svg
                      className="w-5 h-5 text-text-primary mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    인터랙티브 3D 시각화
                  </div>
                  <div className="flex items-center text-text-secondary">
                    <svg
                      className="w-5 h-5 text-text-primary mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    가구 배치 시뮬레이션
                  </div>
                </div>
                <Link
                  to="/room-planner"
                  className="inline-block w-full bg-primary text-white font-semibold py-4 px-8 rounded-xl hover:bg-secondary transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                >
                  Start Room Planning
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
