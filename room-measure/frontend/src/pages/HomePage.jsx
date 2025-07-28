import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="relative min-h-screen bg-cover bg-center flex items-center justify-center" style={{ backgroundImage: 'url(/hero-bg.jpg)' }}>
      <div className="absolute inset-0 bg-black opacity-60"></div>
      <div className="relative z-10 container mx-auto px-4 pt-24 md:pt-28 pb-8 text-center">
      <h1 className="text-6xl md:text-7xl font-black text-text-primary mb-8 leading-none tracking-tight">
        Welcome to <span className="text-primary bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">이집맞집</span>
      </h1>
      <p className="text-xl md:text-2xl text-text-secondary mb-12 max-w-4xl mx-auto leading-relaxed font-light">
        AI-powered interior design platform: room measurement, 3D furniture placement, and house search all in one.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 max-w-5xl mx-auto">
        <div className="group bg-surface rounded-2xl shadow-2xl p-8 transform transition-all duration-500 hover:scale-105 hover:shadow-primary/20 hover:shadow-2xl border border-border hover:border-primary/50 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary to-accent"></div>
          <h2 className="text-2xl font-bold text-text-primary mb-6 group-hover:text-primary transition-colors duration-300">Find Your Dream Home</h2>
          <p className="text-text-secondary mb-8 leading-relaxed">
            Find houses that perfectly match your room structure and interior preferences.
          </p>
          <Link
            to="/find-house"
            className="inline-block bg-gradient-to-r from-primary to-secondary text-white px-8 py-4 rounded-xl font-semibold hover:shadow-lg hover:shadow-primary/30 transition-all duration-300 transform hover:-translate-y-1"
          >
            Find Your Home
          </Link>
        </div>

        <div className="group bg-surface rounded-2xl shadow-2xl p-8 transform transition-all duration-500 hover:scale-105 hover:shadow-primary/20 hover:shadow-2xl border border-border hover:border-primary/50 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary to-accent"></div>
          <h2 className="text-2xl font-bold text-text-primary mb-6 group-hover:text-primary transition-colors duration-300">2D/3D Room Planner</h2>
          <p className="text-text-secondary mb-8 leading-relaxed">
            Measure room dimensions from a single photo and create 2D floor plans with 3D furniture placement.
          </p>
          <Link
            to="/room-planner"
            className="inline-block bg-gradient-to-r from-primary to-secondary text-white px-8 py-4 rounded-xl font-semibold hover:shadow-lg hover:shadow-primary/30 transition-all duration-300 transform hover:-translate-y-1"
          >
            Start Planning
          </Link>
        </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;