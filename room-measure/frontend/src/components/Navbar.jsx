import React from 'react';

const Navbar = () => {
  return (
    <nav className="bg-surface shadow-lg py-6 px-6 fixed w-full z-50 border-b border-border">
      <div className="container mx-auto flex justify-between items-center">
        <a href="/" className="text-3xl font-black text-text-primary hover:text-primary transition-colors duration-200">이집맛집</a>
        <div className="space-x-8">
          <a href="/find-house" className="text-lg text-text-secondary hover:text-primary transition-colors duration-200 font-bold hover:scale-105 transform">Find Your Dream Home</a>
          <a href="/room-planner" className="text-lg text-text-secondary hover:text-primary transition-colors duration-200 font-bold hover:scale-105 transform">2D/3D Room Planner</a>
          <a href="/ai-design" className="text-lg text-text-secondary hover:text-primary transition-colors duration-200 font-bold hover:scale-105 transform">AI Interior Design</a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
