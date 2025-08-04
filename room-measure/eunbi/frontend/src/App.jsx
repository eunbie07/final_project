import React from "react";
import { Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import RoomPlannerPage from "./pages/RoomPlannerPage";
import AIDesignPage from "./pages/AIDesignPage";
import FindHousePage from "./pages/FindHousePage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-background">
        <Navbar />
        <div className="pt-16">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/room-planner" element={<RoomPlannerPage />} />
            <Route path="/ai-design" element={<AIDesignPage />} />
            <Route path="/find-house" element={<FindHousePage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
          </Routes>
        </div>
      </div>
    </AuthProvider>
  );
}

export default App;