import React from "react";
import { Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import Navbar from "./components/Navbar";
import RoomPlannerPage from "./pages/RoomPlannerPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import AIInteriorPage from "./pages/AIInteriorPage";

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-background">
        <Navbar />
        <div className="pt-16">
          <Routes>
            <Route path="/" element={<RoomPlannerPage />} />
            <Route path="/ai-interior" element={<AIInteriorPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
          </Routes>
        </div>
      </div>
    </AuthProvider>
  );
}

export default App;