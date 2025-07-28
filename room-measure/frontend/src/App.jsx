import React from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import RoomPlannerPage from "./pages/RoomPlannerPage";
import AIDesignPage from "./pages/AIDesignPage";
import FindHousePage from "./pages/FindHousePage";

function App() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="pt-16">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/room-planner" element={<RoomPlannerPage />} />
          <Route path="/ai-design" element={<AIDesignPage />} />
          <Route path="/find-house" element={<FindHousePage />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;