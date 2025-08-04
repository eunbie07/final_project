import { useState } from 'react';

export const useRoomState = (initialRoomSize) => {
  const [w, h, d] = initialRoomSize;
  
  // Furniture state
  const [furniture, setFurniture] = useState([]);
  const [selectedFurniture, setSelectedFurniture] = useState(null);
  const [placementMode, setPlacementMode] = useState(null);
  
  // UI state
  const [showSnapGrid, setShowSnapGrid] = useState(false);
  const [showFloorGrid, setShowFloorGrid] = useState(false);
  const [enableSnap, setEnableSnap] = useState(true);
  const [showCollisions, setShowCollisions] = useState(true);
  const [showWindows, setShowWindows] = useState(false);
  
  // Mode state
  const [activeView, setActiveView] = useState("조감도");
  const [walkthroughMode, setWalkthroughMode] = useState(false);
  const [measurementMode, setMeasurementMode] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  
  // Interaction state
  const [collisionAlert, setCollisionAlert] = useState({
    visible: false,
    collisions: [],
    timeout: null,
  });
  const [measurePoints, setMeasurePoints] = useState([null, null]);
  const [humanPosition, setHumanPosition] = useState([w / 2, 0, d / 2]);
  
  // Window detection state
  const [detectedWindows, setDetectedWindows] = useState([]);
  const [isDetectingWindows, setIsDetectingWindows] = useState(false);
  
  // Loading states
  const [isSaving, setIsSaving] = useState(false);
  
  return {
    // Furniture state
    furniture,
    setFurniture,
    selectedFurniture,
    setSelectedFurniture,
    placementMode,
    setPlacementMode,
    
    // UI state
    showSnapGrid,
    setShowSnapGrid,
    showFloorGrid,
    setShowFloorGrid,
    enableSnap,
    setEnableSnap,
    showCollisions,
    setShowCollisions,
    showWindows,
    setShowWindows,
    
    // Mode state
    activeView,
    setActiveView,
    walkthroughMode,
    setWalkthroughMode,
    measurementMode,
    setMeasurementMode,
    isDragging,
    setIsDragging,
    
    // Interaction state
    collisionAlert,
    setCollisionAlert,
    measurePoints,
    setMeasurePoints,
    humanPosition,
    setHumanPosition,
    
    // Window detection state
    detectedWindows,
    setDetectedWindows,
    isDetectingWindows,
    setIsDetectingWindows,
    
    // Loading states
    isSaving,
    setIsSaving,
  };
};