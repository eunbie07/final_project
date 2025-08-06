import React, { useState, useCallback } from 'react';

// 가구 기본 정보
const furnitureData = {
    double_bed: { type: "furniture", dimensions: { width: 1600, depth: 2100, height: 1000 }, material: "dark wood frame with grey linen bedding" },
    wardrobe: { type: "furniture", dimensions: { width: 600, depth: 1800, height: 2200 }, material: "white matte finish" },
    desk: { type: "furniture", dimensions: { width: 600, depth: 1200, height: 750 }, material: "light oak wood with black metal legs" },
    main_window: { type: "window", dimensions: { width: 50, depth: 2500, height: 1500 }, details: "covered with sheer white curtains" },
    entrance_door: { type: "door", dimensions: { width: 900, depth: 50, height: 2100 }, material: "white painted wood" }
};

// 스타일 컴포넌트
const GlobalStyles = () => (
    <style>{`
        body { font-family: 'Inter', sans-serif; }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `}</style>
);

// 메인 앱 컴포넌트
export default function App() {
    // 상태 관리
    const [roomDims, setRoomDims] = useState({ width: 4000, depth: 5000, height: 2800 });
    const [selectedFurniture, setSelectedFurniture] = useState({
        double_bed: true,
        wardrobe: true,
        desk: true,
        main_window: true,
        entrance_door: true
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [resultImage, setResultImage] = useState('');
    const [generatedJson, setGeneratedJson] = useState('');

    // 핸들러 함수들
    const handleFurnitureChange = (event) => {
        const { name, checked } = event.target;
        setSelectedFurniture(prev => ({ ...prev, [name]: checked }));
    };
    const handleDimChange = (event) => {
        const { name, value } = event.target;
        setRoomDims(prev => ({ ...prev, [name]: parseInt(value, 10) || 0 }));
    };

    // JSON 데이터 생성 로직
    const buildRoomJson = useCallback(() => {
        const objects = Object.keys(selectedFurniture)
            .filter(key => selectedFurniture[key])
            .map(name => {
                const data = furnitureData[name];
                let obj = {
                    type: data.type,
                    name: name,
                    dimensions: data.dimensions,
                    position: {},
                    rotation_z: 0,
                    ...(data.material && { material: data.material }),
                    ...(data.details && { details: data.details })
                };
                switch (name) {
                    case 'double_bed':
                        obj.position = { x: (roomDims.width / 2) - (data.dimensions.width / 2), y: roomDims.depth, z: 0 };
                        obj.rotation_z = 180;
                        break;
                    case 'wardrobe':
                        obj.position = { x: 0, y: roomDims.depth / 2, z: 0 };
                        break;
                    case 'desk':
                        obj.position = { x: roomDims.width, y: (roomDims.depth / 2) + (data.dimensions.depth / 2), z: 0 };
                        break;
                    case 'main_window':
                        obj.position = { x: roomDims.width, y: roomDims.depth / 2, z: 1000 };
                        break;
                    case 'entrance_door':
                        obj.position = { x: data.dimensions.width / 2, y: 0, z: 0 };
                        break;
                    default:
                        obj.position = { x: 0, y: 0, z: 0 };
                }
                return obj;
            });

        return {
            scene: {
                description: "A 3D scene definition for a room. All units are in millimeters. X-axis is width, Y-axis is depth, Z-axis is height.",
                view_point: {
                    camera_position: { x: roomDims.width / 2, y: -1000, z: roomDims.height * 0.6 },
                    look_at: { x: roomDims.width / 2, y: roomDims.depth / 2, z: roomDims.height * 0.4 },
                    style: "cozy, modern, bright, photorealistic, wide-angle lens"
                },
                room: {
                    dimensions: roomDims,
                    walls: { material: "light beige painted plaster" },
                    floor: { material: "light oak wood planks" }
                },
                objects: objects
            }
        };
    }, [roomDims, selectedFurniture]);

    // 이미지 생성 버튼 클릭 핸들러
    const handleGenerate = async () => {
        setIsLoading(true);
        setError('');
        setResultImage('');
        
        const roomJson = buildRoomJson();
        setGeneratedJson(JSON.stringify(roomJson, null, 2));

        try {
            // FastAPI 백엔드 서버에 요청
            const response = await fetch('http://127.0.0.1:8000/generate-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(roomJson)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '이미지 생성에 실패했습니다.');
            }
            
            setResultImage(data.imageUrl);

        } catch (err) {
            setError(`오류 발생: ${err.message}`);
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            <GlobalStyles />
            <div className="bg-gray-100 text-gray-800">
                <div className="container mx-auto p-4 md:p-8 max-w-4xl">
                    <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
                        <h1 className="text-3xl font-bold mb-2 text-gray-900">AI 방 꾸미기 (React + FastAPI)</h1>
                        <p className="text-gray-600 mb-6">방 크기와 가구를 선택하면 AI가 이미지를 생성해줍니다.</p>
                        
                        {/* API 키 입력 필드 제거됨 */}

                        <div className="grid md:grid-cols-2 gap-6">
                            {/* 방 크기 설정 */}
                             <div>
                                <h2 className="text-xl font-semibold mb-3">1. 방 크기 설정 (mm)</h2>
                                <div className="space-y-3">
                                    <div>
                                        <label htmlFor="roomWidth" className="text-sm font-medium">가로 (Width)</label>
                                        <input type="number" id="roomWidth" name="width" value={roomDims.width} onChange={handleDimChange} className="mt-1 w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg" />
                                    </div>
                                    <div>
                                        <label htmlFor="roomDepth" className="text-sm font-medium">세로 (Depth)</label>
                                        <input type="number" id="roomDepth" name="depth" value={roomDims.depth} onChange={handleDimChange} className="mt-1 w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg" />
                                    </div>
                                    <div>
                                        <label htmlFor="roomHeight" className="text-sm font-medium">높이 (Height)</label>
                                        <input type="number" id="roomHeight" name="height" value={roomDims.height} onChange={handleDimChange} className="mt-1 w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg" />
                                    </div>
                                </div>
                            </div>
                            {/* 가구 선택 */}
                            <div>
                                <h2 className="text-xl font-semibold mb-3">2. 가구 선택</h2>
                                <div className="space-y-2">
                                    {Object.keys(furnitureData).map(name => (
                                        <div key={name} className="flex items-center">
                                            <input type="checkbox" id={name} name={name} checked={selectedFurniture[name]} onChange={handleFurnitureChange} className="h-4 w-4 rounded" />
                                            <label htmlFor={name} className="ml-2">{name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</label>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="mt-8">
                            <button onClick={handleGenerate} disabled={isLoading} className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors duration-300 disabled:bg-blue-300">
                                {isLoading ? '생성 중...' : '이미지 생성하기'}
                            </button>
                        </div>
                        
                        {(isLoading || error || resultImage || generatedJson) && (
                             <div className="mt-8">
                                <h2 className="text-xl font-semibold mb-4">생성 결과</h2>
                                {isLoading && <div className="loader mx-auto"></div>}
                                {error && <div className="mt-4 text-red-600">{error}</div>}
                                {resultImage && (
                                    <div className="w-full aspect-video bg-gray-200 rounded-lg overflow-hidden flex items-center justify-center mt-4">
                                        <img src={resultImage} alt="생성된 이미지" className="w-full h-full object-contain" />
                                    </div>
                                )}
                                {generatedJson && (
                                    <>
                                        <h3 className="text-lg font-semibold mt-6 mb-2">전송된 데이터 (JSON)</h3>
                                        <pre className="bg-gray-900 text-white text-xs p-4 rounded-lg overflow-x-auto">{generatedJson}</pre>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}
