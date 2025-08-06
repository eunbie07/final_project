"""
실시간 좌표 동기화 및 검증 시스템
RoomBox.jsx와 Dify 간의 실시간 좌표 일관성 보장
"""

import asyncio
import json
import websockets
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import hashlib
from roombox_integration import RoomBoxDataProcessor, DifyRoomImageGenerator
from dify_rag import DifyLayoutRAG


class RealtimeCoordinateSync:
    """실시간 좌표 동기화 시스템"""
    
    def __init__(self, dify_api_key: str, dify_app_id: str, dify_dataset_id: str = None):
        self.dify_generator = DifyRoomImageGenerator(dify_api_key, dify_app_id, dify_dataset_id)
        self.active_sessions = {}  # session_id -> session_data
        self.change_callbacks = []  # 좌표 변경 콜백 함수들
        
        # 좌표 변경 감지를 위한 해시 추적
        self.layout_hashes = {}  # session_id -> hash
        
    def register_change_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """좌표 변경 감지 콜백 등록"""
        self.change_callbacks.append(callback)
    
    async def start_session(self, session_id: str, initial_room_data: Dict[str, Any]) -> Dict[str, Any]:
        """새 세션 시작"""
        
        try:
            # 세션 데이터 초기화
            self.active_sessions[session_id] = {
                "room_data": initial_room_data,
                "last_update": datetime.now(),
                "validation_errors": [],
                "auto_corrections": {},
                "generation_history": []
            }
            
            # 초기 좌표 검증
            validation_result = await self._validate_and_correct_coordinates(session_id, initial_room_data)
            
            # 해시 생성 및 저장
            layout_hash = self._generate_layout_hash(initial_room_data)
            self.layout_hashes[session_id] = layout_hash
            
            return {
                "success": True,
                "session_id": session_id,
                "initial_validation": validation_result,
                "layout_hash": layout_hash,
                "message": "세션이 성공적으로 시작되었습니다"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "세션 시작 실패"
            }
    
    async def update_coordinates(self, session_id: str, updated_room_data: Dict[str, Any]) -> Dict[str, Any]:
        """좌표 업데이트 처리"""
        
        if session_id not in self.active_sessions:
            return {
                "success": False,
                "error": "Invalid session",
                "message": "유효하지 않은 세션입니다"
            }
        
        try:
            # 변경 감지
            new_hash = self._generate_layout_hash(updated_room_data)
            old_hash = self.layout_hashes.get(session_id)
            
            if new_hash == old_hash:
                return {
                    "success": True,
                    "changed": False,
                    "message": "좌표 변경 없음"
                }
            
            # 좌표 검증 및 보정
            validation_result = await self._validate_and_correct_coordinates(session_id, updated_room_data)
            
            # 세션 데이터 업데이트
            session_data = self.active_sessions[session_id]
            session_data["room_data"] = updated_room_data
            session_data["last_update"] = datetime.now()
            session_data["validation_errors"] = validation_result.get("errors", [])
            session_data["auto_corrections"] = validation_result.get("corrections", {})
            
            # 해시 업데이트
            self.layout_hashes[session_id] = new_hash
            
            # 변경 콜백 실행
            for callback in self.change_callbacks:
                try:
                    await callback(session_id, updated_room_data)
                except Exception as e:
                    print(f"콜백 실행 오류: {e}")
            
            return {
                "success": True,
                "changed": True,
                "validation": validation_result,
                "layout_hash": new_hash,
                "message": "좌표가 성공적으로 업데이트되었습니다"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "좌표 업데이트 실패"
            }
    
    async def generate_consistent_image(self, session_id: str, style: str = "modern") -> Dict[str, Any]:
        """일관성 있는 이미지 생성"""
        
        if session_id not in self.active_sessions:
            return {
                "success": False,
                "error": "Invalid session"
            }
        
        session_data = self.active_sessions[session_id]
        room_data = session_data["room_data"]
        
        try:
            # Dify 최적화된 이미지 생성
            result = await self.dify_generator.generate_consistent_room_image(
                room_data, style, session_id
            )
            
            # 생성 기록 저장
            generation_record = {
                "timestamp": datetime.now().isoformat(),
                "style": style,
                "layout_hash": self.layout_hashes[session_id],
                "result": result,
                "coordinates_snapshot": self._extract_coordinates_summary(room_data)
            }
            
            session_data["generation_history"].append(generation_record)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_and_correct_coordinates(self, session_id: str, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """좌표 검증 및 자동 보정"""
        
        try:
            # RoomBox 데이터 파싱
            room_layout = self.dify_generator.data_processor.parse_roombox_data(room_data)
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "corrections": {},
                "furniture_validations": []
            }
            
            # 각 가구별 검증
            for furniture in room_layout.furniture:
                furniture_validation = self.dify_generator.data_processor.coordinate_validator.validate_furniture_position(
                    furniture, room_layout
                )
                
                validation_result["furniture_validations"].append({
                    "furniture": furniture.name,
                    "validation": furniture_validation
                })
                
                if not furniture_validation["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(furniture_validation["errors"])
                
                validation_result["warnings"].extend(furniture_validation["warnings"])
                
                if furniture_validation.get("corrections"):
                    validation_result["corrections"][furniture.name] = furniture_validation["corrections"]
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"검증 프로세스 오류: {str(e)}"],
                "warnings": [],
                "corrections": {}
            }
    
    def _generate_layout_hash(self, room_data: Dict[str, Any]) -> str:
        """레이아웃 해시 생성 (변경 감지용)"""
        
        # 좌표 관련 데이터만 추출하여 해시 생성
        coordinate_data = {
            "room": room_data.get("scene", {}).get("room", {}),
            "objects": []
        }
        
        for obj in room_data.get("scene", {}).get("objects", []):
            if obj.get("type") == "furniture":
                coordinate_data["objects"].append({
                    "name": obj.get("name"),
                    "position": obj.get("position"),
                    "dimensions": obj.get("dimensions"),
                    "rotation": obj.get("rotation", obj.get("rotation_z", 0))
                })
        
        # JSON 직렬화 후 해시 생성
        json_str = json.dumps(coordinate_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _extract_coordinates_summary(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """좌표 요약 정보 추출"""
        
        summary = {
            "room_size": room_data.get("scene", {}).get("room", {}),
            "furniture_count": 0,
            "furniture_positions": []
        }
        
        for obj in room_data.get("scene", {}).get("objects", []):
            if obj.get("type") == "furniture":
                summary["furniture_count"] += 1
                summary["furniture_positions"].append({
                    "name": obj.get("name"),
                    "position": obj.get("position"),
                    "dimensions": obj.get("dimensions")
                })
        
        return summary
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        
        if session_id not in self.active_sessions:
            return {
                "exists": False,
                "message": "세션을 찾을 수 없습니다"
            }
        
        session_data = self.active_sessions[session_id]
        
        return {
            "exists": True,
            "session_id": session_id,
            "last_update": session_data["last_update"].isoformat(),
            "validation_errors": session_data["validation_errors"],
            "auto_corrections": session_data["auto_corrections"],
            "generation_count": len(session_data["generation_history"]),
            "current_hash": self.layout_hashes.get(session_id),
            "coordinates_summary": self._extract_coordinates_summary(session_data["room_data"])
        }
    
    async def close_session(self, session_id: str) -> Dict[str, Any]:
        """세션 종료"""
        
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
        if session_id in self.layout_hashes:
            del self.layout_hashes[session_id]
        
        return {
            "success": True,
            "message": f"세션 {session_id}이 종료되었습니다"
        }


class WebSocketCoordinateServer:
    """WebSocket 기반 실시간 좌표 동기화 서버"""
    
    def __init__(self, sync_system: RealtimeCoordinateSync, host: str = "localhost", port: int = 8765):
        self.sync_system = sync_system
        self.host = host
        self.port = port
        self.clients = {}  # websocket -> session_id
    
    async def register_client(self, websocket, path):
        """클라이언트 등록 및 메시지 처리"""
        
        session_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                command = data.get("command")
                
                if command == "start_session":
                    session_id = data.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    room_data = data.get("room_data")
                    
                    result = await self.sync_system.start_session(session_id, room_data)
                    self.clients[websocket] = session_id
                    
                    await websocket.send(json.dumps({
                        "type": "session_started",
                        "result": result
                    }))
                
                elif command == "update_coordinates":
                    if websocket in self.clients:
                        session_id = self.clients[websocket]
                        room_data = data.get("room_data")
                        
                        result = await self.sync_system.update_coordinates(session_id, room_data)
                        
                        await websocket.send(json.dumps({
                            "type": "coordinates_updated",
                            "result": result
                        }))
                
                elif command == "generate_image":
                    if websocket in self.clients:
                        session_id = self.clients[websocket]
                        style = data.get("style", "modern")
                        
                        result = await self.sync_system.generate_consistent_image(session_id, style)
                        
                        await websocket.send(json.dumps({
                            "type": "image_generated",
                            "result": result
                        }))
                
                elif command == "get_status":
                    if websocket in self.clients:
                        session_id = self.clients[websocket]
                        status = await self.sync_system.get_session_status(session_id)
                        
                        await websocket.send(json.dumps({
                            "type": "session_status",
                            "status": status
                        }))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))
        finally:
            # 클라이언트 연결 해제 시 세션 정리
            if websocket in self.clients:
                session_id = self.clients[websocket]
                await self.sync_system.close_session(session_id)
                del self.clients[websocket]
    
    async def start_server(self):
        """WebSocket 서버 시작"""
        print(f"🚀 실시간 좌표 동기화 서버 시작: ws://{self.host}:{self.port}")
        
        async with websockets.serve(self.register_client, self.host, self.port):
            print("⚡ 서버가 클라이언트 연결을 기다리고 있습니다...")
            await asyncio.Future()  # 무한 대기


# 사용 예시
async def main():
    """메인 실행 함수"""
    # 환경 변수에서 API 키 로드 (실제 사용 시)
    import os
    
    DIFY_API_KEY = os.getenv("DIFY_API_KEY", "your-dify-api-key")
    DIFY_APP_ID = os.getenv("DIFY_APP_ID", "your-dify-app-id")
    DIFY_DATASET_ID = os.getenv("DIFY_DATASET_ID", "your-dify-dataset-id")
    
    # 실시간 동기화 시스템 초기화
    sync_system = RealtimeCoordinateSync(DIFY_API_KEY, DIFY_APP_ID, DIFY_DATASET_ID)
    
    # WebSocket 서버 초기화 및 시작
    ws_server = WebSocketCoordinateServer(sync_system)
    
    # 서버 실행
    await ws_server.start_server()


def cli_main():
    """CLI 진입점 (uv run sync-server)"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()