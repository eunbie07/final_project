#!/usr/bin/env python3
"""
GLB 파일에서 가구를 분리하는 Blender 스크립트
사용법: blender --background --python separate_furniture.py
"""

import bpy
import os
import sys

# Blender에서 실행할 때만 작동
if bpy.app.binary_path:
    # 기존 모든 오브젝트 삭제
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # GLB 파일 경로
    input_file = "C:/Users/kibwa07/Documents/GitHub/final_project/room-measure/frontend/public/low_poly_furnitures_full_bundle.glb"
    output_dir = "C:/Users/kibwa07/Documents/GitHub/final_project/room-measure/frontend/public/furniture_models/"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # GLB 파일 로드
    try:
        bpy.ops.import_scene.gltf(filepath=input_file)
        print(f"Successfully loaded: {input_file}")
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        sys.exit(1)
    
    # 현재 씬의 모든 오브젝트 출력
    print("All objects in scene:")
    for obj in bpy.context.scene.objects:
        print(f"  - {obj.name} (type: {obj.type})")
    
    # 가구별 키워드 매핑 (대소문자 무시)
    furniture_keywords = {
        'bed': ['bed', 'mattress', '침대', 'cama'],
        'desk': ['desk', 'table', '책상', 'mesa', 'bureau'],
        'chair': ['chair', 'seat', '의자', 'silla', 'chaise'],
        'sofa': ['sofa', 'couch', '소파', 'settee'],
        'table': ['dining', 'coffee', '테이블', 'mesa'],
        'wardrobe': ['wardrobe', 'closet', 'cabinet', '옷장', 'armario']
    }
    
    # 각 가구별로 오브젝트 분리 및 저장
    for furniture_type, keywords in furniture_keywords.items():
        print(f"\n=== Processing {furniture_type} ===")
        
        # 모든 오브젝트 선택 해제
        bpy.ops.object.select_all(action='DESELECT')
        
        # 키워드에 맞는 오브젝트 찾기
        matching_objects = []
        for obj in bpy.context.scene.objects:
            obj_name_lower = obj.name.lower()
            for keyword in keywords:
                if keyword.lower() in obj_name_lower:
                    matching_objects.append(obj)
                    obj.select_set(True)
                    print(f"  Found: {obj.name}")
                    break
        
        if matching_objects:
            # 첫 번째 오브젝트를 액티브로 설정
            bpy.context.view_layer.objects.active = matching_objects[0]
            
            # 선택된 오브젝트들을 GLB로 내보내기
            output_path = os.path.join(output_dir, f"{furniture_type}.glb")
            try:
                bpy.ops.export_scene.gltf(
                    filepath=output_path,
                    use_selection=True,
                    export_format='GLB',
                    export_materials='EXPORT',
                    export_colors=True,
                    export_cameras=False,
                    export_lights=False
                )
                print(f"  Exported: {output_path}")
            except Exception as e:
                print(f"  Error exporting {furniture_type}: {e}")
        else:
            print(f"  No objects found for {furniture_type}")
        
        # 선택 해제
        bpy.ops.object.select_all(action='DESELECT')
    
    print("\n=== Furniture separation completed ===")

else:
    print("This script must be run from within Blender.")
    print("Usage: blender --background --python separate_furniture.py")