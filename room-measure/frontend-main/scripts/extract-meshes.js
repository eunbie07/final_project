import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

// GLB 파일에서 개별 메시를 추출하여 별도 파일로 저장하는 스크립트
const extractMeshesToSeparateFiles = async () => {
  const loader = new GLTFLoader();
  const exporter = new GLTFExporter();
  
  try {
    console.log('GLB 파일 로딩 중...');
    const gltf = await new Promise((resolve, reject) => {
      loader.load('/low_poly_furnitures_full_bundle.glb', resolve, undefined, reject);
    });
    
    console.log('=== 메시 추출 시작 ===');
    const meshes = [];
    
    // 모든 메시 수집
    gltf.scene.traverse((child) => {
      if (child.isMesh) {
        meshes.push({
          name: child.name,
          mesh: child.clone()
        });
      }
    });
    
    console.log(`총 ${meshes.length}개의 메시를 찾았습니다.`);
    
    // 각 메시를 개별적으로 처리
    for (let i = 0; i < meshes.length; i++) {
      const { name, mesh } = meshes[i];
      
      try {
        // 새로운 씬 생성
        const scene = new THREE.Scene();
        
        // 메시를 씬에 추가 (위치를 원점으로 초기화)
        mesh.position.set(0, 0, 0);
        scene.add(mesh);
        
        // GLB로 내보내기
        const result = await new Promise((resolve, reject) => {
          exporter.parse(
            scene,
            (gltf) => resolve(gltf),
            { binary: true }, // GLB 형식으로 내보내기
            (error) => reject(error)
          );
        });
        
        // Blob 생성 및 다운로드
        const blob = new Blob([result], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `${name}.glb`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
        
        console.log(`✅ ${name}.glb 파일 다운로드 완료`);
        
        // 브라우저 과부하 방지를 위한 지연
        await new Promise(resolve => setTimeout(resolve, 500));
        
      } catch (error) {
        console.error(`❌ ${name} 메시 내보내기 실패:`, error);
      }
    }
    
    console.log('=== 모든 메시 추출 완료 ===');
    console.log('다운로드된 파일들을 /public/models/ 폴더에 복사하세요.');
    
    // 가구 매핑 가이드 생성
    console.log('\n=== 가구 매핑 가이드 ===');
    console.log('각 다운로드된 GLB 파일을 열어서 어떤 가구인지 확인 후,');
    console.log('아래와 같이 파일명을 변경하세요:');
    console.log('');
    console.log('침대 모양의 파일 → bed.glb');
    console.log('의자 모양의 파일 → chair.glb'); 
    console.log('책상 모양의 파일 → desk.glb');
    console.log('소파 모양의 파일 → sofa.glb');
    console.log('테이블 모양의 파일 → table.glb');
    console.log('옷장 모양의 파일 → wardrobe.glb');
    
  } catch (error) {
    console.error('GLB 파일 처리 실패:', error);
  }
};

// 선택적으로 특정 메시들만 추출하는 함수
const extractSpecificMeshes = async (meshNames) => {
  const loader = new GLTFLoader();
  const exporter = new GLTFExporter(); 
  
  try {
    const gltf = await new Promise((resolve, reject) => {
      loader.load('/low_poly_furnitures_full_bundle.glb', resolve, undefined, reject);
    });
    
    for (const meshName of meshNames) {
      const mesh = gltf.scene.getObjectByName(meshName);
      
      if (mesh && mesh.isMesh) {
        const scene = new THREE.Scene();
        const clonedMesh = mesh.clone();
        clonedMesh.position.set(0, 0, 0);
        scene.add(clonedMesh);
        
        const result = await new Promise((resolve, reject) => {
          exporter.parse(scene, resolve, { binary: true }, reject);
        });
        
        const blob = new Blob([result], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `${meshName}.glb`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
        console.log(`✅ ${meshName}.glb 다운로드 완료`);
      } else {
        console.warn(`⚠️ ${meshName} 메시를 찾을 수 없습니다`);
      }
    }
  } catch (error) {
    console.error('특정 메시 추출 실패:', error);
  }
};

// 브라우저에서 사용할 수 있도록 전역 함수로 등록
if (typeof window !== 'undefined') {
  window.extractMeshesToSeparateFiles = extractMeshesToSeparateFiles;
  window.extractSpecificMeshes = extractSpecificMeshes;
  
  console.log('🚀 GLB 메시 추출 도구가 준비되었습니다!');
  console.log('');
  console.log('사용법:');
  console.log('1. 모든 메시 추출: extractMeshesToSeparateFiles()');
  console.log('2. 특정 메시만 추출: extractSpecificMeshes(["Object_8", "Object_20", "Object_6"])');
  console.log('');
  console.log('⚠️ 주의: 33개 파일이 모두 다운로드됩니다. 팝업 차단을 해제해주세요.');
}

export { extractMeshesToSeparateFiles, extractSpecificMeshes };