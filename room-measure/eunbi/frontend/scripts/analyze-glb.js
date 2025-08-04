import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// GLB 파일 분석 스크립트
const analyzeGLB = async () => {
  const loader = new GLTFLoader();
  
  try {
    console.log('GLB 파일 로딩 중...');
    const gltf = await new Promise((resolve, reject) => {
      loader.load('/low_poly_furnitures_full_bundle.glb', resolve, undefined, reject);
    });
    
    console.log('=== GLB 파일 분석 결과 ===');
    const meshNames = [];
    const meshDetails = [];
    
    gltf.scene.traverse((child) => {
      if (child.isMesh) {
        meshNames.push(child.name);
        meshDetails.push({
          name: child.name,
          geometry: child.geometry.type,
          materialCount: Array.isArray(child.material) ? child.material.length : 1,
          hasTexture: child.material && (
            child.material.map || 
            (Array.isArray(child.material) && child.material.some(m => m.map))
          )
        });
      }
    });
    
    console.log('총 메시 개수:', meshNames.length);
    console.log('\n=== 모든 메시 이름 목록 ===');
    meshNames.sort().forEach((name, index) => {
      console.log(`${index + 1}. ${name}`);
    });
    
    console.log('\n=== 가구별 추천 메시 매핑 ===');
    
    // 침대 관련
    const bedMeshes = meshNames.filter(name => 
      /bed|mattress|침대/i.test(name)
    );
    console.log('침대 (bed):', bedMeshes);
    
    // 책상 관련  
    const deskMeshes = meshNames.filter(name => 
      /desk|table|책상|테이블|bureau/i.test(name)
    );
    console.log('책상 (desk):', deskMeshes);
    
    // 의자 관련
    const chairMeshes = meshNames.filter(name => 
      /chair|seat|의자/i.test(name)
    );
    console.log('의자 (chair):', chairMeshes);
    
    // 소파 관련
    const sofaMeshes = meshNames.filter(name => 
      /sofa|couch|소파/i.test(name)
    );
    console.log('소파 (sofa):', sofaMeshes);
    
    // 옷장 관련
    const wardrobeMeshes = meshNames.filter(name => 
      /wardrobe|closet|cabinet|armoire|dresser|cupboard|storage|옷장/i.test(name)
    );
    console.log('옷장 (wardrobe):', wardrobeMeshes);
    
    console.log('\n=== 상세 정보 ===');
    meshDetails.forEach((detail, index) => {
      console.log(`${index + 1}. ${detail.name}:`);
      console.log(`   - 지오메트리: ${detail.geometry}`);
      console.log(`   - 재질 개수: ${detail.materialCount}`);
      console.log(`   - 텍스처 여부: ${detail.hasTexture ? '있음' : '없음'}`);
    });
    
    // furniture.js 업데이트용 코드 생성
    console.log('\n=== furniture.js 업데이트 코드 ===');
    console.log(`// 분석 결과를 바탕으로 meshNames 업데이트
export const FURNITURE_PRESETS = {
  bed: {
    name: "침대",
    size: [150, 60, 200],
    color: "#FFB6C1", 
    icon: "faBed",
    category: "bedroom",
    model3D: {
      path: "/low_poly_furnitures_full_bundle.glb",
      meshNames: ${JSON.stringify(bedMeshes.length > 0 ? bedMeshes : ['bed'])},
      scale: [1, 1, 1],
      rotation: [0, 0, 0]
    }
  },
  desk: {
    name: "책상",
    size: [120, 75, 60],
    color: "#98FB98",
    icon: "faTable", 
    category: "office",
    model3D: {
      path: "/low_poly_furnitures_full_bundle.glb",
      meshNames: ${JSON.stringify(deskMeshes.length > 0 ? deskMeshes : ['desk'])},
      scale: [1, 1, 1],
      rotation: [0, 0, 0]
    }
  },
  chair: {
    name: "의자",
    size: [50, 85, 50],
    color: "#90EE90",
    icon: "faChair",
    category: "office", 
    model3D: {
      path: "/low_poly_furnitures_full_bundle.glb",
      meshNames: ${JSON.stringify(chairMeshes.length > 0 ? chairMeshes : ['chair'])},
      scale: [1, 1, 1],
      rotation: [0, 0, 0]
    }
  },
  sofa: {
    name: "소파", 
    size: [180, 85, 80],
    color: "#87CEEB",
    icon: "faCouch",
    category: "living",
    model3D: {
      path: "/low_poly_furnitures_full_bundle.glb", 
      meshNames: ${JSON.stringify(sofaMeshes.length > 0 ? sofaMeshes : ['sofa'])},
      scale: [1, 1, 1],
      rotation: [0, 0, 0]
    }
  },
  wardrobe: {
    name: "옷장",
    size: [80, 200, 60], 
    color: "#DDA0DD",
    icon: "faBox",
    category: "storage",
    model3D: {
      path: "/low_poly_furnitures_full_bundle.glb",
      meshNames: ${JSON.stringify(wardrobeMeshes.length > 0 ? wardrobeMeshes : ['wardrobe'])},
      scale: [1, 1, 1], 
      rotation: [0, 0, 0]
    }
  }
};`);
    
  } catch (error) {
    console.error('GLB 파일 분석 실패:', error);
  }
};

// 브라우저에서 실행
if (typeof window !== 'undefined') {
  window.analyzeGLB = analyzeGLB;
  console.log('analyzeGLB() 함수를 실행하여 GLB 파일을 분석하세요.');
}

export default analyzeGLB;