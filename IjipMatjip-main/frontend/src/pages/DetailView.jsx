import { useState, useEffect } from 'react'
import {useLocation, Link} from 'react-router-dom'
import axios from 'axios'
import InfrastructureMap from '../components/InfrastructureMap'


function DetailView() {
  const location = useLocation()
  // todo 
  // const {propertyId} = useParams() 확장용
  const propertyData = location.state?.propertyData;
  const searchConditions = location.state?.conditions;

  const [loading, setLoading] = useState(true)
  const [commuteTime, setCommuteTime] = useState(null);
  // 인프라 데이터와 필터 상태 관리
  const [infrastructure, setInfrastructure] = useState({schools:[], subways:[], hospitals:[], marts:[], parks:[]})
  const [infraFilters, setInfraFilters] = useState({
    school: true,
    subway: false,
    hospital: false,
    mart: false,
    park: false,
  })

  useEffect(() => {
    if(!propertyData?.latitude){
      setLoading(false);
      return;
    }

    const fetchDetails = async () => {
      setLoading(true);
      try{
        // 여러 API 호출을 병렬로 처리
        const promises = [];

        // 1. 출퇴근 시간 계산 (회사 주소가 있을 경우)
        if (searchConditions?.commute?.address){
          const geocodePromise = axios.post('/api/geocode',{
            address: searchConditions.commute.address
          })
          promises.push(geocodePromise)
        }

        // 2. 주변 인프라 정보 요청
        const infraPromise = axios.post('/api/infrastructure', {
          latitude: propertyData.latitude,
          longitude: propertyData.longitude,
          radius_km:0.5
        });
        promises.push(infraPromise)

        // 병렬 호출 실행
        const results = await Promise.all(promises);

        let resultIndex = 0;
        if(searchConditions?.commute?.address){
          const workCoords = results[resultIndex].data;
          const directionsResponse = await axios.post('/api/directions',{
            origin:{lat:propertyData.latitude, lng:propertyData.longitude},
            destination: workCoords
          })
          setCommuteTime(directionsResponse.data.duration_minutes);
          resultIndex++;
        }

        setInfrastructure(results[resultIndex].data);
      } catch(error){
        console.error("상세 정보 로딩 실패 : ",error);
      } finally {
        setLoading(false);
      }
    };
    fetchDetails();
  },[propertyData,searchConditions]);

  const handleFilterChange = (e) => {
    const {name, checked} = e.target;
    setInfraFilters(prev => ({...prev, [name]:checked}));
  };

  if(!propertyData){
    return (
      <div>
        <p>매물 정보가 없습니다.</p>
        <Link to="/">처음으로 돌아가기</Link>
      </div>
    );
  }

  // 필터링된 마커 목록 생성
  const getFilteredMarkers = () => {
    if(!infrastructure) return [];

    let markers = [];
    if(infraFilters.school) markers = markers.concat(infrastructure.schools.map(item => ({...item,type:'school'})));
    if(infraFilters.subway) markers = markers.concat(infrastructure.subways.map(item => ({...item, type:'subway'})))
    if(infraFilters.hospital) markers = markers.concat(infrastructure.hospitals.map(item => ({...item, type:'hospital'})))
    if(infraFilters.mart) markers = markers.concat(infrastructure.marts.map(item => ({...item, type:'mart'})))
    if(infraFilters.park) markers = markers.concat(infrastructure.parks.map(item => ({...item, type:'park'})))
    return markers;
  }

  return (
    <div>
      <Link to='/recommend' state={{conditions:searchConditions}} className='text-blue-500 hover:underline'>
        &larr; 목록으로 돌아가기
      </Link>
      <div className='mt-4 grid md:grid-cols-2 gap-8'>
        {/* 좌측: 지도분석 */}
        <div>
          <h3 className='text-xl font-bold mb-2'>지도 분석</h3>
          <div className='flex gap-4 mb-2 flex-wrap'>
            <label>
              <input type='checkbox' name='school' checked={infraFilters.school} onChange={handleFilterChange}/>
              학교
            </label>
            <label>
              <input type='checkbox' name='subway' checked={infraFilters.subway} onChange={handleFilterChange}/>
              지하철
            </label>
            <label>
              <input type='checkbox' name='hospital' checked={infraFilters.hospital} onChange={handleFilterChange}/>
              병원
            </label>
            <label>
              <input type='checkbox' name='mart' checked={infraFilters.mart} onChange={handleFilterChange}/>
              마트
            </label>
            <label>
              <input type='checkbox' name='park' checked={infraFilters.park} onChange={handleFilterChange}/>
              공원
            </label>
          </div>
          <div className='border rounded-lg overflow-hidden'>
            <InfrastructureMap
              lat={propertyData.latitude}
              lng={propertyData.longitude}
              isPropertyMarker={true}
              markers={getFilteredMarkers()}
            />
          </div>
        </div>

        {/* 우측: 상세정보 */}
        <div className='space-y-6'>
          <div>
            <h3 className='text-2xl font-bold'>{propertyData.name}</h3>
            <p className='text-gray-600'>{Math.round(propertyData.size_m2 /3.3)}평, {propertyData.floor}층</p>
          </div>
          <div className='p-4 border rounded-lg'>
            <p className='text-sm text-gray-500'>금액</p>
            <p className='text-3xl font-bold text-blue-600'>
              {propertyData.predicted_price ? `${(propertyData.predicted_price / 10000).toFixed(1)}억 원`: '정보 없음'}
            </p>
          </div>
          <div className='p-4 border rounded-lg'>
            <h4 className='font-bold mb-2'>주변 인프라 (반경 500m)</h4>
            <ul className='space-y-1 text-sm'>
              <li>🏫 학교: {infrastructure.schools.length}개</li>
              <li>🚇 지하철: {infrastructure.subways.length}개</li>
              <li>🏥 병원: {infrastructure.hospitals.length}개</li>
              <li>🛒 마트: {infrastructure.marts.length}개</li>
              <li>🌳 공원: {infrastructure.parks.length}개</li>
            </ul>
          </div>
          <div className='p-4 border rounded-lg'>
            <h4 className='font-bold mb-2'>출퇴근 분석</h4>
            {loading ? <p className='text-sm'>계산 중...</p>:
            commuteTime ? <p className='text-sm'>🚇 대중교통: 약 {commuteTime}분</p> : <p className='text-sm'>회사 주소가 입력되지 않았습니다.</p>}
          </div>
        </div>
      </div>    
    </div>
  )
}

export default DetailView