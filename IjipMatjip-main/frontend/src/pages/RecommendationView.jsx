import { useState, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import NeighborhoodCard from '../components/NeighborhoodCard';

function RecommendationView() {
  const location = useLocation();
  const navigate = useNavigate();
  const searchConditions = location.state?.conditions;

  const [recommendations, setRecommendations] = useState([]);
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDong, setSelectedDong] = useState(null);

  useEffect(() => {
    if (!searchConditions) {
      alert("검색 조건이 없습니다. 이전 페이지로 돌아갑니다.");
      navigate('/');
      return;
    }

    const fetchAllRecommendations = async () => {
      setLoading(true);
      try {
        const payload = {
          preferences: searchConditions.lifestyle,
          region: searchConditions.region,
          budget: searchConditions.budget,
          size_pyeong: searchConditions.size,
        };
        const response = await axios.post('/api/recommend/neighborhood', payload);
        const { neighborhoods: initialNeighborhoods, properties: initialProperties } = response.data;
        
        let finalNeighborhoods = initialNeighborhoods || [];
        let finalProperties = initialProperties || [];

        // --- 👇 출퇴근 시간 계산 로직 수정 ---
        if (searchConditions.commute?.address && (initialNeighborhoods?.length > 0 || initialProperties?.length > 0)) {
          const geocodeResponse = await axios.post('/api/geocode', { address: searchConditions.commute.address });
          const workCoords = geocodeResponse.data;

          // 동네 출퇴근 시간 계산
          if (initialNeighborhoods?.length > 0) {
            const neighborhoodCommutePromises = initialNeighborhoods.map(dong => 
              axios.post('/api/directions', {
                origin: { lat: dong.latitude, lng: dong.longitude },
                destination: workCoords
              })
            );
            const neighborhoodCommuteResults = await Promise.all(neighborhoodCommutePromises);
            finalNeighborhoods = initialNeighborhoods.map((dong, index) => ({
              ...dong,
              commute_minutes: neighborhoodCommuteResults[index].data.duration_minutes
            }));
          }

          // 매물 출퇴근 시간 계산
          if (initialProperties?.length > 0) {
            const propertyCommutePromises = initialProperties.map(prop => 
              axios.post('/api/directions', {
                origin: { lat: prop.latitude, lng: prop.longitude },
                destination: workCoords
              })
            );
            const propertyCommuteResults = await Promise.all(propertyCommutePromises);
            finalProperties = initialProperties.map((prop, index) => ({
              ...prop,
              commute_minutes: propertyCommuteResults[index].data.duration_minutes
            }));
          }
        }
        console.log('state에 설정 전 최종 동네 데이터 :',finalNeighborhoods)
        setRecommendations(finalNeighborhoods);
        setProperties(finalProperties);

      } catch (error) {
        console.error('추천 데이터를 불러오는 데 실패했습니다.', error);
        alert(error.response?.data?.detail || '데이터를 불러올 수 없습니다.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchAllRecommendations();
  }, [searchConditions, navigate]);
  
  const handleNeighborhoodClick = (dongName) => {
    if (selectedDong === dongName) {
      setSelectedDong(null);
    } else {
      setSelectedDong(dongName);
    }
  };

  const filteredProperties = selectedDong
    ? properties.filter(prop => prop.dong_name === selectedDong)
    : properties;

  if (loading) {
    return <div className="text-center p-8">AI가 최적의 동네와 매물을 찾고 있습니다...</div>;
  }

  return (
    <div>
      <h2 className="text-2xl font-bold">2. AI 추천 결과입니다.</h2>
      <h3 className='my-4 text-xl font-semibold'>이런 동네는 어떠세요? (클릭하여 매물 필터링)</h3>
      <div className='flex gap-4 flex-wrap'>
        {recommendations.map((dong, index) => (
          <NeighborhoodCard
            key={index}
            dongData={dong}
            onCardClick={() => handleNeighborhoodClick(dong.dong)}
            isSelected={selectedDong === dong.dong}
          />
        ))}
      </div>

      <h3 className='mt-8 text-xl font-bold'>
        맞춤 매물 리스트 {selectedDong && `(${selectedDong})`}
      </h3>
      <div className='flex flex-col gap-4 mt-4'>
        {filteredProperties.map((prop, index) => (
          <Link to={`/detail/${index}`} state={{ propertyData: prop, conditions: searchConditions }} key={index} className='no-underline text-black'>
            <div className='flex items-center gap-4 p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow'>
              <div className='w-24 h-24 bg-gray-200 rounded flex items-center justify-center text-gray-500 font-bold text-lg'>?</div>
              <div className='flex flex-col justify-center'>
                <p className='font-semibold text-lg'>{prop.dong_name} {prop.name}</p>
                <p className='text-md text-gray-600'>{prop.size_m2 ? `${Math.round(prop.size_m2 / 3.3)}평` : ''}, {prop.floor}층</p>
                <p className='text-blue-600 font-bold text-xl mt-1'>
                  가격: {prop.price ? `${(prop.price / 10000).toFixed(1)}억 원` : '정보 없음'}
                </p>
                {prop.commute_minutes && (
                  <p className="text-sm font-medium text-gray-700 mt-1">
                    출퇴근: 🚇 약 {prop.commute_minutes}분
                  </p>
                )}
              </div>
            </div>
          </Link>
        ))}
        {filteredProperties.length === 0 && <p>해당 조건에 맞는 매물이 없습니다.</p>}
      </div>
    </div>
  );
}

export default RecommendationView;