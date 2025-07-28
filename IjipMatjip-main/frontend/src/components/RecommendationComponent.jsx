import { useState } from "react"
import axios from "axios"
const RecommendationComponent = ({onDongClick}) => {
  const [preferences, setPreferences] = useState({
    school:true,
    subway:false,
    price:false,
  })
  const [workAddress, setWorkAddress] = useState("서울특별시 서초구 서초대로74길 33")
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)

  const handleCheckboxChange = (e) => {
    const {name, checked} = e.target
    setPreferences(prev => ({...prev, [name]:checked}))
  }

  const handleRecommendClick = async() => {
    setLoading(true)
    setRecommendations([])

    try {
      // 1. 회사 주소 좌표 반환
      const geocodeResponse = await axios.post('/api/geocode', {address:workAddress})
      const workCoords = geocodeResponse.data;
      console.log('회사좌표',workCoords)

      // 2. 추천 동네 목록 받기
      const selectedPreferences = Object.keys(preferences).filter(key => preferences[key])
      const recommendResponse = await axios.post('/api/recommend/neighborhood', {
        preferences: selectedPreferences
      })
      const recommendedDongs = recommendResponse.data;
      console.log('추천 동네 목록:',recommendedDongs)

      // 3. 각 동네별 출퇴근 시간 계산 (병렬처리)
      const commutePromises = recommendedDongs.map(dong => {
        return axios.post('/api/directions', {
          origin: {lat: dong.latitude, lng: dong.longitude},
          destination:workCoords
        })
      })
      
      const commuteResults = await Promise.all(commutePromises)

      // 4. 추천 결과에 출퇴근 시간 정보 합침
      const finalRecommendations = recommendedDongs.map((dong,index) => ({...dong, commute_minutes: commuteResults[index].data.duration_minutes}))
      
      setRecommendations(finalRecommendations)

    } catch(error){
      console.error("추천 데이터를 불러오는 데 실패했습니다.", error)
      alert(error.response?.data?.detail || "데이터를 불러올 수 없습니다.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1>맞춤 동네 추천받기</h1>
      <div>
        <label><input type="checkbox" name="school" checked={preferences.school} onChange={handleCheckboxChange} />학군</label>
        <label className="ml-4"><input type="checkbox" name="subway" checked={preferences.subway} onChange={handleCheckboxChange} />역세권</label>
        <label className="ml-4"><input type="checkbox" name="price" checked={preferences.price} onChange={handleCheckboxChange} />가격</label>
      </div>
      <div className="mt-4">
        <label>회사 주소 :</label>
        <input type="text" value={workAddress} onChange={(e) => setWorkAddress(e.target.value)} placeholder="회사 주소를 입력하세요" className="w-xs border border-gray-300 p-1"/>
      </div>
      <button onClick={handleRecommendClick} disabled={loading} className="mt-4">{loading ? '추천 중...':'추천받기'}</button>
      <h2>추천 동네 목록</h2>
      <ul>
        {recommendations.map((dong, index) => (
          <li key={index} onClick={() => onDongClick({lat:dong.latitude, lng: dong.longitude})} className="cursor-pointer mb-2.5">
            <strong>{dong.dong}</strong> (총점: {dong.total_score.toFixed(2)})
            {dong.commute_minutes && <span>| <strong>출퇴근 : 약 {dong.commute_minutes}분</strong></span>}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default RecommendationComponent
