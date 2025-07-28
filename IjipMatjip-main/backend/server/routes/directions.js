import express from 'express'
import axios from 'axios'
import 'dotenv/config'

const directionsRouter = express.Router()

//카카오 길찾기 API URL
const KAKAO_DIRECTIONS_URL = 'https://apis-navi.kakaomobility.com/v1/directions'
const KAKAO_REST_API_KEY = process.env.KAKAO_REST_API_KEY

directionsRouter.post('/', async (req,res) => {
  //origin: 출발지, destination: 도착지
  const {origin, destination} = req.body
    // 👇 디버깅을 위해 받은 좌표를 그대로 출력해봅니다.
  // console.log('--- 길찾기 요청 수신 ---');
  // console.log('출발지 (origin):', origin);
  // console.log('도착지 (destination):', destination);

  if (!origin || !destination) {
    return res.status(400).json({detail:'출발지와 도착지 좌표가 모두 필요합니다.'})
  }

  try{
    //카카오는 경도, 위도 순으로 좌표를 받음
    const params = {
      origin:`${origin.lng},${origin.lat}`,
      destination:`${destination.lng},${destination.lat}`
    }

    // console.log("카카오 길 찾기 파라미터", params)

    const response = await axios.get(KAKAO_DIRECTIONS_URL, {
      params,
      headers:{
        Authorization: `KakaoAK ${KAKAO_REST_API_KEY}`,
      }
    })

    //경로가 하나 이상 있는 경우, 첫 번째 경로의 요약 정보 사용
    if (response.data.routes && response.data.routes.length > 0) {
      const durationInSeconds = response.data.routes[0].summary.duration
      const durationInMinutes = Math.round(durationInSeconds / 60)

      res.json({
        duration_seconds: durationInSeconds,
        duration_minutes: durationInMinutes
      })
    } else{
      res.status(404).json({detail: '해당 경로를 찾을 수 없습니다.'})
    }
  } catch (error){
    console.error('카카오 길찾기 API 통신 중 에러 발생 : ', error.message)
    if(error.response) {
      res.status(error.response.status).json(error.response.data)
    } else{
      res.status(500).json({detail: '길찾기 중 서버에서 에러가 발생했습니다.'})
    }
  }
})

export default directionsRouter