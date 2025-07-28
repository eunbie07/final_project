import express from 'express'
import axios from 'axios'
import 'dotenv/config'

const geocodingRouter = express.Router()

// 카카오 로컬 API 주소 검색 URL
const KAKAO_GEOCODE_URL = 'https://dapi.kakao.com/v2/local/search/address.json'
const KAKAO_REST_API_KEY = process.env.KAKAO_REST_API_KEY;

geocodingRouter.post('/', async(req, res) => {
  const {address} = req.body;

  if (!address) {
    return res.status(400).json({ detail: '주소를 입력해주세요.'})
  }

  try {
    const response = await axios.get(KAKAO_GEOCODE_URL,{
      params: { query: address},
      headers: {
        Authorization: `KakaoAK ${KAKAO_REST_API_KEY}`
        // Authorization: `KakaoAK ${KAKAO_REST_API_KEY}` 이거는 바꾸지 말기.
      }
    })
    if(response.data.documents.length === 0) {
      return res.status(404).json({detail: '해당 주소 좌표를 찾을 수 없습니다.'})
    }

    const location = response.data.documents[0];
    res.json({
      lat:parseFloat(location.y),
      lng:parseFloat(location.x),
    })
  } catch (error) {
    console.error( '카카오 주소 변환 API 통신 중 에러 발생: ', error.message)
    res.status(500).json({detail:'주소 변환 중 서버에서 에러가 발생했습니다.'})
  }
})

export default geocodingRouter