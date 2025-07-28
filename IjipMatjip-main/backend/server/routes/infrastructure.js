import express from 'express'
import axios from 'axios'
import 'dotenv/config'

const infrastructureRouter = express.Router()

const AI_SERVER_URL = process.env.FASTAPI_SERVER

infrastructureRouter.post('/', async (req,res) => {
  try{
    // 1. 프론트로부터 받은 좌표, 반경 정보 가져옴
    const {latitude, longitude, radius_km} = req.body;

    // 2. AI 서버의 /infrastructure API 호출
    const response = await axios.post(`${AI_SERVER_URL}/infrastructure`, {latitude,longitude,radius_km,})

    // 3. AI 서버의 응답을 프론트에 전달
    res.json(response.data)
  } catch(error){
    console.error('AI 서버 인프라 API 통신 중 에러 발생: ', error.message)
    if(error.response){
      res.status(error.response.status).json(error.response.data)
    }else{
      res.status(500).json({detail:'AI 서버와 통신할 수 없습니다.'})
    }
  }
})

export default infrastructureRouter