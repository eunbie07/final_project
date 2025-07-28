import express from 'express'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import db from '../db.js'
import 'dotenv/config'

const loginRouter = express.Router();

loginRouter.post('/', async (req,res) => {
  try{
    const {email, password} = req.body;

    if(!email || !password){
      return res.status(400).json({message:'이메일과 비밀번호는 필수입니다.'});
    }
    
    const [users] = await db.query(
      'SELECT * FROM users WHERE email = ?',[email]
    );
    
    if(users.length === 0){
      return res.status(401).json({message: '사용자를 찾을 수 없습니다.'});
    }

    const user = users[0];
    const isPasswordMatch = await bcrypt.compare(password, user.password);

    if (!isPasswordMatch) {
      return res.status(401).json({message: '비밀번호가 일치하지 않습니다.'})
    }

    const payload = {id:user.id, email:user.email};
    const token = jwt.sign(
      payload,
      process.env.JWT_SECRET,
      {expiresIn:'1h'}
    );

    res.status(200).json({
      message: '로그인 성공!',
      token: token
    })
  }catch(error){
    console.error('로그인 처리 중 오류 발생 : ', error)
    res.status(500).json({message : '서버 내부에서 오류가 발생했습니다.'})
  }
});

export default loginRouter;

