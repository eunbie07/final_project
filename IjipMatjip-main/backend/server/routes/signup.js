import express from 'express'
import bcrypt from 'bcryptjs'
import pool from '../db.js'

const signupRouter = express.Router();

//POST /api/signup
signupRouter.post('/', async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password){
          return res.status(400).json({message: '이메일과 비밀번호는 필수입니다.'});
        }
        const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
        if(!emailRegex.test(email)){
          return res.status(400).json({message: '올바른 이메일 형식이 아닙니다.'})
        }
        
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
        if (!passwordRegex.test(password)) {
            return res.status(400).json({
                message: '비밀번호는 8자 이상이며, 대문자, 소문자, 숫자, 특수문자를 각각 하나 이상 포함해야 합니다.'
            });
        }
        
        // 3. 이메일 중복 확인
        const [existingUsers] = await pool.query(
          'SELECT email FROM users WHERE email = ? ', [email]
        )

        if (existingUsers.length > 0) {
            return res.status(409).json({ message: '이미 가입된 이메일입니다.' });
        }

        const hashedPassword = await bcrypt.hash(password,10)

        await pool.query('INSERT INTO users (email, password) VALUES (?, ?)', [email, hashedPassword]);

        res.status(201).json({ message: '회원가입이 성공적으로 완료되었습니다.' });

    } catch (error) {
        console.error('회원가입 처리 중 오류 발생: ' , error);
        res.status(500).json({ message: '서버 오류가 발생했습니다.' });
    }
});

export default signupRouter;