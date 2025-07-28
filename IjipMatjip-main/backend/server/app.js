import express from 'express'
import signupRouter from './routes/signup.js'
import loginRouter from './routes/login.js'
import recommendRouter from  './routes/recommend.js'
import infrastructureRouter from './routes/infrastructure.js'
import geocodingRouter from './routes/geocoding.js'
import directionsRouter from './routes/directions.js'

const app = express();
const port = 8001;

app.use(express.json());

app.use('/api/signup', signupRouter);

app.use('/api/login', loginRouter)

app.use('/api/recommend',recommendRouter)

app.use('/api/infrastructure',infrastructureRouter)

app.use('/api/geocode', geocodingRouter)

app.use('/api/directions', directionsRouter)

app.listen(port, () => {
  console.log(`✅ Server is running on port ${port}`)
})