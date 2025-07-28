import React from 'react'
import ReactDOM from 'react-dom/client'
import {createBrowserRouter, RouterProvider} from 'react-router-dom'
import './index.css'

import App from './App.jsx'
import PreferenceView from './pages/PreferenceView.jsx'
import RecommendationView from './pages/RecommendationView.jsx'
import DetailView from './pages/DetailView.jsx'

const router = createBrowserRouter([
  {
    path:'/',
    element:<App/>,
    children:[
      {path:'/', element:<PreferenceView/>},// 조건 입력
      {path:'/recommend',element:<RecommendationView/>}, // 추천 결과
      {path:'/detail/:propertyId',element:<DetailView/>} // 상세분석
    ]
  }
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router}/>
  </React.StrictMode>
)
