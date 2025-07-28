import {Map, MapMarker} from 'react-kakao-maps-sdk'

const MapComponent = ({lat,lng}) => {
  return(
    <Map center={{lat:lat, lng:lng}} style={{width:'100%', height:'450px'}} level={4}>
      <MapMarker position={{lat:lat, lng:lng}}/>
    </Map>
  )
}

export default MapComponent