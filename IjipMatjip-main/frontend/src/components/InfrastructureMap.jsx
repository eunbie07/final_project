import {Map as KakaoMap, MapMarker} from 'react-kakao-maps-sdk';

const markerIcons = {
  school: { src: '/images/school.png', size: { width: 40, height: 40 } },
  subway: { src: '/images/subway.png', size: { width: 40, height: 40 } },
  hospital: { src: '/images/hospital.png', size: { width: 40, height: 40 } },
  mart: { src: '/images/mart.png', size: { width: 40, height: 40 } },
  park: { src: '/images/park.png', size: { width: 40, height: 40 } },
};

const InfrastructureMap = ({lat, lng, isPropertyMarker = false, markers = []}) => {
  // 원래있던 useEffect API 제거

  return (
    <KakaoMap center={{ lat, lng}} style={{width:'100%', height:'450px'}} level={5}>
      {isPropertyMarker && (
        <MapMarker
          position={{ lat, lng }}
          image={{
            src:'/images/home_pin.png',
            size:{width:48, height:48},
            options: {offset: {x:24, y:48}}
          }}
        />
      )}
      {markers.map((marker, index) => (
        <MapMarker
          key={index}
          position={{lat:marker.latitude, lng:marker.longitude}}
          image={markerIcons[marker.type]}
          title={marker.name}
        />
      ))}
    </KakaoMap>
  )
}

export default InfrastructureMap