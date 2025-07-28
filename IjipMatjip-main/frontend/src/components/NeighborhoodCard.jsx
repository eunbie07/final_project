
const NeighborhoodCard = ({dongData}) => {
  const tags = [];
  // 점수에 따라 동적으로 태그 생성
  if (dongData.school_score > 70) tags.push('#학군우수')
  if (dongData.subway_score > 70) tags.push('#역세권')
  if (dongData.price_score > 70) tags.push('#가성비')
  
  return (
    <div className="border border-solid border-[#e0e0e0], rounded-lg p-4 w-2xs cursor-pointer shadow-md">
      <h4 className="font-bold text-lg text-gray-800">{dongData.dong}</h4>
      <p className="text-sm text-gray-600 my-2">
        {/* todo : 나중에 LLM으로 동네 요약 생성 */}
        주변 인프라가 잘 갖춰진 동네
      </p>
      {dongData.commute_minutes && (
        <p className="text-sm font-medium text-gray-700">출퇴근: 🚇 약 {dongData.commute_minutes}분</p>
      )}
      <div className="mt-3 flex gap-2 flex-wrap">
        {tags.map(tag => (
          <span key={tag} className="bg-indigo-100 text-indigo-800 text-xs font-medium px-2 py-1 rounded-full">
            {tag}
          </span>
        ))}
      </div>
    </div>
  )
}

export default NeighborhoodCard