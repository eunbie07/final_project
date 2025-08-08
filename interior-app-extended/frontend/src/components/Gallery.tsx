import React from "react";

export default function Gallery({ items }:{ items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="gallery">
      {items.map((src,i)=>(<img key={i} src={src} />))}
    </div>
  );
}
