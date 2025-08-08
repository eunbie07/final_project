import React from "react";

type Props = { onImage: (img: HTMLImageElement, file: File) => void; };

export default function Uploader({ onImage }: Props) {
  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => onImage(img, f);
    img.src = URL.createObjectURL(f);
  };
  return <input type="file" accept="image/*" onChange={onChange} />;
}
