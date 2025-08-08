import React, { useEffect, useImperativeHandle, useRef, useState, forwardRef } from "react";

export type PresetKey = "walls" | "floor" | "bedding";
export type MaskCanvasHandle = { applyPreset: (k: PresetKey)=>void; clearMask: ()=>void; };

type Props = {
  baseImage?: HTMLImageElement;
  brushSize?: number;
  onMaskBlob: (b: Blob) => void;
};

const MaskCanvas = forwardRef<MaskCanvasHandle, Props>(function MaskCanvas({ baseImage, brushSize = 24, onMaskBlob }, ref) {
  const baseRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement>(null);
  const [drawing, setDrawing] = useState(false);

  useImperativeHandle(ref, () => ({
    applyPreset: (k: PresetKey) => drawPreset(k),
    clearMask: () => clearMask()
  }));

  useEffect(() => {
    if (!baseImage || !baseRef.current) return;
    const c = baseRef.current;
    c.width = 512; c.height = 512;
    const ctx = c.getContext("2d")!;
    const scale = Math.min(512 / baseImage.width, 512 / baseImage.height);
    const w = Math.round(baseImage.width * scale);
    const h = Math.round(baseImage.height * scale);
    const dx = Math.floor((512 - w) / 2);
    const dy = Math.floor((512 - h) / 2);
    ctx.clearRect(0,0,512,512);
    ctx.drawImage(baseImage, dx, dy, w, h);

    if (maskRef.current) {
      maskRef.current.width = 512; maskRef.current.height = 512;
      const mctx = maskRef.current.getContext("2d")!;
      mctx.clearRect(0,0,512,512);
    }
  }, [baseImage]);

  useEffect(() => {
    if (!maskRef.current) return;
    const canvas = maskRef.current;
    const onDown = (e: MouseEvent) => { setDrawing(true); draw(e); };
    const onUp = () => { setDrawing(false); emitBlob(); };
    const onMove = (e: MouseEvent) => { if (drawing) draw(e); };

    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("mousemove", onMove);
    return () => {
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("mousemove", onMove);
    };
  }, [drawing, brushSize]);

  const draw = (e: MouseEvent) => {
    if (!maskRef.current) return;
    const rect = maskRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ctx = maskRef.current.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(x, y, (brushSize || 24) / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const clearMask = () => {
    if (!maskRef.current) return;
    const mctx = maskRef.current.getContext("2d")!;
    mctx.clearRect(0,0,512,512);
    emitBlob();
  };

  const drawPreset = (k: PresetKey) => {
    if (!maskRef.current) return;
    const mctx = maskRef.current.getContext("2d")!;
    mctx.fillStyle = "white";
    if (k === "walls") {
      mctx.fillRect(0, 0, 512, Math.floor(512 * 0.4));
    } else if (k === "floor") {
      const h = Math.floor(512 * 0.3);
      mctx.fillRect(0, 512 - h, 512, h);
    } else if (k === "bedding") {
      const cx = 256, cy = 320, rx = 180, ry = 100;
      mctx.beginPath();
      mctx.ellipse(cx, cy, rx, ry, 0, Math.PI * 2);
      mctx.fill();
    }
    emitBlob();
  };

  const emitBlob = () => {
    if (!maskRef.current) return;
    maskRef.current.toBlob((b) => b && onMaskBlob(b), "image/png");
  };

  return (
    <div className="canvasWrap">
      <canvas ref={baseRef} />
      <canvas ref={maskRef} />
    </div>
  );
});

export default MaskCanvas;
