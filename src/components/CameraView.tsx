import { type RefObject } from 'react';

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  mirror: boolean;
}

/** Video feed with the landmark overlay canvas stacked on top. */
export default function CameraView({ videoRef, canvasRef, mirror }: Props) {
  const flip = mirror ? 'scale-x-[-1]' : '';
  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-black aspect-video">
      <video
        ref={videoRef as RefObject<HTMLVideoElement>}
        className={`absolute inset-0 h-full w-full object-cover ${flip}`}
        playsInline
        muted
      />
      <canvas
        ref={canvasRef as RefObject<HTMLCanvasElement>}
        className="absolute inset-0 h-full w-full object-cover"
      />
    </div>
  );
}
