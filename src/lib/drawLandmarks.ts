import type { Hand } from './types';

// MediaPipe hand connections (pairs of landmark indices).
const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17], // palm base
];

/** Draw hand skeletons over a canvas sized to the video. `mirror` flips X. */
export function drawHands(
  ctx: CanvasRenderingContext2D,
  hands: Hand[],
  w: number,
  h: number,
  mirror: boolean,
  color = '#38bdf8'
): void {
  ctx.clearRect(0, 0, w, h);
  const X = (x: number) => (mirror ? (1 - x) * w : x * w);
  const Y = (y: number) => y * h;

  for (const hand of hands) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    for (const [a, b] of CONNECTIONS) {
      if (!hand[a] || !hand[b]) continue;
      ctx.beginPath();
      ctx.moveTo(X(hand[a].x), Y(hand[a].y));
      ctx.lineTo(X(hand[b].x), Y(hand[b].y));
      ctx.stroke();
    }
    ctx.fillStyle = '#f8fafc';
    for (const p of hand) {
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
