// Webcam helpers: start/stop stream, enumerate + switch cameras.

export type Facing = 'user' | 'environment';

export interface CameraHandle {
  stream: MediaStream;
  stop: () => void;
}

/** Start the webcam and attach it to a <video> element. */
export async function startCamera(
  video: HTMLVideoElement,
  facing: Facing = 'user'
): Promise<CameraHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera API not available. Use HTTPS and a modern browser.');
  }

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    const e = err as DOMException;
    if (e.name === 'NotAllowedError') {
      throw new Error('Camera permission denied. Allow camera access and reload.');
    }
    if (e.name === 'NotFoundError') {
      throw new Error('No camera found on this device.');
    }
    throw new Error(`Could not start camera: ${e.message}`);
  }

  video.srcObject = stream;
  await video.play();

  return {
    stream,
    stop: () => {
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    },
  };
}

/** True if the device reports more than one video input (so flipping makes sense). */
export async function hasMultipleCameras(): Promise<boolean> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'videoinput').length > 1;
  } catch {
    return false;
  }
}
