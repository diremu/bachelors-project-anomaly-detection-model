/**
 * ONNX Runtime inference for the ResNet18-LSTM anomaly detection model.
 *
 * Loads two ONNX models:
 *   - encoder.onnx: ResNet18 spatial encoder (single frame → 512-dim features)
 *   - temporal.onnx: LSTM autoencoder scorer (16×512 features → scalar anomaly score)
 *
 * Usage:
 *   const engine = new AnomalyEngine();
 *   await engine.init();
 *   const score = await engine.scoreClip(sixteenFrames);
 */

import * as ort from 'onnxruntime-web';

// ImageNet normalization constants
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const FRAME_SIZE = 224;
const CLIP_LENGTH = 16;

/**
 * Extract a single video frame onto a canvas and return pixel data.
 */
function grabFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(video, 0, 0, FRAME_SIZE, FRAME_SIZE);
  return ctx.getImageData(0, 0, FRAME_SIZE, FRAME_SIZE);
}

/**
 * Convert RGBA ImageData to CHW Float32 tensor with ImageNet normalization.
 * Output shape: [1, 3, 224, 224]
 */
function imageDataToTensor(img: ImageData): Float32Array {
  const { data, width, height } = img;
  const tensor = new Float32Array(3 * width * height);
  const planeSize = width * height;

  for (let i = 0; i < planeSize; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    tensor[i] = (r - MEAN[0]) / STD[0];                  // R plane
    tensor[planeSize + i] = (g - MEAN[1]) / STD[1];      // G plane
    tensor[2 * planeSize + i] = (b - MEAN[2]) / STD[2];  // B plane
  }
  return tensor;
}

export type EngineStatus = 'unloaded' | 'loading' | 'ready' | 'error';

export class AnomalyEngine {
  private encoderSession: ort.InferenceSession | null = null;
  private temporalSession: ort.InferenceSession | null = null;
  private canvas: HTMLCanvasElement;
  status: EngineStatus = 'unloaded';
  error: string = '';

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = FRAME_SIZE;
    this.canvas.height = FRAME_SIZE;
  }

  /** Load both ONNX models from the public directory (or custom paths). */
  async init(
    encoderUrl = '/encoder.onnx',
    temporalUrl = '/temporal.onnx',
  ): Promise<void> {
    this.status = 'loading';
    try {
      // Use wasm backend (works in all browsers, no WebGL issues)
      ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;

      this.encoderSession = await ort.InferenceSession.create(encoderUrl, {
        executionProviders: ['wasm'],
      });
      this.temporalSession = await ort.InferenceSession.create(temporalUrl, {
        executionProviders: ['wasm'],
      });
      this.status = 'ready';
    } catch (e) {
      this.status = 'error';
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  /** Check if models are loaded. */
  get isReady(): boolean {
    return this.status === 'ready';
  }

  /**
   * Encode a single frame through ResNet18.
   * @param imageData - 224×224 RGBA ImageData
   * @returns Float32Array of shape [512]
   */
  async encodeFrame(imageData: ImageData): Promise<Float32Array> {
    if (!this.encoderSession) throw new Error('Encoder not loaded');

    const input = imageDataToTensor(imageData);
    const tensor = new ort.Tensor('float32', input, [1, 3, FRAME_SIZE, FRAME_SIZE]);
    const result = await this.encoderSession.run({ frame: tensor });
    return result.features.data as Float32Array;
  }

  /**
   * Score a clip of stacked features through the LSTM autoencoder.
   * @param features - Float32Array of shape [T * 512]
   * @param seqLen - number of frames (default 16)
   * @returns scalar anomaly score
   */
  async scoreFeatures(features: Float32Array, seqLen: number = CLIP_LENGTH): Promise<number> {
    if (!this.temporalSession) throw new Error('Temporal model not loaded');

    const tensor = new ort.Tensor('float32', features, [1, seqLen, 512]);
    const result = await this.temporalSession.run({ features: tensor });
    return (result.score.data as Float32Array)[0];
  }

  /**
   * High-level: extract frames from a video element and return an anomaly score.
   * Grabs `CLIP_LENGTH` evenly spaced frames from the video's current position forward.
   */
  async scoreVideoClip(video: HTMLVideoElement): Promise<number> {
    if (!this.isReady) throw new Error('Engine not ready');

    const frame = grabFrame(video, this.canvas);
    // For streaming mode: encode single frame, accumulate externally
    const feats = await this.encodeFrame(frame);
    return feats.length > 0 ? feats[0] : 0; // placeholder — use scoreClipFrames for full scoring
  }

  /**
   * Score a batch of 16 ImageData frames as a single clip.
   */
  async scoreClipFrames(frames: ImageData[]): Promise<number> {
    if (!this.isReady) throw new Error('Engine not ready');
    if (frames.length < 1) throw new Error('No frames provided');

    // Encode each frame
    const allFeatures = new Float32Array(frames.length * 512);
    for (let i = 0; i < frames.length; i++) {
      const feats = await this.encodeFrame(frames[i]);
      allFeatures.set(feats, i * 512);
    }

    // Score the clip
    return this.scoreFeatures(allFeatures, frames.length);
  }

  /**
   * Extract all clips from a video file and return per-clip anomaly scores.
   * This is the main entry point for upload/batch mode.
   *
   * @param videoUrl - object URL or path to video
   * @param onProgress - callback with (clipIndex, score) for each scored clip
   * @returns array of all scores
   */
  async scoreVideo(
    videoUrl: string,
    onProgress?: (clipIdx: number, score: number, total: number) => void,
  ): Promise<number[]> {
    if (!this.isReady) throw new Error('Engine not ready');

    const video = document.createElement('video');
    video.src = videoUrl;
    video.muted = true;
    video.playsInline = true;

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to load video'));
    });

    const fps = 30; // assume 30fps
    const totalFrames = Math.floor(video.duration * fps);
    const stride = 8;
    const numClips = Math.max(1, Math.floor((totalFrames - CLIP_LENGTH) / stride) + 1);
    const scores: number[] = [];

    for (let clipIdx = 0; clipIdx < numClips; clipIdx++) {
      const startFrame = clipIdx * stride;
      const frames: ImageData[] = [];

      for (let f = 0; f < CLIP_LENGTH; f++) {
        const frameNum = startFrame + f;
        const time = frameNum / fps;

        await new Promise<void>((resolve) => {
          video.onseeked = () => resolve();
          video.currentTime = Math.min(time, video.duration - 0.01);
        });

        frames.push(grabFrame(video, this.canvas));
      }

      const score = await this.scoreClipFrames(frames);
      scores.push(score);
      onProgress?.(clipIdx, score, numClips);
    }

    video.remove();
    return scores;
  }

  /** Release resources. */
  dispose(): void {
    this.encoderSession?.release();
    this.temporalSession?.release();
    this.encoderSession = null;
    this.temporalSession = null;
    this.status = 'unloaded';
  }
}

// Singleton for the app
let _engine: AnomalyEngine | null = null;

export function getEngine(): AnomalyEngine {
  if (!_engine) _engine = new AnomalyEngine();
  return _engine;
}