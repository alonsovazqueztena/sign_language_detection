'use client';

import Head from 'next/head';
import { useRef, useState } from 'react';
import { detectSignLanguage, Detection } from '../utils/api';

export default function Home() {
  const [detections, setDetections] = useState<Detection[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      alert('Please select an image file.');
      return;
    }

    try {
      const data = await detectSignLanguage(file);
      setDetections(data.detections);
      displayDetection(file, data.detections);
    } catch (err) {
      console.error(err);
      alert('Detection failed. See console for details.');
    }
  };

  const displayDetection = (imageFile: File, detections: Detection[]) => {
    const reader = new FileReader();
    reader.onload = function (e) {
      if (!e.target?.result) return;
      const img = new Image();
      img.onload = function () {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Clear canvas and draw the uploaded image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw detection overlay if available
        if (detections && detections.length > 0) {
          const detection = detections[0];
          const [x_min, y_min, x_max, y_max] = detection.bbox;
          // Scale if needed based on canvas and image sizes:
          const scaleX = canvas.width / img.width;
          const scaleY = canvas.height / img.height;
          const boxX = x_min * scaleX;
          const boxY = y_min * scaleY;
          const boxWidth = (x_max - x_min) * scaleX;
          const boxHeight = (y_max - y_min) * scaleY;

          ctx.lineWidth = 3;
          ctx.strokeStyle = 'red';
          ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
          ctx.fillStyle = 'red';
          ctx.font = '20px Arial';
          ctx.fillText(detection.label, boxX, boxY - 5);
        }
      };
      img.src = e.target.result as string;
    };
    reader.readAsDataURL(imageFile);
  };

  return (
    <>
      <Head>
        <title>Sign Language Detector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <div className="min-h-screen bg-gradient-to-br from-blue-100 to-indigo-200 flex flex-col items-center justify-center py-10">
        <h1 className="text-5xl font-bold text-gray-800 mb-6">Sign Language Detector</h1>
        <div className="bg-white rounded-xl shadow-2xl p-8 w-full max-w-xl text-center">
          <p className="text-xl text-gray-600 mb-4">
            Upload an image to detect sign language gestures.
          </p>
          <input
            type="file"
            ref={fileInputRef}
            className="mb-4 block w-full max-w-xs mx-auto border border-gray-300 p-2 rounded-md"
          />
          <button
            onClick={handleUpload}
            className="px-5 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition"
          >
            Detect
          </button>
          <div className="mt-6">
            <canvas
              ref={canvasRef}
              width="600"
              height="400"
              className="border border-gray-300 rounded-lg mx-auto"
            ></canvas>
          </div>
        </div>
      </div>
    </>
  );
}
