// client/src/app/page.tsx
'use client';

import Head from 'next/head';
import { useRef, useState, useEffect } from 'react';
import { detectSignLanguage, Detection } from '../utils/api';

// Define a type for each captured image result
type CapturedResult = {
  imageUrl: string;
  detection: Detection | null;
};

// Define a type for a saved session
type Session = {
  timestamp: number;
  chain: string[];
};

const LETTER_OPTIONS = Array.from({ length: 26 }, (_, i) =>
  String.fromCharCode(65 + i)
);

export default function Home() {
  // Existing states for detection, loading, webcam, etc.
  const [detections, setDetections] = useState<Detection[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [webcamActive, setWebcamActive] = useState(true);
  const [capturedImages, setCapturedImages] = useState<CapturedResult[]>([]);
  // Editable letter chain as an array of strings (one per captured image)
  const [letterChain, setLetterChain] = useState<string[]>([]);
  // Which letter (by index) is currently being edited (if any)
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  // Session history: an array of saved sessions (letter chain and a timestamp)
  const [sessions, setSessions] = useState<Session[]>([]);

  // Refs for file input, main canvas, video element, and the current stream
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Initialize (or reinitialize) the webcam if active.
  useEffect(() => {
    if (webcamActive) initWebcam();
    return () => {
      stopWebcam();
    };
  }, [webcamActive]);

  const initWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const toggleWebcam = () => {
    if (webcamActive) {
      stopWebcam();
      setWebcamActive(false);
    } else {
      setWebcamActive(true);
    }
  };

  // When a photo is captured (via upload or webcam), add its result and update the chain
  const addCapturedResult = (imageFile: File, detection: Detection | null) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        const imageUrl = e.target.result as string;
        setCapturedImages((prev) => [...prev, { imageUrl, detection }]);
        // Append the detected letter to the editable letter chain.
        setLetterChain((prev) => [
          ...prev,
          detection && detection.label ? detection.label : ''
        ]);
      }
    };
    reader.readAsDataURL(imageFile);
  };

  // Handle file upload detection
  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      alert('Please select an image file.');
      return;
    }
    setLoading(true);
    try {
      const data = await detectSignLanguage(file);
      setDetections(data.detections);
      displayDetection(file, data.detections);
      addCapturedResult(
        file,
        data.detections && data.detections.length > 0 ? data.detections[0] : null
      );
    } catch (err) {
      console.error(err);
      alert('Detection failed. See console for details.');
    } finally {
      setLoading(false);
    }
  };

  // Capture a photo from the webcam and send it for detection
  const capturePhoto = () => {
    if (!videoRef.current) {
      alert('Webcam not available');
      return;
    }
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoRef.current.videoWidth;
    tempCanvas.height = videoRef.current.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;
    tempCtx.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
    tempCanvas.toBlob(async (blob) => {
      if (!blob) return;
      const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
      setLoading(true);
      try {
        const data = await detectSignLanguage(file);
        setDetections(data.detections);
        displayDetection(file, data.detections);
        addCapturedResult(
          file,
          data.detections && data.detections.length > 0 ? data.detections[0] : null
        );
      } catch (err) {
        console.error(err);
        alert('Detection failed. See console for details.');
      } finally {
        setLoading(false);
      }
    }, 'image/jpeg');
  };

  // Display the detection result on the main canvas
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
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        if (detections && detections.length > 0) {
          const detection = detections[0];
          const [x_min, y_min, x_max, y_max] = detection.bbox;
          const scaleX = canvas.width / img.width;
          const scaleY = canvas.height / img.height;
          const boxX = x_min * scaleX;
          const boxY = y_min * scaleY;
          const boxWidth = (x_max - x_min) * scaleX;
          const boxHeight = (y_max - y_min) * scaleY;
          ctx.lineWidth = 3;
          ctx.strokeStyle = '#e63946';
          ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
          ctx.fillStyle = '#e63946';
          ctx.font = '20px sans-serif';
          ctx.fillText(detection.label, boxX, boxY - 5);
        }
      };
      img.src = e.target.result as string;
    };
    reader.readAsDataURL(imageFile);
  };

  // Handler to update a single letter in the chain
  const updateLetter = (index: number, letter: string) => {
    setLetterChain((prev) => {
      const updated = [...prev];
      updated[index] = letter;
      return updated;
    });
    setEditingIndex(null);
  };

  // Save the current session: store the letter chain along with a timestamp
  const saveSession = () => {
    if (letterChain.length === 0) {
      alert('No letters to save for this session.');
      return;
    }
    setSessions((prev) => [
      ...prev,
      { timestamp: Date.now(), chain: [...letterChain] }
    ]);
    // Clear current session's chain and captured images if desired.
    setLetterChain([]);
    setCapturedImages([]);
  };

  // Download session transcript as a text file
  const downloadSession = (session: Session) => {
    const transcript = session.chain.join('');
    const blob = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `session-${session.timestamp}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Create a chain string from the captured detections
  // (The editable letter chain is stored as an array of letters.)
  // Slider navigation for previous captures is preserved.
  const [sliderIndex, setSliderIndex] = useState(0);
  const nextSlide = () => {
    setSliderIndex((prev) => (prev + 1) % capturedImages.length);
  };
  const prevSlide = () => {
    setSliderIndex((prev) => (prev - 1 + capturedImages.length) % capturedImages.length);
  };

  return (
    <>
      <Head>
        <title>Sign Language Detector - AHHH</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <main>
        {/* Logo/Header */}
        <header className="logo-header">
          <div className="logo">
            <img src="/a.gif" alt="A" className="logo-letter" />
            <img src="/h.gif" alt="H" className="logo-letter" />
            <img src="/h.gif" alt="H" className="logo-letter" />
            <img src="/h.gif" alt="H" className="logo-letter" />
          </div>
          <h1>Sign Language Detector</h1>
          <p>Use your webcam, capture photos, or upload images to detect sign language.</p>
        </header>

        <section className="container">
          <div className="webcam-container">
            <h2>Webcam Feed</h2>
            {webcamActive ? (
              <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />
            ) : (
              <div className="webcam-off">Webcam is off</div>
            )}
            <div className="webcam-controls">
              <button onClick={toggleWebcam} className="control-button">
                {webcamActive ? 'Turn Webcam Off' : 'Turn Webcam On'}
              </button>
              {webcamActive && (
                <button onClick={capturePhoto} className="control-button">
                  Capture Photo
                </button>
              )}
            </div>
          </div>
          <div className="detection-container">
            <h2>Upload & Detect</h2>
            <input type="file" ref={fileInputRef} className="file-input" />
            <button onClick={handleUpload} className="detect-button" disabled={loading}>
              {loading ? 'Detecting...' : 'Detect'}
            </button>
            <canvas ref={canvasRef} width="600" height="400" className="detection-canvas" />
            <div className="chain-box">
              <label htmlFor="chainText">Letter Chain:</label>
              <div className="letter-chain">
                {letterChain.map((letter, index) =>
                  editingIndex === index ? (
                    <select
                      key={index}
                      value={letter}
                      onChange={(e) => updateLetter(index, e.target.value)}
                      onBlur={() => setEditingIndex(null)}
                      autoFocus
                    >
                      {LETTER_OPTIONS.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span
                      key={index}
                      className="chain-letter"
                      onClick={() => setEditingIndex(index)}
                      title="Click to edit"
                    >
                      {letter}
                    </span>
                  )
                )}
              </div>
            </div>
            <div className="session-controls">
              <button onClick={saveSession} className="control-button">
                Save Session
              </button>
            </div>
            {capturedImages.length > 0 && (
              <div className="slider-toggle">
                <button className="control-button" onClick={() => setSliderIndex(0)}>
                  Show Previous Captures
                </button>
                <div className="slider-container">
                  <button className="slider-button" onClick={prevSlide}>
                    Previous
                  </button>
                  <div className="slider-image">
                    <img
                      src={capturedImages[sliderIndex].imageUrl}
                      alt="Previous capture"
                    />
                    {capturedImages[sliderIndex].detection && (
                      <div className="slider-label">
                        {capturedImages[sliderIndex].detection?.label}
                      </div>
                    )}
                  </div>
                  <button className="slider-button" onClick={nextSlide}>
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>

        {sessions.length > 0 && (
          <section className="session-history">
            <h2>Session History</h2>
            {sessions.map((session, idx) => (
              <div key={session.timestamp} className="session">
                <div className="session-info">
                  <span>
                    Session {idx + 1} -{' '}
                    {new Date(session.timestamp).toLocaleString()}
                  </span>
                  <button
                    onClick={() => downloadSession(session)}
                    className="control-button"
                  >
                    Download Transcript
                  </button>
                </div>
                <div className="session-chain">
                  {session.chain.join('')}
                </div>
              </div>
            ))}
          </section>
        )}
      </main>
      <style jsx>{`
        main {
          max-width: 1200px;
          margin: 40px auto;
          padding: 0 20px;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          color: #333;
        }
        .logo-header {
          text-align: center;
          margin-bottom: 30px;
        }
        .logo {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          margin-bottom: 10px;
        }
        .logo-letter {
          height: 50px;
          width: auto;
        }
        header h1 {
          font-size: 2.5rem;
          margin: 0;
        }
        header p {
          font-size: 1.125rem;
          color: #555;
          margin: 5px 0 20px;
        }
        .container {
          display: flex;
          flex-direction: row;
          gap: 20px;
        }
        .webcam-container,
        .detection-container {
          flex: 1;
          background: white;
          padding: 20px;
          border-radius: 10px;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        .webcam-container h2,
        .detection-container h2 {
          text-align: center;
        }
        .webcam-video {
          width: 100%;
          height: auto;
          border-radius: 8px;
          margin-bottom: 10px;
        }
        .webcam-off {
          width: 100%;
          height: 300px;
          display: flex;
          justify-content: center;
          align-items: center;
          background: #ccc;
          border-radius: 8px;
          font-size: 1.2rem;
          color: #555;
          margin-bottom: 10px;
        }
        .webcam-controls {
          display: flex;
          justify-content: center;
          gap: 10px;
          margin-bottom: 10px;
        }
        .control-button {
          padding: 8px 16px;
          font-size: 0.9rem;
          background-color: #457b9d;
          border: none;
          color: white;
          border-radius: 5px;
          cursor: pointer;
          transition: background-color 0.3s ease;
        }
        .control-button:hover {
          background-color: #1d3557;
        }
        .file-input {
          display: block;
          margin: 0 auto 10px;
          padding: 8px;
          font-size: 1rem;
        }
        .detect-button {
          display: block;
          margin: 0 auto 20px;
          padding: 10px 20px;
          font-size: 1rem;
          background-color: #457b9d;
          border: none;
          color: white;
          border-radius: 5px;
          cursor: pointer;
          transition: background-color 0.3s ease;
        }
        .detect-button:hover:not(:disabled) {
          background-color: #1d3557;
        }
        .detect-button:disabled {
          background-color: #a8dadc;
          cursor: not-allowed;
        }
        .detection-canvas {
          display: block;
          margin: 0 auto 20px;
          border: 2px solid #ccc;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .chain-box {
          text-align: center;
          margin-bottom: 20px;
        }
        .chain-box label {
          font-weight: bold;
        }
        .letter-chain {
          display: inline-block;
          margin-top: 10px;
        }
        .chain-letter {
          margin: 0 5px;
          padding: 2px 4px;
          border-bottom: 1px dashed #888;
          cursor: pointer;
          transition: color 0.3s ease;
        }
        .chain-letter:hover {
          color: #e63946;
        }
        .session-controls {
          text-align: center;
          margin-bottom: 20px;
        }
        .slider-toggle {
          text-align: center;
          margin-bottom: 10px;
        }
        .slider-container {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
        }
        .slider-button {
          padding: 8px 16px;
          font-size: 0.9rem;
          background-color: #457b9d;
          border: none;
          color: white;
          border-radius: 5px;
          cursor: pointer;
          transition: background-color 0.3s ease;
        }
        .slider-button:hover {
          background-color: #1d3557;
        }
        .slider-image {
          position: relative;
          max-width: 300px;
        }
        .slider-image img {
          width: 100%;
          border-radius: 8px;
        }
        .slider-label {
          position: absolute;
          bottom: 5px;
          left: 5px;
          background: rgba(230, 57, 70, 0.8);
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 1rem;
        }
        .session-history {
          background: #fff;
          padding: 20px;
          border-radius: 10px;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
          margin-top: 30px;
        }
        .session-history h2 {
          text-align: center;
        }
        .session {
          border-bottom: 1px solid #ddd;
          padding: 10px 0;
        }
        .session-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.9rem;
          margin-bottom: 5px;
        }
        .session-chain {
          font-size: 1.2rem;
          text-align: center;
          word-break: break-all;
        }
        @media (max-width: 768px) {
          .container {
            flex-direction: column;
          }
          .webcam-container,
          .detection-container {
            margin-bottom: 20px;
          }
          .detection-canvas {
            width: 100%;
            height: auto;
          }
        }
      `}</style>
      <style jsx global>{`
        body {
          margin: 0;
          padding: 0;
          background: #f7f7f7;
        }
      `}</style>
    </>
  );
}
