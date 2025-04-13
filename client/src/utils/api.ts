// client/src/utils/api.ts

export interface Detection {
  bbox: number[];      // [x_min, y_min, x_max, y_max]
  centroid: number[];  // [center_x, center_y]
  label: string;
}

export interface DetectionResponse {
  detections: Detection[];
}

export async function detectSignLanguage(imageFile: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append('image', imageFile);

  // Use the backend URL from the environment variable (or fallback to localhost in development)
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/detect/';

  const response = await fetch(API_URL, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Detection API error: ${response.statusText}`);
  }

  return response.json();
}
