const BASE_URL = "http://localhost:8080/api"; // Replace with your backend's URL

/**
 * Sends the image path to the backend for prediction.
 * @param {string} imagePath - The path of the image to upload.
 * @returns {Promise} - A promise resolving with the prediction result.
 */
export const predictImage = async (imagePath) => {
  const payload = { image_path: imagePath };

  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    const data = await response.json();
    return data; // Assuming `data` contains the prediction result
  } catch (error) {
    console.error("Prediction error:", error);
    throw error;
  }
};