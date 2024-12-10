const BASE_URL = "http://localhost:5000"; // Replace with your backend's URL

/**
 * Sends an image file to the backend for prediction.
 * @param {File} imageFile - The image file to upload.
 * @returns {Promise} - A promise resolving with the prediction result.
 */
export const predictImage = async (imageFile) => {
  const formData = new FormData();
  formData.append("image", imageFile);

  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      body: formData,
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