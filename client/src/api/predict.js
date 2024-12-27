import axios from 'axios';

const predictUrl = 'http://localhost:8080/api/predict';
    
const predictApi = async (file) => {
    const formData = new FormData();
    try {
      formData.append('imagePath', file.name);
      // Send the FormData
      const response = await axios.post(predictUrl, formData, {
        headers: {
          'accept': '*/*',
        },
      });
    
      console.log('Prediction:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('Error predicting file:', error.response || error.message);
    }
    return result;

};

export default predictApi;