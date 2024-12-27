import axios from 'axios';

const uploadUrl = 'http://localhost:8080/api/upload';

const UploadApi = async (file) => {
    const formData = new FormData();
    try {
        formData.append('file', file);
        const response = await axios.post(uploadUrl, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        console.log('File uploaded:', response.data);
    } catch (error) {
        console.error('Error uploading file:', error);
        throw error;
    }
};

export default UploadApi;