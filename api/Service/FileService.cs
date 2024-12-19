using api.Model.Config;
using Microsoft.Extensions.Options;


namespace api.Service
{
    public class FileService
    {
        private readonly FileConfig _config;
        private readonly ILogger<FileService> _logger;
    
        public FileService(IOptions<FileConfig> config, ILogger<FileService> logger)
        {
            _config = config.Value;
            _logger = logger;
        }

        public async Task<string> UploadFile(IFormFile? file)
        {
            // Ensure the file is not null or empty
            if (file == null || file.Length == 0)
            {
                _logger.LogWarning("Invalid file upload attempt.");
                throw new ArgumentException("The uploaded file is null or empty.");
            }

            var sharedPath = _config.MountSharePath;

            if (!Directory.Exists(sharedPath))
            {
                Directory.CreateDirectory(sharedPath);
            }
            
            var filePath = Path.Combine(sharedPath, file.FileName);
            
            // Save the file
            try
            {
                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }

                return filePath;
            }
            catch (Exception ex)
            {
                throw new IOException("Error while saving the file.", ex);
            }
        }
    }   
}