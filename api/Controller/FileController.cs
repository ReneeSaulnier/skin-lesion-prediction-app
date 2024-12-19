using Microsoft.AspNetCore.Mvc;
using api.Service;

namespace api.Controller
{
    [ApiController]
    [Route("api")]
    public class FileController : ControllerBase
    {
        private readonly FileService _fileService;

        public FileController(FileService fileService)
        {
            _fileService = fileService;
        }
        
        [HttpPost("upload")]
        public async Task<IActionResult> UploadFile(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("File is empty or null");
            }

            try
            {
                var filePath = await _fileService.UploadFile(file);
                return Ok(filePath);            
            }
            catch (Exception ex)
            {
                return BadRequest(ex.Message);
            }
        }
    }
}