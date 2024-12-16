using Microsoft.AspNetCore.Mvc;
using api.Services;

namespace api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class MlController : ControllerBase
    {
        private readonly MlServiceClient _mlServiceClient;

        public PredictionController(MlServiceClient mlServiceClient)
        {
            _mlServiceClient = mlServiceClient;
        }

        [HttpPost("predict")]
        public async Task<IActionResult> Predict(IFormFile file)
        {
            if (file == null || file.Length == 0)
                return BadRequest("No file uploaded.");

            using var stream = file.OpenReadStream();
            var result = await _mlServiceClient.PredictAsync(stream, file.FileName);

            return Ok(new { prediction = result });
        }
    }
}