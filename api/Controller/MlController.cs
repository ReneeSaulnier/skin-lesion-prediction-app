using Microsoft.AspNetCore.Mvc;
using api.Service;

namespace api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class PredictionController : ControllerBase
    {
        private readonly MlService _mlService;

        public PredictionController(MlService mlService)
        {
            _mlService = mlService;
        }

        [HttpGet("predict")]
        public async Task<IActionResult> Predict([FromQuery] string imagePath)
        {
            if (string.IsNullOrEmpty(imagePath))
            {
                return BadRequest("Image path cannot be empty.");
            }

            try
            {
                var result = await _mlService.PredictAsync(imagePath);
                return Ok(new { prediction = result });
            }
            catch (Exception ex)
            {
                return BadRequest(ex.Message);
            }
        }
    }
}