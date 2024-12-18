using System.Text;
using System.Text.Json;
using api.Model.Config;

namespace api.Service
{
    public class MlService
    {
       private readonly MlConfig _config;
       private readonly HttpClient _httpClient;
       private readonly ILogger<MlService> _logger;

       public MlService(MlConfig config, ILogger<MlService> logger)
       {
           _config = config;
           _logger = logger;
           
           if (string.IsNullOrEmpty(_config.BaseUrl))
           {
               throw new ArgumentException("Base url is required");
           }
           
           _httpClient = new HttpClient();
           _httpClient.BaseAddress = new Uri(_config.BaseUrl);
           _logger.LogInformation($"BaseUrl: {_config.BaseUrl}");
       }

       public async Task<PredictionResponse> PredictAsync(string imagePath)
       {
           // Payload
           var payload = new { image_path = imagePath };
           var jsonPayload = JsonSerializer.Serialize(payload);
           var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

           try
           {
               var response = await _httpClient.PostAsync("predict/", content);

               if (response.IsSuccessStatusCode)
               {
                   var jsonResponse = await response.Content.ReadAsStringAsync();
                   var result = JsonSerializer.Deserialize<PredictionResponse>(jsonResponse);
                   return result;
               }
           }
           catch (Exception ex)
           {
               throw new Exception($"Failed to communicate with FastAPI: {ex.Message}");
           }
           
           return null;
       }
       
       // Class to hold the response
       public class PredictionResponse
       {
           public required string Predicted_Class { get; set; }
           public required string Confidence { get; set; }
       }
    }
}