using System.Text.Json;
using System.Text.Json.Serialization;
using api.Model.Config;
using Microsoft.Extensions.Options;

namespace api.Service
{
    public class MlService
    {
       private readonly MlConfig _config;
       private readonly HttpClient _httpClient;
       private readonly ILogger<MlService> _logger;

       public MlService(IOptions<MlConfig> config, ILogger<MlService> logger)
       {
           _config = config.Value;
           _logger = logger;
           
           if (string.IsNullOrEmpty(_config.BaseUrl))
           {
               throw new ArgumentException("Base url is required");
           }
           
           _httpClient = new HttpClient();
           _httpClient.BaseAddress = new Uri(_config.BaseUrl);
           _logger.LogInformation($"BaseUrl: {_config.BaseUrl}");
       }

       public async Task<string> PredictAsync(string imagePath)
       {
           try
           {
               // Construct the query string
               string imagePath1 = "/shared/images/" + imagePath;
               var requestUri = $"api/predict?image_path={Uri.EscapeDataString(imagePath1)}";

               // Make the GET request
               var response = await _httpClient.PostAsync(requestUri, null);

               if (response.IsSuccessStatusCode)
               {
                   var jsonResult = await response.Content.ReadAsStringAsync();
                   return jsonResult;
               }

               _logger.LogError($"Prediction failed: {await response.Content.ReadAsStringAsync()}");
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
           public required string PredictedClass { get; set; }

           public required string Confidence { get; set; }
       }
    }
}