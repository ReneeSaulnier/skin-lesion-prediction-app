using System.Text;
using System.Text.Json;
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

       public async Task<PredictionResponse> PredictAsync(string imagePath)
       {
           // // Payload
           // var payload = new { image_path = imagePath };
           // var jsonPayload = JsonSerializer.Serialize(payload);
           // var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
           //
           // try
           // {
           //     var response = await _httpClient.PostAsync("predict/", content);
           //     Console.WriteLine("Response: " + await response.Content.ReadAsStringAsync());
           //
           //     if (response.IsSuccessStatusCode)
           //     {
           //         var jsonResponse = await response.Content.ReadAsStringAsync();
           //         var result = JsonSerializer.Deserialize<PredictionResponse>(jsonResponse);
           //         return result;
           //     }
           // }
           // catch (Exception ex)
           // {
           //     throw new Exception($"Failed to communicate with FastAPI: {ex.Message}");
           // }

           //return null;
           var encodedPath = Uri.EscapeDataString(imagePath);
           var url = $"http://localhost:5000/api/predict?imagePath=={encodedPath}";

           try
           {
               var response = await _httpClient.GetAsync(url);
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