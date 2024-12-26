using System.Text.Json;
using System.Text.Json.Serialization;
using api.Model.Config;
using Microsoft.Extensions.Options;

namespace api.Service
{
    public class MlService
    {
       private readonly MlConfig _mlConfig;
       private readonly FileConfig _fileConfig;
       private readonly HttpClient _httpClient;
       private readonly ILogger<MlService> _logger;

       public MlService(IOptions<MlConfig> mlConfig, IOptions<FileConfig> fileConfig, ILogger<MlService> logger)
       {
           _mlConfig = mlConfig.Value;
           _fileConfig = fileConfig.Value;
           _logger = logger;
           
           if (string.IsNullOrEmpty(_mlConfig.BaseUrl))
           {
               throw new ArgumentException("Base url is required");
           }
           
           _httpClient = new HttpClient();
           _httpClient.BaseAddress = new Uri(_mlConfig.BaseUrl);
           _logger.LogInformation($"BaseUrl: {_mlConfig.BaseUrl}");
       }

       public async Task<string> PredictAsync(string imagePath)
       {
           try
           {
               // Construct the query string
               var requestUri = $"api/predict?image_path={Uri.EscapeDataString(_fileConfig.MountSharePath + imagePath)}";

               // Make the GET request, 
               // Content is null because its being passed in the requesturi
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