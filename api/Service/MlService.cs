using api.Model.Config;
using Microsoft.Extensions.Logging;

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

       public async Task<string> PredictAsync()
       {
           // try
           // {
           //     _logger.LogInformation("Predicting...");
           // }
           return "test";
       }
    }
}