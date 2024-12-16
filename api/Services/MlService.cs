using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace api.Services
{
    public class MlService
    {
        private readonly HttpClient _httpClient;

        public MlServiceClient(HttpClient httpClient)
        {
            _httpClient = httpClient;
            _httpClient.BaseAddress = new Uri("http://ml-service:8000");
        }

        public async Task<string> PredictAsync(Stream imageStream, string fileName)
        {
            using var content = new MultipartFormDataContent();
            var fileContent = new StreamContent(imageStream);
            fileContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpg");
            content.Add(fileContent, "file", fileName);

            var response = await _httpClient.PostAsync("/predict/", content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
    }
}