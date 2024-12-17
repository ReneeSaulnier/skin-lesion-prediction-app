using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

using api.Model.Config;
using api.Service;

namespace api
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create the Host to automatically handle DI, configuration, and logging
            var host = Host.CreateDefaultBuilder(args)
                .ConfigureServices((context, services) =>
                {
                    // Automatically bind "mlConfig" section in appsettings.json to MlConfig
                    services.Configure<MlConfig>(context.Configuration.GetSection("mlConfig"));

                    // Register services
                    services.AddSingleton<MlService>();

                    // Add logging
                    services.AddLogging();
                })
                .Build();

            // Run the application logic
            RunApplication(host.Services);
        }

        static void RunApplication(IServiceProvider services)
        {
            using (var scope = services.CreateScope())
            {
                var serviceProvider = scope.ServiceProvider;

                try
                {
                    // Resolve logger and MlService
                    var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                    var mlService = serviceProvider.GetRequiredService<MlService>();

                    logger.LogInformation("Application started.");

                    // Call service method
                    var result = mlService.PredictAsync();
                    Console.WriteLine($"Prediction Result: {result}");

                    logger.LogInformation("Application finished.");
                }
                catch (Exception ex)
                {
                    var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                    logger.LogError(ex, "An error occurred while running the application.");
                }
            }
        }
    }
}