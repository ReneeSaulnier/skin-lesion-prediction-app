using api.Service;
using api.Model.Config;

namespace api
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create a host for the application
            var host = Host.CreateDefaultBuilder(args)
                .ConfigureServices((context, services) =>
                {
                    // Appsettings
                    services.Configure<MlConfig>(context.Configuration.GetSection("mlConfig"));

                    // Service
                    services.AddSingleton<MlService>();

                    // Logger
                    services.AddLogging();

                    // Controller
                    services.AddControllers();
                })
                .Build();

            // Run the application
            host.Run();
        }
    }
}