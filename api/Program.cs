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
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.ConfigureServices((context, services) =>
                    {
                        // Bind "mlConfig" section in appsettings.json to MlConfig
                        services.Configure<MlConfig>(context.Configuration.GetSection("mlConfig"));
                        services.Configure<FileConfig>(context.Configuration.GetSection("file"));

                        // Register Service
                        services.AddSingleton<MlService>();
                        services.AddSingleton<FileService>();

                        // Add controllers
                        services.AddControllers();

                        // Add Swagger
                        services.AddEndpointsApiExplorer();
                        services.AddSwaggerGen();
                    });

                    webBuilder.Configure((context, app) =>
                    {
                        // Enable Swagger only for development or production (if needed)
                        //if (context.HostingEnvironment.IsDevelopment() || context.HostingEnvironment.IsProduction())
                        //{
                            app.UseSwagger();
                            app.UseSwaggerUI(c =>
                            {
                                c.SwaggerEndpoint("/swagger/v1/swagger.json", "Skin Cancer Prediction API v1");
                                c.RoutePrefix = string.Empty;
                            });
                        //}

                        // Add routing and map controllers
                        app.UseRouting();
                        app.UseAuthorization();
                        app.UseEndpoints(endpoints =>
                        {
                            endpoints.MapControllers();
                        });
                    });
                })
                .Build();

            // Run the application
            host.Run();
        }
    }
}