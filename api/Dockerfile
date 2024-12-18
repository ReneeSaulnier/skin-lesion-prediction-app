# Use the .NET SDK image to build the app
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

# Set the working directory inside the container
WORKDIR /app

# Copy everything (including .csproj) into the container
COPY . .

# Restore dependencies and build the app
RUN dotnet publish -c Release -o /publish

# Use the runtime image to run the app
FROM mcr.microsoft.com/dotnet/aspnet:8.0

# Set the working directory for the app
WORKDIR /app

# Copy the built files from the build stage
COPY --from=build /publish .

# Expose the port for the API
EXPOSE 8080

# Start the application
ENTRYPOINT ["dotnet", "api.dll"]