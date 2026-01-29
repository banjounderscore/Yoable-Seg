using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;

namespace YoableSeg.Managers
{
    public class PythonServiceManager : IDisposable
    {
        private static PythonServiceManager _instance;
        public static PythonServiceManager Instance => _instance ??= new PythonServiceManager();

        private Process _serverProcess;
        private readonly HttpClient _httpClient;
        private bool _isRunning;
        private readonly string _serviceDirectory;
        private readonly int _port;

        public bool IsRunning => _isRunning;
        public event Action<string> OnLogMessage;
        public event Action<bool> OnStatusChanged;

        public PythonServiceManager()
        {
            _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
            _port = Properties.Settings.Default.PythonServicePort;
            _serviceDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "YoableSeg", "PythonService");
        }

        public async Task<bool> StartServiceAsync()
        {
            if (_isRunning) return true;

            var venvPath = Path.Combine(_serviceDirectory, "venv");
            var pythonExe = Path.Combine(venvPath, "Scripts", "python.exe");

            // Check if venv exists
            if (!File.Exists(pythonExe))
            {
                var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "PythonService", "install.ps1");
                var result = MessageBox.Show(
                    $"SAM3 Python environment not found.\n\nPlease run the installation script first:\n{scriptPath}\n\nOpen installation folder?",
                    "SAM3 Setup Required",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Information);

                if (result == MessageBoxResult.Yes)
                {
                    Process.Start("explorer.exe", Path.GetDirectoryName(scriptPath));
                }
                return false;
            }

            // Copy service files
            CopyServiceFiles();

            // Kill any existing process on port
            await KillProcessOnPortAsync(_port);

            // Start the service
            var serverScript = Path.Combine(_serviceDirectory, "inference_server.py");
            
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"-u \"{serverScript}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = _serviceDirectory
            };

            psi.EnvironmentVariables["PORT"] = _port.ToString();
            psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

            _serverProcess = new Process { StartInfo = psi };
            _serverProcess.OutputDataReceived += (s, e) => { if (!string.IsNullOrEmpty(e.Data)) OnLogMessage?.Invoke(e.Data); };
            _serverProcess.ErrorDataReceived += (s, e) => { if (!string.IsNullOrEmpty(e.Data)) OnLogMessage?.Invoke(e.Data); };

            _serverProcess.Start();
            _serverProcess.BeginOutputReadLine();
            _serverProcess.BeginErrorReadLine();

            // Wait for service to be ready
            for (int i = 0; i < 60; i++)
            {
                await Task.Delay(1000);
                if (await CheckHealthAsync())
                {
                    _isRunning = true;
                    OnStatusChanged?.Invoke(true);
                    OnLogMessage?.Invoke($"SAM3 ready on port {_port}");
                    return true;
                }
            }

            return false;
        }

        private void CopyServiceFiles()
        {
            var sourceDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "PythonService");
            var files = new[] { "inference_server.py", "models.py", "utils.py" };

            Directory.CreateDirectory(_serviceDirectory);

            foreach (var file in files)
            {
                var src = Path.Combine(sourceDir, file);
                var dst = Path.Combine(_serviceDirectory, file);
                if (File.Exists(src))
                {
                    File.Copy(src, dst, true);
                }
            }
        }

        private async Task KillProcessOnPortAsync(int port)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "powershell",
                    Arguments = $"-Command \"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object {{ Stop-Process -Id $_ -Force }}\"",
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                await process.WaitForExitAsync();
                await Task.Delay(1000);
            }
            catch { }
        }

        public async Task<bool> CheckHealthAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"http://localhost:{_port}/health");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        public async Task<(bool success, double[][] polygon)> RequestBoxMaskAsync(
            string imagePath, 
            double x1, 
            double y1, 
            double x2, 
            double y2, 
            double simplification = -1,
            List<List<double[]>> existingPolygons = null,
            double nmsIouThreshold = 0.5)
        {
            try
            {
                if (simplification < 0)
                    simplification = Properties.Settings.Default.PolygonSimplificationTolerance;

                var request = new
                {
                    image_path = imagePath,
                    box = new[] { x1, y1, x2, y2 },
                    simplification = simplification,
                    min_sam3_confidence = Properties.Settings.Default.MinSam3Confidence,
                    existing_polygons = existingPolygons,
                    nms_iou_threshold = nmsIouThreshold
                };

                var json = JsonSerializer.Serialize(request);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                var response = await _httpClient.PostAsync($"http://localhost:{_port}/box_to_mask", content);
                
                if (!response.IsSuccessStatusCode)
                    return (false, null);

                var resultJson = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<MaskResponse>(resultJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                
                return (true, result?.polygon);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"RequestBoxMaskAsync error: {ex.Message}");
                return (false, null);
            }
        }

        public async Task<List<AutoLabelResult>> RequestAutoLabelAsync(
            string imagePath, 
            List<string> prompts, 
            Dictionary<string, int> promptToClassMapping,
            float textThreshold,
            float boxThreshold,
            int maxInstances,
            double simplification = -1,
            List<List<double[]>> existingPolygons = null,
            double nmsIouThreshold = 0.5)
        {
            try
            {
                if (simplification < 0)
                    simplification = Properties.Settings.Default.PolygonSimplificationTolerance;

                var request = new
                {
                    image_path = imagePath,
                    prompts = prompts,
                    text_threshold = textThreshold,
                    box_threshold = boxThreshold,
                    max_instances = maxInstances,
                    simplification = simplification,
                    min_sam3_confidence = Properties.Settings.Default.MinSam3Confidence,
                    existing_polygons = existingPolygons,
                    nms_iou_threshold = nmsIouThreshold
                };

                var json = JsonSerializer.Serialize(request);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                var response = await _httpClient.PostAsync($"http://localhost:{_port}/autolabel", content);
                
                if (!response.IsSuccessStatusCode)
                    return null;

                var resultJson = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<AutoLabelResponseData>(resultJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                
                if (result?.instances == null)
                    return null;

                var labelResults = new List<AutoLabelResult>();
                foreach (var instance in result.instances)
                {
                    if (promptToClassMapping.TryGetValue(instance.phrase.ToLower(), out int classId))
                    {
                        labelResults.Add(new AutoLabelResult
                        {
                            polygon = instance.polygon,
                            classId = classId,
                            score = instance.score,
                            phrase = instance.phrase
                        });
                    }
                }

                return labelResults;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"RequestAutoLabelAsync error: {ex.Message}");
                return null;
            }
        }

        public void StopService()
        {
            if (!_isRunning && _serverProcess == null)
                return;

            _isRunning = false;

            try
            {
                if (_serverProcess != null && !_serverProcess.HasExited)
                {
                    try
                    {
                        _serverProcess.Kill(true);
                        
                        var waitTask = Task.Run(() => _serverProcess.WaitForExit(500));
                        waitTask.Wait(500);
                        
                        _serverProcess.Dispose();
                    }
                    catch { }
                }

                var killPortTask = Task.Run(() => KillProcessByPortSync(_port));
                killPortTask.Wait(1000);

                var killPythonTask = Task.Run(() => KillAllServicePythonProcesses());
                killPythonTask.Wait(1000);
            }
            catch { }

            _serverProcess = null;
            OnStatusChanged?.Invoke(false);
        }

        private void KillProcessByPortSync(int port)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "powershell",
                    Arguments = $"-Command \"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object {{ Stop-Process -Id $_ -Force }}\"",
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                process?.WaitForExit(2000);
            }
            catch { }
        }

        private void KillAllServicePythonProcesses()
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "powershell",
                    Arguments = "-Command \"Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*YoableSeg*' } | Stop-Process -Force\"",
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                process?.WaitForExit(2000);
            }
            catch { }
        }

        public void Dispose()
        {
            StopService();
            _httpClient?.Dispose();
        }

        private class MaskResponse
        {
            public int width { get; set; }
            public int height { get; set; }
            public double[][] polygon { get; set; }
            public double score { get; set; }
        }

        private class AutoLabelResponseData
        {
            public int width { get; set; }
            public int height { get; set; }
            public List<AutoLabelInstance> instances { get; set; }
        }

        private class AutoLabelInstance
        {
            public double[] box { get; set; }
            public double score { get; set; }
            public string phrase { get; set; }
            public double[][] polygon { get; set; }
        }

        public class AutoLabelResult
        {
            public double[][] polygon { get; set; }
            public int classId { get; set; }
            public double score { get; set; }
            public string phrase { get; set; }
        }
    }
}
