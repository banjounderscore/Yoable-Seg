using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using Point = System.Windows.Point;

namespace YoableSeg.Managers
{
    public class LabelManager
    {
        private ConcurrentDictionary<string, List<LabelData>> labelStorage = new();
        public ConcurrentDictionary<string, List<LabelData>> LabelStorage => labelStorage;
        public int LabelLoadBatchSize { get; set; } = 500;

        private HashSet<int> validClassIds = new HashSet<int> { 0 };
        private int defaultClassId = 0;

        public void SetValidClassIds(IEnumerable<int> classIds)
        {
            validClassIds = new HashSet<int>(classIds);
            
            if (validClassIds.Count == 0)
            {
                validClassIds.Add(0);
            }
            
            defaultClassId = validClassIds.Contains(0) ? 0 : validClassIds.First();
        }

        private bool ValidateAndFixClassId(LabelData label)
        {
            if (!validClassIds.Contains(label.ClassId))
            {
                label.ClassId = defaultClassId;
                return true;
            }
            return false;
        }

        public int FixAllOrphanedLabels()
        {
            int fixedCount = 0;
            
            foreach (var kvp in labelStorage)
            {
                foreach (var label in kvp.Value)
                {
                    if (ValidateAndFixClassId(label))
                    {
                        fixedCount++;
                    }
                }
            }
            
            return fixedCount;
        }

        public List<LabelData> GetLabels(string fileName)
        {
            if (!labelStorage.ContainsKey(fileName))
                return new List<LabelData>();
            
            var labels = labelStorage[fileName];
            
            foreach (var label in labels)
            {
                ValidateAndFixClassId(label);
            }
            
            return new List<LabelData>(labels);
        }

        public void SaveLabels(string fileName, List<LabelData> labels)
        {
            labelStorage.AddOrUpdate(fileName,
                new List<LabelData>(labels),
                (k, v) => new List<LabelData>(labels));
        }

        public void ClearAll()
        {
            labelStorage.Clear();
        }

        private float ParseFloat(string value)
        {
            if (float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float result))
            {
                return result;
            }

            var cultureWithComma = new CultureInfo("de-DE");
            if (float.TryParse(value, NumberStyles.Float, cultureWithComma, out result))
            {
                return result;
            }

            value = value.Replace(',', '.');
            if (float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out result))
            {
                return result;
            }

            throw new FormatException($"Unable to parse '{value}' as a float.");
        }

        public async Task<int> LoadYoloSegLabelsBatchAsync(
            string labelsDirectory,
            ImageManager imageManager,
            IProgress<(int current, int total, string message)> progress = null,
            CancellationToken cancellationToken = default,
            bool enableParallelProcessing = true)
        {
            if (!Directory.Exists(labelsDirectory))
                return 0;

            var labelFiles = Directory.GetFiles(labelsDirectory, "*.txt").ToArray();

            if (labelFiles.Length == 0)
                return 0;

            int totalLabelsAdded = 0;
            int processedFiles = 0;
            int totalFiles = labelFiles.Length;
            int batchSize = LabelLoadBatchSize;

            return await Task.Run(() =>
            {
                for (int i = 0; i < labelFiles.Length; i += batchSize)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var batch = labelFiles.Skip(i).Take(batchSize).ToArray();
                    int batchLabelsAdded = 0;

                    if (enableParallelProcessing)
                    {
                        var results = batch.AsParallel()
                            .WithCancellation(cancellationToken)
                            .Select(labelFile => LoadYoloSegLabelsOptimized(labelFile, imageManager))
                            .ToArray();

                        batchLabelsAdded = results.Sum();
                    }
                    else
                    {
                        foreach (var labelFile in batch)
                        {
                            batchLabelsAdded += LoadYoloSegLabelsOptimized(labelFile, imageManager);
                        }
                    }

                    totalLabelsAdded += batchLabelsAdded;
                    processedFiles += batch.Length;

                    progress?.Report((processedFiles, totalFiles,
                        $"Loading labels... {processedFiles}/{totalFiles} ({totalLabelsAdded} labels)"));
                }

                return totalLabelsAdded;
            }, cancellationToken);
        }

        private int LoadYoloSegLabelsOptimized(string labelFile, ImageManager imageManager)
        {
            if (!File.Exists(labelFile))
                return 0;

            string labelFileName = Path.GetFileNameWithoutExtension(labelFile);

            string[] possibleExtensions = { ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG" };
            ImageManager.ImageInfo imageInfo = null;
            string matchingImageFile = null;

            foreach (var ext in possibleExtensions)
            {
                string potentialImageFile = labelFileName + ext;
                if (imageManager.ImagePathMap.TryGetValue(potentialImageFile, out imageInfo))
                {
                    matchingImageFile = potentialImageFile;
                    break;
                }
            }

            if (imageInfo == null)
                return 0;

            int labelsAdded = 0;

            try
            {
                int imgWidth = (int)imageInfo.OriginalDimensions.Width;
                int imgHeight = (int)imageInfo.OriginalDimensions.Height;

                if (!labelStorage.ContainsKey(matchingImageFile))
                    labelStorage.TryAdd(matchingImageFile, new List<LabelData>());

                var newLabels = new List<LabelData>(50);

                foreach (var line in File.ReadLines(labelFile))
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    try
                    {
                        var tokens = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        
                        if (tokens.Length < 7)
                            continue;

                        int classId = 0;
                        if (int.TryParse(tokens[0], out int parsedClassId))
                        {
                            classId = parsedClassId;
                        }
                        
                        if (!validClassIds.Contains(classId))
                        {
                            Debug.WriteLine($"LabelManager: Fixing orphaned ClassId {classId} -> {defaultClassId} during import");
                            classId = defaultClassId;
                        }

                        var points = new List<Point>();
                        for (int i = 1; i < tokens.Length - 1; i += 2)
                        {
                            if (!float.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out float xNorm))
                                xNorm = ParseFloat(tokens[i]);
                            if (!float.TryParse(tokens[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float yNorm))
                                yNorm = ParseFloat(tokens[i + 1]);

                            double x = xNorm * imgWidth;
                            double y = yNorm * imgHeight;
                            points.Add(new Point(x, y));
                        }

                        if (points.Count >= 3)
                        {
                            var existingCount = labelStorage[matchingImageFile].Count;
                            var label = new LabelData(
                                $"Imported Label {existingCount + newLabels.Count + 1}",
                                points,
                                classId);
                            newLabels.Add(label);
                            labelsAdded++;
                        }
                    }
                    catch (FormatException)
                    {
                        continue;
                    }
                }

                if (newLabels.Count > 0)
                {
                    labelStorage.AddOrUpdate(
                        matchingImageFile,
                        newLabels,
                        (k, existing) =>
                        {
                            existing.AddRange(newLabels);
                            return existing;
                        });
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error loading YOLO-seg labels from {labelFile}: {ex.Message}");
            }

            return labelsAdded;
        }

        public int LoadYoloSegLabels(string labelFile, string imagePath, ImageManager imageManager)
        {
            if (!File.Exists(labelFile)) return 0;

            string fileName = Path.GetFileName(imagePath);

            if (!imageManager.ImagePathMap.TryGetValue(Path.GetFileName(imagePath), out ImageManager.ImageInfo imageInfo))
            {
                return 0;
            }
            imagePath = imageInfo.Path;

            int labelsAdded = 0;

            try
            {
                int imgWidth = (int)imageInfo.OriginalDimensions.Width;
                int imgHeight = (int)imageInfo.OriginalDimensions.Height;

                if (!labelStorage.ContainsKey(fileName))
                    labelStorage.TryAdd(fileName, new List<LabelData>());

                using StreamReader reader = new StreamReader(labelFile);
                string? line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 7) continue;

                    try
                    {
                        int classId = 0;
                        if (int.TryParse(parts[0], out int parsedClassId))
                        {
                            classId = parsedClassId;
                        }
                        
                        if (!validClassIds.Contains(classId))
                        {
                            classId = defaultClassId;
                        }

                        var points = new List<Point>();
                        for (int i = 1; i < parts.Length - 1; i += 2)
                        {
                            float xNorm = ParseFloat(parts[i]);
                            float yNorm = ParseFloat(parts[i + 1]);

                            double x = xNorm * imgWidth;
                            double y = yNorm * imgHeight;
                            points.Add(new Point(x, y));
                        }

                        if (points.Count >= 3)
                        {
                            var labelCount = labelStorage[fileName].Count + 1;
                            var label = new LabelData($"Imported Label {labelCount}", points, classId);
                            labelStorage[fileName].Add(label);
                            labelsAdded++;
                        }
                    }
                    catch (FormatException)
                    {
                        continue;
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error loading YOLO-seg labels: {ex.Message}");
            }

            return labelsAdded;
        }

        public void ExportLabelsToYoloSeg(string filePath, string imagePath, List<LabelData> labelsToExport)
        {
            using Bitmap image = new Bitmap(imagePath);
            int imageWidth = image.Width;
            int imageHeight = image.Height;

            using StreamWriter writer = new(filePath) { AutoFlush = true };

            foreach (var label in labelsToExport)
            {
                if (!label.IsValid()) continue;

                var sb = new System.Text.StringBuilder();
                sb.Append(label.ClassId);

                foreach (var point in label.Points)
                {
                    float xNorm = (float)(point.X / imageWidth);
                    float yNorm = (float)(point.Y / imageHeight);
                    sb.Append($" {xNorm:F6} {yNorm:F6}");
                }

                writer.WriteLine(sb.ToString());
            }
        }

        public async Task ExportSegLabelsBatchAsync(
            string outputDirectory,
            ImageManager imageManager,
            IProgress<(int current, int total, string message)> progress = null,
            CancellationToken cancellationToken = default)
        {
            if (!Directory.Exists(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            var labelsToExport = labelStorage.Where(kvp => kvp.Value.Count > 0).ToArray();
            int totalFiles = labelsToExport.Length;
            int processedFiles = 0;

            await Task.Run(() =>
            {
                Parallel.ForEach(labelsToExport,
                    new ParallelOptions { CancellationToken = cancellationToken },
                    (kvp) =>
                    {
                        string fileName = kvp.Key;
                        var labels = kvp.Value;

                        if (!imageManager.ImagePathMap.TryGetValue(fileName, out var imageInfo))
                            return;

                        string labelFileName = Path.GetFileNameWithoutExtension(fileName) + ".txt";
                        string labelFilePath = Path.Combine(outputDirectory, labelFileName);

                        try
                        {
                            int imageWidth = (int)imageInfo.OriginalDimensions.Width;
                            int imageHeight = (int)imageInfo.OriginalDimensions.Height;

                            using StreamWriter writer = new(labelFilePath) { AutoFlush = true };

                            foreach (var label in labels)
                            {
                                if (!label.IsValid()) continue;

                                var sb = new System.Text.StringBuilder();
                                sb.Append(label.ClassId);

                                foreach (var point in label.Points)
                                {
                                    float xNorm = (float)(point.X / imageWidth);
                                    float yNorm = (float)(point.Y / imageHeight);
                                    sb.Append($" {xNorm:F6} {yNorm:F6}");
                                }

                                writer.WriteLine(sb.ToString());
                            }

                            Interlocked.Increment(ref processedFiles);
                            progress?.Report((processedFiles, totalFiles,
                                $"Exporting labels... {processedFiles}/{totalFiles}"));
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine($"Error exporting YOLO-seg labels: {ex.Message}");
                        }
                    });
            }, cancellationToken);
        }

        public void AddAIPolygonLabels(string fileName, List<(List<Point> polygon, int classId)> detectedPolygons)
        {
            if (!labelStorage.ContainsKey(fileName))
                labelStorage.TryAdd(fileName, new List<LabelData>());

            foreach (var detection in detectedPolygons)
            {
                if (detection.polygon == null || detection.polygon.Count < 3)
                    continue;
                
                int classId = detection.classId;
                if (!validClassIds.Contains(classId))
                {
                    classId = defaultClassId;
                }

                var labelCount = labelStorage[fileName].Count + 1;
                var label = new LabelData($"AI Label {labelCount}", detection.polygon, classId);

                labelStorage.AddOrUpdate(
                    fileName,
                    new List<LabelData> { label },
                    (k, existing) =>
                    {
                        existing.Add(label);
                        return existing;
                    });
            }
        }

        public void AddAILabels(string fileName, List<(System.Drawing.Rectangle box, int classId)> detectedBoxes)
        {
            var polygons = detectedBoxes.Select(detection =>
            {
                var polygon = new List<Point>
                {
                    new Point(detection.box.Left, detection.box.Top),
                    new Point(detection.box.Right, detection.box.Top),
                    new Point(detection.box.Right, detection.box.Bottom),
                    new Point(detection.box.Left, detection.box.Bottom)
                };
                return (polygon, detection.classId);
            }).ToList();
            
            AddAIPolygonLabels(fileName, polygons);
        }

        public Task<int> LoadYoloLabelsBatchAsync(
            string labelsDirectory,
            ImageManager imageManager,
            IProgress<(int current, int total, string message)> progress = null,
            CancellationToken cancellationToken = default,
            bool enableParallelProcessing = true)
        {
            return LoadYoloSegLabelsBatchAsync(labelsDirectory, imageManager, progress, cancellationToken, enableParallelProcessing);
        }

        public async Task ExportLabelsBatchAsync(
            string outputDirectory,
            ImageManager imageManager,
            IProgress<(int current, int total, string message)> progress = null,
            CancellationToken cancellationToken = default)
        {
            await ExportSegLabelsBatchAsync(outputDirectory, imageManager, progress, cancellationToken);
        }

        public void ExportLabelsToYolo(string filePath, string imagePath, List<LabelData> labelsToExport)
        {
            ExportLabelsToYoloSeg(filePath, imagePath, labelsToExport);
        }
    }
}