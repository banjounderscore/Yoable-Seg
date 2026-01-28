using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;

namespace YoableSeg.Models
{
    /// <summary>
    /// Represents all data for a YoableSeg project (segmentation/polygon labels)
    /// </summary>
    public class ProjectData
    {
        public const string CurrentVersion = "2.0";
        public const string MinimumSupportedVersion = "2.0";

        public string ProjectName { get; set; }
        public string ProjectPath { get; set; }
        public string ProjectFolder { get; set; }
        public DateTime CreatedDate { get; set; }
        public DateTime LastModified { get; set; }
        public string Version { get; set; } = CurrentVersion;

        public static bool IsVersionSupported(string version)
        {
            if (string.IsNullOrEmpty(version)) return false;
            
            if (!System.Version.TryParse(version, out var projectVersion))
                return false;
            
            if (!System.Version.TryParse(MinimumSupportedVersion, out var minVersion))
                return false;

            return projectVersion >= minVersion;
        }

        public static string GetVersionError(string version)
        {
            return $"This project uses an incompatible format (v{version}).\n\n" +
                   $"YoableSeg requires segmentation projects (v{MinimumSupportedVersion}+).\n\n" +
                   "Projects created with the original Yoable (box labels) cannot be opened.\n" +
                   "Please create a new project for segmentation work.";
        }

        // Class definitions for multi-class labeling
        public List<LabelClass> Classes { get; set; } = new List<LabelClass>();

        // Helper methods for class management
        public LabelClass GetClassById(int id)
        {
            return Classes?.FirstOrDefault(c => c.ClassId == id);
        }

        public LabelClass GetDefaultClass()
        {
            return Classes?.FirstOrDefault() ?? new LabelClass("default", "#E57373", 0);
        }

        public int GetNextClassId()
        {
            return Classes?.Any() == true ? Classes.Max(c => c.ClassId) + 1 : 0;
        }

        public bool HasClasses => Classes?.Count > 0;

        // Image references (paths only, no copies)
        public List<ImageReference> Images { get; set; } = new List<ImageReference>();

        // Labels created in-app (stored in project folder)
        // Key: filename, Value: relative path to label file in project folder
        public Dictionary<string, string> AppCreatedLabels { get; set; } = new Dictionary<string, string>();

        // Labels that reference external .txt files (imported labels)
        // Key: filename, Value: full path to external label file
        public Dictionary<string, string> ImportedLabelPaths { get; set; } = new Dictionary<string, string>();

        // UI State
        public Dictionary<string, ImageStatus> ImageStatuses { get; set; } = new Dictionary<string, ImageStatus>();
        public int LastSelectedImageIndex { get; set; } = -1;
        public string CurrentSortMode { get; set; } = "ByName";
        public string CurrentFilterMode { get; set; } = "All";

        // Statistics (optional, for display purposes)
        public int TotalImages => Images?.Count ?? 0;
        public int TotalLabels => (AppCreatedLabels?.Count ?? 0) + (ImportedLabelPaths?.Count ?? 0);
    }

    /// <summary>
    /// Reference to an image file (no actual image data stored)
    /// </summary>
    public class ImageReference
    {
        public string FileName { get; set; }
        public string FullPath { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }

        public ImageReference()
        {
        }

        public ImageReference(string fileName, string fullPath, Size dimensions)
        {
            FileName = fileName;
            FullPath = fullPath;
            Width = dimensions.Width;
            Height = dimensions.Height;
        }

        public Size GetSize()
        {
            return new Size(Width, Height);
        }
    }
}
