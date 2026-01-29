using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Linq;

namespace YoableSeg
{
    public class LabelData
    {
        public string Name { get; set; }
        public List<Point> Points { get; set; } = new();
        public int ClassId { get; set; } = 0;

        public LabelData(string name, List<Point> points, int classId = 0)
        {
            Name = name;
            Points = points ?? new List<Point>();
            ClassId = classId;
        }

        public LabelData(LabelData source)
        {
            Name = source.Name;
            Points = source.Points.Select(p => new Point(p.X, p.Y)).ToList();
            ClassId = source.ClassId;
        }

        public Rect GetBoundingBox()
        {
            if (Points == null || Points.Count == 0)
                return Rect.Empty;

            double minX = Points.Min(p => p.X);
            double minY = Points.Min(p => p.Y);
            double maxX = Points.Max(p => p.X);
            double maxY = Points.Max(p => p.Y);

            return new Rect(minX, minY, maxX - minX, maxY - minY);
        }

        public bool ContainsPoint(Point testPoint)
        {
            if (Points == null || Points.Count < 3)
                return false;

            bool inside = false;
            int j = Points.Count - 1;

            for (int i = 0; i < Points.Count; i++)
            {
                if ((Points[i].Y < testPoint.Y && Points[j].Y >= testPoint.Y ||
                     Points[j].Y < testPoint.Y && Points[i].Y >= testPoint.Y) &&
                    (Points[i].X <= testPoint.X || Points[j].X <= testPoint.X))
                {
                    if (Points[i].X + (testPoint.Y - Points[i].Y) / (Points[j].Y - Points[i].Y) * (Points[j].X - Points[i].X) < testPoint.X)
                    {
                        inside = !inside;
                    }
                }
                j = i;
            }

            return inside;
        }

        public bool IsValid()
        {
            return Points != null && Points.Count >= 3;
        }

        public void Translate(double dx, double dy)
        {
            for (int i = 0; i < Points.Count; i++)
            {
                Points[i] = new Point(Points[i].X + dx, Points[i].Y + dy);
            }
        }

        public List<Point> ScalePoints(double scaleX, double scaleY)
        {
            return Points.Select(p => new Point(p.X * scaleX, p.Y * scaleY)).ToList();
        }
    }


    public class DrawingCanvas : Canvas
    {
        public event EventHandler LabelsChanged;
        public event Func<Point, Rect?, Task<List<Point>>> RequestSamMask;

        public ImageSource Image { get; set; }
        private Size originalImageDimensions;
        private Rect imageDisplayRect;
        public List<LabelData> Labels { get; set; } = new();
        public LabelData SelectedLabel { get; set; }
        public int CurrentClassId { get; set; } = 0;
        public event EventHandler<int> CurrentClassChanged;
        private List<LabelClass> availableClasses = new List<LabelClass>();

        public void SetAvailableClasses(List<LabelClass> classes)
        {
            availableClasses = classes ?? new List<LabelClass>();

            if (availableClasses.Any() && !availableClasses.Any(c => c.ClassId == CurrentClassId))
            {
                CurrentClassId = availableClasses.First().ClassId;
                OnCurrentClassChanged();
            }
        }

        public string GetCurrentClassColor()
        {
            var currentClass = availableClasses.FirstOrDefault(c => c.ClassId == CurrentClassId);
            return currentClass?.ColorHex ?? "#E57373";
        }

        protected virtual void OnCurrentClassChanged()
        {
            CurrentClassChanged?.Invoke(this, CurrentClassId);
            InvalidateVisual();
        }

        public HashSet<LabelData> SelectedLabels { get; set; } = new();
        private static List<LabelData> clipboard = new();

        public event Action<List<LabelData>> BeforeLabelChange;

        public ListBox LabelListBox { get; set; }
        public Rect CurrentRect { get; set; }
        public bool IsDrawingBox { get; set; }
        public bool IsWaitingForSam { get; set; }
        public Point cursorPosition { get; set; } = new Point(0, 0);
        public Point StartPoint { get; set; }
        public Matrix TransformMatrix { get; set; } = Matrix.Identity;

        private const double ClickDragThreshold = 3.0;
        private double zoomFactor = 1.0;
        private const double zoomStep = 0.1;
        private Point zoomCenter = new Point(0, 0);
        private Matrix transformMatrix = Matrix.Identity;
        private bool mouseDownOnEmpty = false;
        private LabelData mouseDownLabel = null;
        private LabelData previouslySelectedLabel = null;


        public DrawingCanvas()
        {
            Background = Brushes.Transparent;
            ClipToBounds = true;
            Focusable = true;

            MouseWheel += DrawingCanvas_MouseWheel;
            MouseMove += DrawingCanvas_MouseMove;
            MouseUp += DrawingCanvas_MouseUp;
            MouseDown += DrawingCanvas_MouseDown;
            MouseLeave += DrawingCanvas_MouseLeave;
            PreviewKeyDown += DrawingCanvas_PreviewKeyDown;
        }

        protected virtual void OnLabelsChanged()
        {
            LabelsChanged?.Invoke(this, EventArgs.Empty);
        }

        protected virtual void OnBeforeLabelChange()
        {
            var snapshot = Labels.Select(l => new LabelData(l)).ToList();
            BeforeLabelChange?.Invoke(snapshot);
        }

        // Copy selected labels
        private void CopyLabels()
        {
            clipboard.Clear();

            if (SelectedLabels.Count > 0)
            {
                // Copy all selected labels
                foreach (var label in SelectedLabels)
                {
                    clipboard.Add(new LabelData(label));
                }

                // Optional: Show feedback
                if (Application.Current.MainWindow is MainWindow mainWindow)
                {
                    mainWindow.Title = $"YoableSeg - Copied {clipboard.Count} label(s)";
                }
            }
            else if (SelectedLabel != null)
            {
                // Copy single selected label
                clipboard.Add(new LabelData(SelectedLabel));

                // Optional: Show feedback
                if (Application.Current.MainWindow is MainWindow mainWindow)
                {
                    mainWindow.Title = "YoableSeg - Copied 1 label";
                }
            }
        }

        private void PasteLabels()
        {
            if (clipboard.Count == 0) return;

            OnBeforeLabelChange();
            SelectedLabels.Clear();

            double offsetX = 20;
            double offsetY = 20;

            foreach (var clipboardLabel in clipboard)
            {
                var offsetPoints = clipboardLabel.Points.Select(p => new Point(p.X + offsetX, p.Y + offsetY)).ToList();
                var newLabel = new LabelData($"Label {Labels.Count + 1}", offsetPoints, CurrentClassId);

                Labels.Add(newLabel);
                SelectedLabels.Add(newLabel);
            }

            RefreshLabelListBox();
            InvalidateVisual();
            OnLabelsChanged();

            if (Application.Current.MainWindow is MainWindow mainWindow)
            {
                mainWindow.Title = $"YoableSeg - Pasted {clipboard.Count} label(s)";
            }
        }

        // Refresh label list box
        private void RefreshLabelListBox()
        {
            if (LabelListBox != null)
            {
                LabelListBox.Items.Clear();
                foreach (var label in Labels)
                {
                    LabelListBox.Items.Add(label.Name);
                }
            }
        }

        public void ResetZoom()
        {
            zoomFactor = 1.0;
            transformMatrix = Matrix.Identity;
            RenderTransform = new MatrixTransform(transformMatrix);
            CalculateImageDisplayRect();
            InvalidateVisual();
        }

        private Rect ScaleRectToCanvas(Rect originalRect)
        {
            if (Image == null || imageDisplayRect.Width == 0) return originalRect;

            double scaleX = imageDisplayRect.Width / originalImageDimensions.Width;
            double scaleY = imageDisplayRect.Height / originalImageDimensions.Height;

            return new Rect(
                originalRect.X * scaleX + imageDisplayRect.X,
                originalRect.Y * scaleY + imageDisplayRect.Y,
                originalRect.Width * scaleX,
                originalRect.Height * scaleY
            );
        }

        private List<Point> ScalePolygonToCanvas(List<Point> imagePoints)
        {
            if (Image == null || imagePoints == null || imageDisplayRect.Width == 0) return imagePoints;

            double scaleX = imageDisplayRect.Width / originalImageDimensions.Width;
            double scaleY = imageDisplayRect.Height / originalImageDimensions.Height;

            return imagePoints.Select(p => new Point(
                p.X * scaleX + imageDisplayRect.X,
                p.Y * scaleY + imageDisplayRect.Y
            )).ToList();
        }

        private Point ScalePointToImage(Point canvasPoint)
        {
            if (Image == null || imageDisplayRect.Width == 0) return canvasPoint;

            double scaleX = originalImageDimensions.Width / imageDisplayRect.Width;
            double scaleY = originalImageDimensions.Height / imageDisplayRect.Height;
            
            return new Point(
                (canvasPoint.X - imageDisplayRect.X) * scaleX,
                (canvasPoint.Y - imageDisplayRect.Y) * scaleY
            );
        }

        private Rect ScaleRectToImage(Rect canvasRect)
        {
            if (Image == null || imageDisplayRect.Width == 0) return canvasRect;

            double scaleX = originalImageDimensions.Width / imageDisplayRect.Width;
            double scaleY = originalImageDimensions.Height / imageDisplayRect.Height;
            
            return new Rect(
                (canvasRect.X - imageDisplayRect.X) * scaleX,
                (canvasRect.Y - imageDisplayRect.Y) * scaleY,
                canvasRect.Width * scaleX,
                canvasRect.Height * scaleY
            );
        }

        private StreamGeometry CreatePolygonGeometry(List<Point> points)
        {
            var geometry = new StreamGeometry();
            if (points == null || points.Count < 3) return geometry;

            using (var ctx = geometry.Open())
            {
                ctx.BeginFigure(points[0], true, true);
                for (int i = 1; i < points.Count; i++)
                {
                    ctx.LineTo(points[i], true, false);
                }
            }
            geometry.Freeze();
            return geometry;
        }

        public void LoadImage(string imagePath, Size originalDimensions)
        {
            if (string.IsNullOrEmpty(imagePath) || !File.Exists(imagePath)) return;

            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();

            Image = bitmap;
            originalImageDimensions = originalDimensions;

            Labels.Clear();
            SelectedLabel = null;
            SelectedLabels.Clear();
            LabelListBox?.Items.Clear();
            
            CalculateImageDisplayRect();
            InvalidateVisual();
        }

        private void CalculateImageDisplayRect()
        {
            if (Image == null || ActualWidth == 0 || ActualHeight == 0)
            {
                imageDisplayRect = new Rect(0, 0, 0, 0);
                return;
            }

            double imageAspect = originalImageDimensions.Width / originalImageDimensions.Height;
            double containerAspect = ActualWidth / ActualHeight;

            double displayWidth, displayHeight;
            double offsetX = 0, offsetY = 0;

            if (imageAspect > containerAspect)
            {
                displayWidth = ActualWidth;
                displayHeight = ActualWidth / imageAspect;
                offsetY = (ActualHeight - displayHeight) / 2;
            }
            else
            {
                displayHeight = ActualHeight;
                displayWidth = ActualHeight * imageAspect;
                offsetX = (ActualWidth - displayWidth) / 2;
            }

            imageDisplayRect = new Rect(offsetX, offsetY, displayWidth, displayHeight);
        }

        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            base.OnRenderSizeChanged(sizeInfo);
            CalculateImageDisplayRect();
            InvalidateVisual();
        }

        protected override void OnRender(DrawingContext dc)
        {
            base.OnRender(dc);

            if (Image != null && imageDisplayRect.Width > 0 && imageDisplayRect.Height > 0)
            {
                dc.DrawRectangle(Brushes.Black, null, new Rect(0, 0, ActualWidth, ActualHeight));
                dc.DrawImage(Image, imageDisplayRect);
            }

            double thickness = Properties.Settings.Default.LabelThickness;
            double fillOpacity = Properties.Settings.Default.PolygonFillOpacity / 100.0;

            foreach (var label in Labels)
            {
                if (!label.IsValid()) continue;

                var labelClass = availableClasses.FirstOrDefault(c => c.ClassId == label.ClassId);
                string colorHex = labelClass?.ColorHex ?? "#E57373";

                var color = (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString(colorHex);
                bool isSelected = SelectedLabels.Contains(label) || label == SelectedLabel;

                if (isSelected)
                {
                    color.R = (byte)Math.Min(255, color.R + 60);
                    color.G = (byte)Math.Min(255, color.G + 60);
                    color.B = (byte)Math.Min(255, color.B + 60);
                }

                var scaledPoints = ScalePolygonToCanvas(label.Points);
                var geometry = CreatePolygonGeometry(scaledPoints);

                var fillColor = color;
                fillColor.A = (byte)(255 * fillOpacity);
                var fillBrush = new SolidColorBrush(fillColor);
                var strokeBrush = new SolidColorBrush(color);
                var pen = new Pen(strokeBrush, isSelected ? thickness + 1 : thickness);

                dc.DrawGeometry(fillBrush, pen, geometry);
            }

            if (IsDrawingBox)
            {
                string currentColorHex = GetCurrentClassColor();
                var currentColor = (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString(currentColorHex);
                var currentBrush = new SolidColorBrush(currentColor);
                var dashPen = new Pen(currentBrush, thickness);
                dashPen.DashStyle = DashStyles.Dash;
                dc.DrawRectangle(null, dashPen, CurrentRect);
            }

            if (IsWaitingForSam)
            {
                var loadingText = new FormattedText(
                    "Processing...",
                    System.Globalization.CultureInfo.CurrentCulture,
                    FlowDirection.LeftToRight,
                    new Typeface("Segoe UI"),
                    14,
                    Brushes.White,
                    VisualTreeHelper.GetDpi(this).PixelsPerDip);
                dc.DrawText(loadingText, new Point(10, 10));
            }

            if (Properties.Settings.Default.EnableCrosshair)
            {
                DrawCrosshair(dc);
            }
        }


        private void DrawCrosshair(DrawingContext dc)
        {
            if (cursorPosition == new Point(0, 0) || imageDisplayRect.Width == 0) return;

            var colorBrush = (SolidColorBrush)(new BrushConverter().ConvertFrom(Properties.Settings.Default.CrosshairColor));
            double crosshairSize = Properties.Settings.Default.CrosshairSize;

            Pen crosshairPen = new Pen(colorBrush, crosshairSize);
            dc.DrawLine(crosshairPen, new Point(imageDisplayRect.Left, cursorPosition.Y), new Point(imageDisplayRect.Right, cursorPosition.Y));
            dc.DrawLine(crosshairPen, new Point(cursorPosition.X, imageDisplayRect.Top), new Point(cursorPosition.X, imageDisplayRect.Bottom));
        }

        private void DrawingCanvas_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            // PRIORITY 1: If shift is held and labels are selected, cycle through classes
            if ((Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) && 
                (SelectedLabel != null || SelectedLabels.Count > 0) && 
                availableClasses.Count > 1)
            {
                // Get the current class of the selected label
                int currentClassId = SelectedLabel?.ClassId ?? SelectedLabels.FirstOrDefault()?.ClassId ?? CurrentClassId;
                int currentIndex = availableClasses.FindIndex(c => c.ClassId == currentClassId);
                
                if (currentIndex < 0) currentIndex = 0;

                if (e.Delta > 0) // Scroll up - previous class
                {
                    currentIndex = currentIndex <= 0 ? availableClasses.Count - 1 : currentIndex - 1;
                }
                else // Scroll down - next class
                {
                    currentIndex = currentIndex >= availableClasses.Count - 1 ? 0 : currentIndex + 1;
                }

                OnBeforeLabelChange();
                var newClassId = availableClasses[currentIndex].ClassId;
                
                // Update all selected labels
                if (SelectedLabels.Count > 0)
                {
                    foreach (var label in SelectedLabels)
                    {
                        label.ClassId = newClassId;
                    }
                }
                else if (SelectedLabel != null)
                {
                    SelectedLabel.ClassId = newClassId;
                }
                
                // Refresh UI to show new class colors
                if (Application.Current.MainWindow is MainWindow mainWindow)
                {
                    mainWindow.RefreshLabelListFromCanvas();
                }
                
                OnLabelsChanged();
                InvalidateVisual();
                e.Handled = true;
                return;
            }

            // PRIORITY 2: If we're currently drawing a box, cycle through classes
            if (IsDrawingBox && availableClasses.Count > 1)
            {
                int currentIndex = availableClasses.FindIndex(c => c.ClassId == CurrentClassId);

                if (e.Delta > 0) // Scroll up - previous class
                {
                    currentIndex = currentIndex <= 0 ? availableClasses.Count - 1 : currentIndex - 1;
                }
                else // Scroll down - next class
                {
                    currentIndex = currentIndex >= availableClasses.Count - 1 ? 0 : currentIndex + 1;
                }

                CurrentClassId = availableClasses[currentIndex].ClassId;
                OnCurrentClassChanged();
                e.Handled = true;
                return;
            }

            if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
            {
                // Existing zoom functionality
                double oldZoomFactor = zoomFactor;
                zoomCenter = e.GetPosition(this);

                if (e.Delta > 0)
                {
                    zoomFactor *= 1 + zoomStep;
                }
                else if (e.Delta < 0)
                {
                    zoomFactor /= 1 + zoomStep;
                }

                zoomFactor = Math.Max(1.0, Math.Min(zoomFactor, 5.0));

                if (zoomFactor == 1.0)
                {
                    transformMatrix = Matrix.Identity;
                    RenderTransform = new MatrixTransform(transformMatrix);
                    InvalidateVisual();
                    return;
                }

                Point newZoomCenter = e.GetPosition(this);
                double offsetX = newZoomCenter.X - zoomCenter.X;
                double offsetY = newZoomCenter.Y - zoomCenter.Y;

                transformMatrix.Translate(-zoomCenter.X, -zoomCenter.Y);
                transformMatrix.Scale(zoomFactor / oldZoomFactor, zoomFactor / oldZoomFactor);
                transformMatrix.Translate(zoomCenter.X - offsetX, zoomCenter.Y - offsetY);

                RenderTransform = new MatrixTransform(transformMatrix);
                InvalidateVisual();
            }
            else
            {
                // Switch images when CTRL is NOT held
                if (Application.Current.MainWindow is MainWindow mainWindow)
                {
                    ListBox imageListBox = mainWindow.ImageListBox;
                    if (imageListBox.Items.Count == 0) return;

                    int newIndex = imageListBox.SelectedIndex + (e.Delta > 0 ? -1 : 1);

                    if (newIndex >= 0 && newIndex < imageListBox.Items.Count)
                    {
                        imageListBox.SelectedIndex = newIndex;
                        imageListBox.ScrollIntoView(imageListBox.SelectedItem);
                    }
                }
            }

            e.Handled = true;
        }

        private void DrawingCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            cursorPosition = e.GetPosition(this);

            if (imageDisplayRect.Contains(cursorPosition) && Properties.Settings.Default.EnableCrosshair)
            {
                Cursor = Cursors.None;
            }
            else
            {
                Cursor = Cursors.Arrow;
            }

            if (mouseDownOnEmpty && !IsDrawingBox)
            {
                double distance = Math.Sqrt(
                    Math.Pow(cursorPosition.X - StartPoint.X, 2) +
                    Math.Pow(cursorPosition.Y - StartPoint.Y, 2));

                if (distance > ClickDragThreshold)
                {
                    IsDrawingBox = true;
                }
            }

            if (IsDrawingBox)
            {
                Point clampedCursor = ClampToBounds(cursorPosition);
                CurrentRect = new Rect(
                    Math.Min(StartPoint.X, clampedCursor.X),
                    Math.Min(StartPoint.Y, clampedCursor.Y),
                    Math.Abs(clampedCursor.X - StartPoint.X),
                    Math.Abs(clampedCursor.Y - StartPoint.Y)
                );
                InvalidateVisual();
                return;
            }

            InvalidateVisual();
        }

        private Point ClampToBounds(Point point)
        {
            if (Image == null || imageDisplayRect.Width == 0) return point;
            
            double clampedX = Math.Max(imageDisplayRect.Left, Math.Min(imageDisplayRect.Right, point.X));
            double clampedY = Math.Max(imageDisplayRect.Top, Math.Min(imageDisplayRect.Bottom, point.Y));
            return new Point(clampedX, clampedY);
        }

        private async void DrawingCanvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            Mouse.Capture(null);

            if (IsWaitingForSam)
            {
                mouseDownOnEmpty = false;
                IsDrawingBox = false;
                CurrentRect = new Rect(0, 0, 0, 0);
                InvalidateVisual();
                return;
            }

            var endPoint = e.GetPosition(this);
            double distance = Math.Sqrt(
                Math.Pow(endPoint.X - StartPoint.X, 2) +
                Math.Pow(endPoint.Y - StartPoint.Y, 2));

            bool wasDrawingBox = IsDrawingBox && CurrentRect.Width > 1 && CurrentRect.Height > 1;
            Rect savedRect = CurrentRect;

            IsDrawingBox = false;
            CurrentRect = new Rect(0, 0, 0, 0);
            mouseDownOnEmpty = false;

            if (distance < ClickDragThreshold)
            {
                HandleLabelClick(endPoint);
                InvalidateVisual();
                return;
            }

            if (wasDrawingBox)
            {
                var imageRect = ScaleRectToImage(savedRect);
                await RequestSamMaskFromBoxAsync(imageRect);
            }

            InvalidateVisual();
        }

        private void HandleLabelClick(Point mousePos)
        {
            bool ctrlPressed = Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl);
            var imageMousePos = ScalePointToImage(mousePos);

            var overlappingLabels = new List<LabelData>();
            for (int i = Labels.Count - 1; i >= 0; i--)
            {
                var label = Labels[i];
                if (!label.IsValid()) continue;

                if (label.ContainsPoint(imageMousePos))
                {
                    overlappingLabels.Add(label);
                }
            }

            if (overlappingLabels.Count == 0)
            {
                previouslySelectedLabel = null;
                return;
            }

            LabelData clickedLabel = overlappingLabels[0];

            if (!ctrlPressed && previouslySelectedLabel != null && overlappingLabels.Contains(previouslySelectedLabel))
            {
                if (overlappingLabels.Count == 1)
                {
                    previouslySelectedLabel = null;
                    return;
                }
                else
                {
                    int currentIndex = overlappingLabels.IndexOf(previouslySelectedLabel);
                    int nextIndex = (currentIndex + 1) % overlappingLabels.Count;
                    clickedLabel = overlappingLabels[nextIndex];
                }
            }
            
            previouslySelectedLabel = null;

            if (ctrlPressed)
            {
                if (SelectedLabels.Contains(clickedLabel))
                {
                    SelectedLabels.Remove(clickedLabel);
                    SelectedLabel = SelectedLabels.FirstOrDefault();
                }
                else
                {
                    SelectedLabels.Add(clickedLabel);
                    SelectedLabel = clickedLabel;
                }
            }
            else
            {
                SelectedLabels.Clear();
                SelectedLabels.Add(clickedLabel);
                SelectedLabel = clickedLabel;
            }

            if (LabelListBox != null)
            {
                if (!ctrlPressed && SelectedLabel != null)
                {
                    LabelListBox.SelectedItem = SelectedLabel.Name;
                }
                else if (ctrlPressed && SelectedLabels.Count == 1 && SelectedLabel != null)
                {
                    LabelListBox.SelectedItem = SelectedLabel.Name;
                }
                else if (SelectedLabels.Count == 0 || SelectedLabels.Count > 1)
                {
                    LabelListBox.SelectedItem = null;
                }
            }
        }

        private async Task RequestSamMaskFromBoxAsync(Rect imageBox)
        {
            if (RequestSamMask == null)
            {
                System.Diagnostics.Debug.WriteLine("RequestSamMask event is null!");
                return;
            }

            IsWaitingForSam = true;
            InvalidateVisual();

            try
            {
                var centerPoint = new Point(
                    imageBox.X + imageBox.Width / 2,
                    imageBox.Y + imageBox.Height / 2);

                var polygonPoints = await RequestSamMask(centerPoint, imageBox);
                
                if (polygonPoints == null)
                {
                    System.Diagnostics.Debug.WriteLine("SAM returned null - SAM3 may not be running");
                }
                else if (polygonPoints.Count < 3)
                {
                    System.Diagnostics.Debug.WriteLine($"SAM returned too few points: {polygonPoints.Count}");
                }
                else
                {
                    OnBeforeLabelChange();
                    var newLabel = new LabelData($"Label {Labels.Count + 1}", polygonPoints, CurrentClassId);
                    Labels.Add(newLabel);

                    if (Application.Current.MainWindow is MainWindow mainWindow)
                    {
                        mainWindow.RefreshLabelListFromCanvas();
                    }

                    OnLabelsChanged();
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"SAM mask request failed: {ex.Message}");
                System.Diagnostics.Debug.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            finally
            {
                IsWaitingForSam = false;
                InvalidateVisual();
            }
        }


        private void DrawingCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            Focus();
            Keyboard.Focus(this);

            Point mousePos = e.GetPosition(this);
            StartPoint = mousePos;
            mouseDownOnEmpty = true;
            CurrentRect = new Rect(StartPoint, new Size(0, 0));

            var imageMousePos = ScalePointToImage(mousePos);
            mouseDownLabel = null;
            for (int i = Labels.Count - 1; i >= 0; i--)
            {
                var label = Labels[i];
                if (label.IsValid() && label.ContainsPoint(imageMousePos))
                {
                    mouseDownLabel = label;
                    break;
                }
            }

            previouslySelectedLabel = SelectedLabel;
            SelectedLabel = null;
            SelectedLabels.Clear();
            if (LabelListBox != null)
                LabelListBox.SelectedItem = null;

            Mouse.Capture(this);
            InvalidateVisual();
        }

        private void DrawingCanvas_MouseLeave(object sender, MouseEventArgs e)
        {
            Cursor = Cursors.Arrow;
            InvalidateVisual();
        }

        private void DrawingCanvas_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
            {
                switch (e.Key)
                {
                    case Key.C:
                        CopyLabels();
                        e.Handled = true;
                        return;

                    case Key.V:
                        PasteLabels();
                        e.Handled = true;
                        return;

                    case Key.A:
                        SelectedLabels.Clear();
                        foreach (var label in Labels)
                        {
                            SelectedLabels.Add(label);
                        }
                        SelectedLabel = Labels.FirstOrDefault();
                        InvalidateVisual();
                        e.Handled = true;
                        return;
                }
            }

            if ((SelectedLabel != null && Labels.Contains(SelectedLabel)) || SelectedLabels.Count > 0)
            {
                switch (e.Key)
                {
                    case Key.Delete:
                        OnBeforeLabelChange();
                        if (SelectedLabels.Count > 0)
                        {
                            var labelsToDelete = SelectedLabels.ToList();
                            foreach (var label in labelsToDelete)
                            {
                                Labels.Remove(label);
                            }
                            SelectedLabels.Clear();
                        }
                        else if (SelectedLabel != null)
                        {
                            Labels.Remove(SelectedLabel);
                        }
                        SelectedLabel = null;

                        if (Application.Current.MainWindow is MainWindow mainWindow)
                        {
                            mainWindow.RefreshLabelListFromCanvas();
                        }

                        OnLabelsChanged();
                        InvalidateVisual();
                        e.Handled = true;
                        break;
                }
            }
        }
    }
}