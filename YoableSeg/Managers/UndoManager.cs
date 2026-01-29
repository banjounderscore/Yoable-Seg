using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;

namespace YoableSeg.Managers
{
    public interface IUndoAction
    {
        void Undo();
        void Redo();
        string Description { get; }
    }

    public class LabelChangeAction : IUndoAction
    {
        private readonly string fileName;
        private readonly List<LabelData> previousLabels;
        private readonly List<LabelData> newLabels;
        private readonly LabelManager labelManager;
        private readonly Action refreshCallback;

        public string Description => "Label Change";

        public LabelChangeAction(
            string fileName,
            List<LabelData> previousLabels,
            List<LabelData> newLabels,
            LabelManager labelManager,
            Action refreshCallback)
        {
            this.fileName = fileName;
            this.previousLabels = previousLabels.Select(l => new LabelData(l)).ToList();
            this.newLabels = newLabels.Select(l => new LabelData(l)).ToList();
            this.labelManager = labelManager;
            this.refreshCallback = refreshCallback;
        }

        public void Undo()
        {
            labelManager.SaveLabels(fileName, previousLabels);
            refreshCallback?.Invoke();
        }

        public void Redo()
        {
            labelManager.SaveLabels(fileName, newLabels);
            refreshCallback?.Invoke();
        }
    }

    public class ImageDeleteAction : IUndoAction
    {
        private readonly string fileName;
        private readonly string filePath;
        private readonly Size originalDimensions;
        private readonly ImageStatus status;
        private readonly List<LabelData> labels;
        private readonly ImageManager imageManager;
        private readonly LabelManager labelManager;
        private readonly Action<string, bool> updateUICallback;

        public string Description => $"Delete Image: {fileName}";

        public ImageDeleteAction(
            string fileName,
            string filePath,
            Size originalDimensions,
            ImageStatus status,
            List<LabelData> labels,
            ImageManager imageManager,
            LabelManager labelManager,
            Action<string, bool> updateUICallback)
        {
            this.fileName = fileName;
            this.filePath = filePath;
            this.originalDimensions = originalDimensions;
            this.status = status;
            this.labels = labels?.Select(l => new LabelData(l)).ToList() ?? new List<LabelData>();
            this.imageManager = imageManager;
            this.labelManager = labelManager;
            this.updateUICallback = updateUICallback;
        }

        public void Undo()
        {
            imageManager.RestoreImage(fileName, filePath, originalDimensions, status);
            
            if (labels.Count > 0)
            {
                labelManager.SaveLabels(fileName, labels.Select(l => new LabelData(l)).ToList());
            }
            
            updateUICallback?.Invoke(fileName, true);
        }

        public void Redo()
        {
            imageManager.RemoveImage(fileName);
            labelManager.LabelStorage.TryRemove(fileName, out _);
            updateUICallback?.Invoke(fileName, false);
        }
    }

    public class UndoManager
    {
        private static UndoManager _instance;
        public static UndoManager Instance => _instance ??= new UndoManager();

        private readonly Stack<IUndoAction> undoStack = new();
        private readonly Stack<IUndoAction> redoStack = new();
        private const int MaxUndoSteps = 50;

        public event EventHandler UndoRedoPerformed;

        public bool CanUndo => undoStack.Count > 0;
        public bool CanRedo => redoStack.Count > 0;
        public int UndoCount => undoStack.Count;
        public int RedoCount => redoStack.Count;

        public void RecordAction(IUndoAction action)
        {
            undoStack.Push(action);

            if (undoStack.Count > MaxUndoSteps)
            {
                var items = undoStack.ToArray().Reverse().Take(MaxUndoSteps).Reverse().ToArray();
                undoStack.Clear();
                foreach (var item in items)
                {
                    undoStack.Push(item);
                }
            }

            redoStack.Clear();
        }

        public bool Undo()
        {
            if (undoStack.Count == 0) return false;

            var action = undoStack.Pop();
            action.Undo();
            redoStack.Push(action);
            
            UndoRedoPerformed?.Invoke(this, EventArgs.Empty);
            return true;
        }

        public bool Redo()
        {
            if (redoStack.Count == 0) return false;

            var action = redoStack.Pop();
            action.Redo();
            undoStack.Push(action);
            
            UndoRedoPerformed?.Invoke(this, EventArgs.Empty);
            return true;
        }

        public void Clear()
        {
            undoStack.Clear();
            redoStack.Clear();
        }

        public string GetNextUndoDescription()
        {
            return undoStack.Count > 0 ? undoStack.Peek().Description : null;
        }

        public string GetNextRedoDescription()
        {
            return redoStack.Count > 0 ? redoStack.Peek().Description : null;
        }
    }
}
