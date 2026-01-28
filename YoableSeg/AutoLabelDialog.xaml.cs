using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using YoableSeg.Managers;

namespace YoableSeg
{
    public class PromptMapping : INotifyPropertyChanged
    {
        private string _prompt;
        private LabelClass _selectedClass;
        private List<LabelClass> _availableClasses;

        public string Prompt
        {
            get => _prompt;
            set { _prompt = value; OnPropertyChanged(nameof(Prompt)); }
        }

        public LabelClass SelectedClass
        {
            get => _selectedClass;
            set { _selectedClass = value; OnPropertyChanged(nameof(SelectedClass)); }
        }

        public List<LabelClass> AvailableClasses
        {
            get => _availableClasses;
            set { _availableClasses = value; OnPropertyChanged(nameof(AvailableClasses)); }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged(string name) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }

    public partial class AutoLabelDialog : Window
    {
        private List<LabelClass> availableClasses;
        private ObservableCollection<PromptMapping> mappings;

        public List<string> Prompts { get; private set; } = new List<string>();
        public Dictionary<string, int> PromptToClassMapping { get; private set; } = new Dictionary<string, int>();
        public float TextThreshold { get; private set; } = 0.25f;
        public float BoxThreshold { get; private set; } = 0.25f;
        public int MaxInstances { get; private set; } = 50;

        public AutoLabelDialog(List<LabelClass> classes)
        {
            InitializeComponent();
            availableClasses = classes ?? new List<LabelClass> { new LabelClass("default", "#E57373", 0) };
            mappings = new ObservableCollection<PromptMapping>();
            MappingListView.ItemsSource = mappings;

            TextThresholdSlider.ValueChanged += (s, e) => TextThresholdValue.Text = e.NewValue.ToString("F2");
            BoxThresholdSlider.ValueChanged += (s, e) => BoxThresholdValue.Text = e.NewValue.ToString("F2");
            MaxInstancesSlider.ValueChanged += (s, e) => MaxInstancesValue.Text = ((int)e.NewValue).ToString();

            TextThresholdSlider.Value = Properties.Settings.Default.TextThreshold;
            BoxThresholdSlider.Value = Properties.Settings.Default.BoxThreshold;
            MaxInstancesSlider.Value = Properties.Settings.Default.MaxInstancesPerImage;

            var savedPrompts = Properties.Settings.Default.LastAutoLabelPrompts;
            if (!string.IsNullOrEmpty(savedPrompts))
            {
                PromptsTextBox.Text = savedPrompts;
            }

            UpdateMapping_Click(null, null);
        }

        private void UpdateMapping_Click(object sender, RoutedEventArgs e)
        {
            var prompts = PromptsTextBox.Text
                .Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(p => p.Trim().ToLower())
                .Where(p => !string.IsNullOrEmpty(p))
                .Distinct()
                .ToList();

            var existingMappings = mappings.ToDictionary(m => m.Prompt, m => m.SelectedClass);
            mappings.Clear();

            foreach (var prompt in prompts)
            {
                var defaultClass = existingMappings.ContainsKey(prompt) 
                    ? existingMappings[prompt] 
                    : availableClasses.FirstOrDefault();

                mappings.Add(new PromptMapping
                {
                    Prompt = prompt,
                    AvailableClasses = availableClasses,
                    SelectedClass = defaultClass
                });
            }
        }

        private void Run_Click(object sender, RoutedEventArgs e)
        {
            if (mappings.Count == 0)
            {
                MessageBox.Show("Please enter at least one prompt.", "No Prompts", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            Prompts = mappings.Select(m => m.Prompt).ToList();
            PromptToClassMapping = mappings
                .Where(m => m.SelectedClass != null)
                .ToDictionary(m => m.Prompt, m => m.SelectedClass.ClassId);

            TextThreshold = (float)TextThresholdSlider.Value;
            BoxThreshold = (float)BoxThresholdSlider.Value;
            MaxInstances = (int)MaxInstancesSlider.Value;

            Properties.Settings.Default.TextThreshold = TextThreshold;
            Properties.Settings.Default.BoxThreshold = BoxThreshold;
            Properties.Settings.Default.MaxInstancesPerImage = MaxInstances;
            Properties.Settings.Default.LastAutoLabelPrompts = PromptsTextBox.Text;
            Properties.Settings.Default.Save();

            DialogResult = true;
            Close();
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
