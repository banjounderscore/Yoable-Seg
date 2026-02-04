using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;

namespace YoableSeg
{
    public partial class AutoLabelDialog : Window
    {
        public List<string> Prompts { get; private set; } = new List<string>();
        public List<string> NegativePrompts { get; private set; } = new List<string>();
        public float ScoreThreshold { get; private set; } = 0.5f;
        public float MaskThreshold { get; private set; } = 0.5f;
        public int MaxInstances { get; private set; } = 50;

        public AutoLabelDialog()
        {
            InitializeComponent();

            ScoreThresholdSlider.ValueChanged += (s, e) => ScoreThresholdValue.Text = e.NewValue.ToString("F2");
            MaskThresholdSlider.ValueChanged += (s, e) => MaskThresholdValue.Text = e.NewValue.ToString("F2");
            MaxInstancesSlider.ValueChanged += (s, e) => MaxInstancesValue.Text = ((int)e.NewValue).ToString();

            ScoreThresholdSlider.Value = Properties.Settings.Default.ScoreThreshold;
            MaskThresholdSlider.Value = Properties.Settings.Default.MaskThreshold;
            MaxInstancesSlider.Value = Properties.Settings.Default.MaxInstancesPerImage;

            var savedPrompts = Properties.Settings.Default.LastAutoLabelPrompts;
            if (!string.IsNullOrEmpty(savedPrompts))
            {
                PromptsTextBox.Text = savedPrompts;
            }

            var savedNegativePrompts = Properties.Settings.Default.LastAutoLabelNegativePrompts;
            if (!string.IsNullOrEmpty(savedNegativePrompts))
            {
                NegativePromptsTextBox.Text = savedNegativePrompts;
            }
        }

        private void Run_Click(object sender, RoutedEventArgs e)
        {
            Prompts = PromptsTextBox.Text
                .Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(p => p.Trim().ToLower())
                .Where(p => !string.IsNullOrEmpty(p))
                .Distinct()
                .ToList();

            if (Prompts.Count == 0)
            {
                MessageBox.Show("Please enter at least one prompt.", "No Prompts", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            NegativePrompts = NegativePromptsTextBox.Text
                .Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(p => p.Trim().ToLower())
                .Where(p => !string.IsNullOrEmpty(p))
                .Distinct()
                .ToList();

            ScoreThreshold = (float)ScoreThresholdSlider.Value;
            MaskThreshold = (float)MaskThresholdSlider.Value;
            MaxInstances = (int)MaxInstancesSlider.Value;

            Properties.Settings.Default.ScoreThreshold = ScoreThreshold;
            Properties.Settings.Default.MaskThreshold = MaskThreshold;
            Properties.Settings.Default.MaxInstancesPerImage = MaxInstances;
            Properties.Settings.Default.LastAutoLabelPrompts = PromptsTextBox.Text;
            Properties.Settings.Default.LastAutoLabelNegativePrompts = NegativePromptsTextBox.Text;
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
