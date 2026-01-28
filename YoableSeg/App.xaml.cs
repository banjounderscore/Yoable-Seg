using System.Configuration;
using System.Data;
using System.Windows;
using YoableSeg.Managers;

namespace YoableSeg
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            // Show the startup window (combined splash screen and project selector)
            var startupWindow = new StartupWindow();
            startupWindow.Show();
        }

        protected override void OnExit(ExitEventArgs e)
        {
            // Ensure Python service is stopped when application exits
            PythonServiceManager.Instance?.StopService();
            PythonServiceManager.Instance?.Dispose();
            
            base.OnExit(e);
        }
    }
}