using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace QQGeminiAssistant
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string venvPythonW = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
                string venvPython = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");
                string guiScript = Path.Combine(baseDir, "gui_launcher.py");

                if (!File.Exists(guiScript))
                {
                    MessageBox.Show(
                        "未找到主界面脚本 gui_launcher.py！\n请确保程序位于项目根目录下运行。",
                        "启动错误",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return;
                }

                string pythonExe = null;
                if (File.Exists(venvPythonW))
                {
                    pythonExe = venvPythonW;
                }
                else if (File.Exists(venvPython))
                {
                    pythonExe = venvPython;
                }
                else
                {
                    pythonExe = "pythonw";
                }

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonExe;
                psi.Arguments = "\"" + guiScript + "\"";
                psi.WorkingDirectory = baseDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;

                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "启动 QQ Gemini 智能助手失败：\n" + ex.Message,
                    "启动异常",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
