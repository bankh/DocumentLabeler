# Copyright (c) <2023-Present> Craftnetics Inc., Sinan Bank
# Permission is hereby not granted, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# To check the existing variables in the kernel we can use the following
# In[ ]: %who #magic who command
# Then it will return main_window
# From main_window object we can list the variables and set the variables
# To get variables the following works
# In[ ]: main_window.push_main_window_vars() # Will result in list of objects/vars
# To set the variables the following works
#

from PyQt5.QtWidgets import QApplication, QWidget
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

class ConsoleTab(QWidget):
    def __init__(self,main_window):
        super().__init__()
        self.main_window = main_window
        self.console_widget = ConsoleWidget(self)
        # To fix the height of the Console within the tab
        self.console_widget.setFixedHeight(200) 
        self.console_widget.push_main_window_vars()
    
    def push_vars(self, variable_dict):
        self.console_widget.push_vars(variable_dict)
    
    def push_main_window_vars(self):
        main_window_vars = vars(self.main_window)
        # self.push_vars(main_window_vars)
        self.console_widget.push_vars(main_window_vars)
    
    def update_main_window_var(self, var_name, new_value):
        self.main_window.update_var(var_name, new_value)
        self.console_widget.push_main_window_vars()
    
    def change_main_window_var(self, var_name, new_value):
        if hasattr(self.main_window, var_name):
            setattr(self.main_window, var_name, new_value)
            self.push_main_window_vars()
        else:
            print(f"{var_name} does not exist in the main_window")
    
    def call_main_window_method(self, method_name):
        method = getattr(self.main_window, method_name)
        result = method()
        self.console_widget.push_vars({"result": result})

class ConsoleWidget(RichJupyterWidget):

    def __init__(self, main_window, customBanner=None, *args, **kwargs):
        super(ConsoleWidget, self).__init__(*args, **kwargs)
        self.main_window = main_window
        self.font_size = 10
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel(show_banner=False)
        self.kernel_manager.kernel.gui = 'qt'
        self.kernel_client = self._kernel_manager.client()
        self.kernel_client.start_channels()

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()

        self.exit_requested.connect(stop)

    def push_vars(self, variableDict):
        """
        Given a dictionary containing name / value pairs, push those variables
        to the Jupyter console widget
        """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clear(self):
        """
        Clears the terminal
        """
        self._control.clear()

    def print_text(self, text):
        """
        Prints some plain text to the console
        """
        self._append_plain_text(text)

    def execute_command(self, command):
        """
        Execute a command in the frame of the console widget
        """
        self._execute(command, False)
    
    def push_main_window_vars(self):
        variableDict = {'main_window': self.main_window}
        self.kernel_manager.kernel.shell.push(variableDict)