"""用 nbclient 执行 notebook，把代码 cell 的输出写入。"""
import os
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

os.chdir(os.path.dirname(os.path.abspath(__file__)))

nb_path = "lakeside-full-pipeline.ipynb"
nb = nbformat.read(nb_path, as_version=4)

km = KernelManager(kernel_name="ml4biz-venv")
km.start_kernel(cwd=os.getcwd())

client = NotebookClient(nb, km=km, timeout=600, allow_errors=False)
client.execute()

nbformat.write(nb, nb_path)
print("executed OK, cells:", len(nb.cells))
