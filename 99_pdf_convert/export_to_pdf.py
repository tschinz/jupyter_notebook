# Author: Cloud Cray
# 2014-12-30
# Run this script in the same folder as the notebook(s) you wish to convert
# This will create both an HTML and a PDF file
#
# You'll need wkhtmltopdf (this will keep syntax highlighting, etc)
#   http://wkhtmltopdf.org/downloads.html

import subprocess
import os

WKHTMLTOPDF_PATH = "c:/Program Files/wkhtmltopdf/bin/wkhtmltopdf"


def get_notebook():  # recursive function to show all notebooks in directory
    files = os.listdir(os.getcwd())
    notebooks = [x for x in files if x.lower().endswith(".ipynb")]
    if len(notebooks) == 0:
        return None
    for i in range(len(notebooks)):
        print("{0}: {1}".format(str(i+1), notebooks[i]))
    print("0: None")
    nb_index = input(" Select > ")
    if nb_index.isnumeric():
        if int(nb_index) == 0:
            return None
        elif int(nb_index) > 0 and int(nb_index) < len(notebooks) +1:
            return notebooks[int(nb_index) - 1]
        else:
            print("Invalid!")
            return get_notebook()
    else:
        return get_notebook()


def export_to_html(filename):
    cmd = 'ipython nbconvert --to html "{0}"'
    subprocess.call(cmd.format(filename))
    return filename.replace(".ipynb", ".html")

        
def convert_to_pdf(filename):
    cmd = '"{0}" "{1}" "{2}"'.format(WKHTMLTOPDF_PATH, filename, filename.replace(".html", ".pdf"))
    subprocess.call(cmd)
    return filename.replace(".html", ".pdf")

    
def export_to_pdf(filename):
    fn = export_to_html(filename)
    return convert_to_pdf(fn)
    
    
def main():
    print("Export IPython notebook to PDF")
    print("    Please select a notebook:")
    x = get_notebook()
    if not x:
        print("No notebook selected.")
        return 0
    else:
        fn = export_to_pdf(x)
        print("File exported as:\n\t{0}".format(fn))
        return 1
      
      
main()