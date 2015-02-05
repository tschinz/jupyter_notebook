set location_base=c:/Users/zas/Workplace/Dropbox/Programming/ipython
set location_sub=01_WP/General
set location_output=c:/Users/zas/Workplace/Dropbox/Programming/ipython/99_pdf_convert/output
set filename=Var_Printhead_Resolution

ipython nbconvert --to latex --post pdf %location_base%/%location_sub%/%filename%.ipynb

::del %filename%.html
move %filename%.tex %location_output%/%filename%.tex
move %filename%.pdf %location_output%/%filename%.pdf

