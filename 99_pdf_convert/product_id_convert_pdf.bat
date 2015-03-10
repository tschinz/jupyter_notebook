set location_base=c:/Users/zas/Workplace/Dropbox/Programming/ipython
set location_sub=01_WP/General
set location_output=c:/Users/zas/Workplace/Dropbox/Programming/ipython/99_pdf_convert/output
set filename=ProductID

ipython nbconvert --to html %location_base%/%location_sub%/%filename%.ipynb
wkhtmltopdf %filename%.html %filename%.pdf
::del %filename%.html
move %filename%.html %location_output%/%filename%.html
move %filename%.pdf %location_output%/%filename%.pdf

