set location_base=c:/Users/zas/Workplace/Dropbox/Programming/ipython
set location_sub=01_WP/VHDL
set location_output=c:/Users/zas/Workplace/Dropbox/Programming/ipython/99_pdf_convert/output
set filename=Jetmapping_Position_Offset_Calculations

ipython nbconvert --to html %location_base%/%location_sub%/%filename%.ipynb
wkhtmltopdf %filename%.html %filename%.pdf
::del %filename%.html
move %filename%.html %location_output%/%filename%.html
move %filename%.pdf %location_output%/%filename%.pdf