set location=c:/Users/zas/Workplace/Dropbox/Programming/ipython/01_WP/General
set filename=PrintHead_Calculations

ipython nbconvert --to html %location%/%filename%.ipynb
wkhtmltopdf %filename%.html %filename%.pdf
del %filename%.html
move %filename%.pdf %location%/%filename%.pdf