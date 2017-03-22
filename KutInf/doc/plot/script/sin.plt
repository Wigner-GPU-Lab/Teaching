set term postscript eps size 1280,720
set out "sin.eps"

set title "Original function"

set autoscale

set xrange [ -pi : pi ]
set yrange [ -1 : 1 ]

set xlabel "x"
set ylabel "sin(x)"

plot "sin.dat" using 1:2 w lines