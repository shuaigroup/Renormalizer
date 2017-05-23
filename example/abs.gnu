set term postscript eps enhanced "Helvetiva" 20 color colortext

set output "abssharp0.eps"
#set xrange [2.5:2.75]
plot "abssharp0.out" u ($3)*27.211:4 w lp 
set output

set output "abssharpT.eps"
#set xrange [2.5:2.75]
plot "abssharpT.out" u ($3)*27.211:4 
set output

set output "absgammaT.eps"
#set xrange [2.5:2.75]
plot "absgammaT.out" u ($1)*27.211:2 w lp 
set output
