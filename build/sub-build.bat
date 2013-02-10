:: because conditional loops in batch files are insufficient (http://www.robvanderwoude.com/for.php section 7)
:: so we get to include this gem of a file

:: 't' is set to the file path with regular slashes
set t=%1

:: 'u' will store the file path with backslashes
set u=!t:/=\!

:: append the referenced file to our main JS file
type !u! >> scenery.js
