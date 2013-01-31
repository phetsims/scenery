

(function(){
    "use strict";
    
    module( 'JSHint' );
    
    // adjust with options from http://www.jshint.com/docs/
    var options = {};
    var globals = {
        document: false,
        Uint16Array: false,
        Uint32Array: false,
        Float32Array: true // we actually polyfill this, so allow it to be set
    };
    
    qHint.sendRequest( '../../build/file-list.txt', function( req ) {
        console.log( req );
        
        test( 'File list OK', function() {
            equal( req.status, 200 );
        } );
        
        var filenames = req.responseText.split( /\r?\n/ );
        
        _.each( filenames, function( filename ) {
            if( filename ) {
                qHint( filename, '../../' + filename + '?random=' + Math.random().toFixed( 20 ), options, globals );
            }
        } );
    } );
    
})();



