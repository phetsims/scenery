

(function(){
    "use strict";
    
    module( 'JSHint' );
    
    // adjust with options from http://www.jshint.com/docs/
    var options = {
        // enforcing options
        curly: true, // brackets for conditionals
        eqeqeq: true,
        
        // relaxing options
        es5: true, // we use ES5 getters and setters for now
        loopfunc: true // we know how not to shoot ourselves in the foot, and this is useful for _.each
    };
    
    var globals = {
        document: false,
        Uint16Array: false,
        Uint32Array: false,
        window: false,
        $: false,
        _: false,
        console: false,
        
        // for DOM.js
        Image: false,
        Blob: false,
        
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



