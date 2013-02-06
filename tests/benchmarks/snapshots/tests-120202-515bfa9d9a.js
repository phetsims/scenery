
(function(){
    
    var main = $( '#main' );
    
    benchmarkTimer.add( 'Canvas creation', function() {
        document.createElement( 'canvas' );
    } );
    
    benchmarkTimer.add( 'Canvas/context creation', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
    } );
    
    benchmarkTimer.add( 'Fast on current version', function() {
        var count = 0;
        for( var i = 0; i < 100; i++ ) {
            count = count * i + Math.sin( i );
        }
    } );
    
    benchmarkTimer.add( 'Slow on current version', function() {
        
    } );
    
    benchmarkTimer.add( 'Fast deferred on current version', function( deferrer ) {
        if( !deferrer ) {
            console.log( 'no deferrer: ' + deferrer );
            console.log( 'fast old' );
        }
        setTimeout( function() {
            deferrer.resolve();
        }, 1000 );
    }, { defer: true } );
    
    benchmarkTimer.add( 'Slow deferred on current version', function( deferrer ) {
        if( !deferrer ) {
            console.log( 'no deferrer: ' + deferrer );
            console.log( 'slow old' );
        }
        deferrer.resolve();
    }, { defer: true } );
    
    benchmarkTimer.add( 'Control Bench A', function() {
        var count = 0;
        for( var i = 0; i < 100; i++ ) {
            count = count * i + Math.sin( i );
        }
    } );
    
    benchmarkTimer.add( 'Control Bench B', function() {
        var count = 0;
        for( var i = 0; i < 100; i++ ) {
            count = count * i + Math.sin( i );
        }
    }, {
        setup: function() {
            var count = 0;
            for( var i = 0; i < 10000; i++ ) {
                count = count * i + Math.sin( i );
            }
        },
        
        teardown: function() {
            var count = 0;
            for( var i = 0; i < 10000; i++ ) {
                count = count * i + Math.sin( i );
            }
        }
    } );
    
})();
