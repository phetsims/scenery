
(function(){
    
    var main = $( '#main' );
    
    var suite = benchmarkTimer.currentSuite;
    
    suite.add( 'Canvas creation', function() {
        document.createElement( 'canvas' );
    } );
    
    suite.add( 'Canvas/context creation', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
    } );
    
    suite.add( 'Fast on current version', function() {
        
    } );
    
    suite.add( 'Slow on current version', function() {
        var count = 0;
        for( var i = 0; i < 100; i++ ) {
            count = count * i + Math.sin( i );
        }
    } );
    
    suite.add( 'Fast deferred on current version', function( deferrer ) {
        if( !deferrer ) {
            console.log( 'no deferrer: ' + deferrer );
            console.log( 'fast current' );
        }
        deferrer.resolve();
    }, { defer: true } );
    
    suite.add( 'Slow deferred on current version', function( deferrer ) {
        if( !deferrer ) {
            console.log( 'no deferrer: ' + deferrer );
            console.log( 'slow current' );
        }
        setTimeout( function() {
            deferrer.resolve();
        }, 1000 );
    }, { defer: true } );
    
    suite.add( 'Control Bench A', function() {
        var count = 0;
        for( var i = 0; i < 100; i++ ) {
            count = count * i + Math.sin( i );
        }
    }, { delay: 0.1, minTime: 1 } );
    
    suite.add( 'Control Bench B', function() {
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
    }, { delay: 0.1, minTime: 1 } );
    
})();
