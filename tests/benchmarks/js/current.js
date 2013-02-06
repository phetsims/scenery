
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
    } );
    
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
    } );
    
    // suite.add( 'a', function( deferred ) {
    //     console.log( 'a.run' );
    //     var count = 0;
    //     for( var i = 0; i < 10000; i++ ) {
    //         count *= Math.sin( i );
    //     }
    // }, {
    //     onStart: function( event ) {
    //         console.log( 'a.onStart' );
    //         console.log( event );
    //     },
        
    //     onAbort: function( event ) {
    //         console.log( 'a.onAbort' );
    //         console.log( event );
    //     },
        
    //     onError: function( event ) {
    //         console.log( 'a.onError' );
    //         console.log( event );
    //     },
        
    //     onReset: function( event ) {
    //         console.log( 'a.onReset' );
    //         console.log( event );
    //     },
        
    //     onCycle: function( event ) {
    //         console.log( 'a.onCycle' );
    //         console.log( event );
    //     },
        
    //     onComplete: function( event ) {
    //         console.log( 'a.onComplete' );
    //         console.log( event );
    //     },
        
    //     setup: function( event ) {
    //         console.log( 'a.setup' );
    //     },
        
    //     teardown: function( event ) {
    //         console.log( 'a.teardown' );
    //     }
    // } );
    
    
    
    // marks.run( [
    //     new Benchmark( 'Canvas creation', function() {
    //         document.createElement( 'canvas' );
    //     } ),
        
    //     new Benchmark( 'Canvas/context creation', function() {
    //         var canvas = document.createElement( 'canvas' );
    //         var context = phet.canvas.initCanvas( canvas );
    //     } ),
        
    //     new Benchmark( 'Fast on current version', function() {
            
    //     } );
        
    //     new Benchmark( 'Slow on current version', function() {
    //         var count = 0;
    //         for( int i = 0; i < 100; i++ ) {
    //             count = count * i + Math.sin( i );
    //         }
    //     } );
    // ] );
    
    
    
    // sceneBench( 'canvas creation', function( deferred ) {
    //     var canvas = document.createElement( 'canvas' );
    // } );
    
    // sceneBench( 'canvas/context creation', function( deferred ) {
    //     var canvas = document.createElement( 'canvas' );
    //     var context = phet.canvas.initCanvas( canvas );
    // } );
    
    // sceneBench( 'Fast On Current', function( deferred ) {
    //     deferred.resolve();
    // }, true );
    
    // sceneBench( 'Slow On Current', function( deferrer ) {
    //     var arb = deferrer;
    //     setTimeout( function() {
    //         arb.resolve();
    //     }, 50 );
    // }, true );
    
})();
