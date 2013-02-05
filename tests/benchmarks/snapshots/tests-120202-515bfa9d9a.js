
(function(){
    
    var main = $( '#main' );
    
    sceneBench( 'canvas creation', function( deferred ) {
        var canvas = document.createElement( 'canvas' );
    } );
    
    sceneBench( 'canvas/context creation', function( deferred ) {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
    } );
    
    // sceneBench( 'Fast On Current', function( deferrer ) {
    //     var arb = deferrer;
    //     setTimeout( function() {
    //         arb.resolve();
    //     }, 50 );
    // }, true );
    
    // sceneBench( 'Slow On Current', function( deferred ) {
    //     deferred.resolve();
    // }, true );
    
})();
