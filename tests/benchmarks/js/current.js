
(function(){
    
    var main = $( '#main' );
    
    sceneBench( 'Fast On Current', function( deferred ) {
        deferred.resolve();
    } );
    
    sceneBench( 'Slow On Current', function( deferrer ) {
        var arb = deferrer;
        setTimeout( function() {
            arb.resolve();
        }, 50 );
    } );
    
    sceneBench( 'canvas creation', function( deferred ) {
        var canvas = document.createElement( 'canvas' );
        deferred.resolve();
    } );
    
    sceneBench( 'canvas/context creation', function( deferred ) {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        deferred.resolve();
    } );
})();
