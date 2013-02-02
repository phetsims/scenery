
(function(){
    
    var main = $( '#main' );
    
    sceneBench( 'Fast On Current', function( deferred ) {
        var f = 1;
        for( var i = 1; i < 100; i++ ) {
            f = f * i;
        }
        deferred.resolve();
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
