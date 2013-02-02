
(function(){

    var main = $( '#main' );

    var suite = new Benchmark.Suite( 'current', {
        onCycle: function( event ) {
            console.log( event );
            main.empty();
        }
    } );

    // var Scene = phet.scene.Scene;
    // var Node = phet.scene.Node;
    // var Shape = phet.scene.Shape;

    suite.add( 'canvas/context creation', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        // deferred.resolve();
    } ).add( 'simple scene creation and iteration', function() {
        var scene = new phet.scene.Scene( $( '#main' ) );
        var root = scene.root;

        scene.updateScene();
        // deferred.resolve();
    } ).run( {
        'async': true
    } );

    //resultDiv.text( 'boo' );
})();
