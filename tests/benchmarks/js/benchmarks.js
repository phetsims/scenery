
(function(){
    var resultDiv = $( '#profiling-results' );
    
    var suite = new Benchmark.Suite;
    
    // var Scene = phet.scene.Scene;
    // var Node = phet.scene.Node;
    // var Shape = phet.scene.Shape;
    
    var main = $( '#main' );
    
    suite.add( 'canvas/context creation', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
    } ).add( 'simple scene creation and iteration', function() {
        var scene = new phet.scene.Scene( $( '#main' ) );
        var root = scene.root;
        root.layerType = phet.scene.layers.CanvasLayer;
        
        scene.rebuildLayers();
        scene.updateScene();
    }, {
        'onCycle': function() {
            main.empty();
        }
    } ).on( 'cycle', function( event ) {
        console.log( event );
        resultDiv.text( resultDiv.text() + " " + String( event ) );
    } ).run();
    
    //resultDiv.text( 'boo' );
})();